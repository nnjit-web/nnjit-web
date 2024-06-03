from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from tvm import tir
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import IterVar, PrimExpr, Var
from tvm.tir.analysis import undefined_vars
from tvm.tir.schedule.schedule import BlockRV

from ..base import ScheduleRule, analysis


def _collect_producers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def auto_inline_producers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def make_iter_fusion_index_map(
    traits: List[IterTrait],
    kind_order: List[IterKind],
) -> tir.IndexMap:
    fused_iters: Dict[IterKind, PrimExpr] = {}
    input_iters: List[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", trait.extent.dtype)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: List[tir.PrimExpr] = [
        fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
    ]

    return tir.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: tir.Block) -> Optional[Tuple[List[IterTrait]]]:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: List[Range]) -> Set[Var]:
        axes: Set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.min)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: Dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in A_axes]
    B_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in B_axes]
    C_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in C_axes]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: tir.Block) -> Optional[Tuple[tir.IndexMap, ...]]:
    """Get index maps for the block

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    A_index_map = make_iter_fusion_index_map(
        A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K]
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K]
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J]
    )
    matmul_index_map = make_iter_fusion_index_map(
        block_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K]
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


class Matmul(ScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vthread_x: int = 1
        vthread_y: int = 1
        micro_size_x: int = 4
        micro_size_y: int = 4
        micro_size_k: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name == "webgpu":
            '''
            return Matmul.Config(
                block_size_x=8,
                block_size_y=8,
                vthread_x=4,
                vthread_y=4,
                micro_size_x=1,
                micro_size_y=1,
                micro_size_k=32,
                vector_size=4,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
            '''
            return Matmul.Config(
                block_size_x=16,
                block_size_y=8,
                vthread_x=8,
                vthread_y=8,
                micro_size_x=1,
                micro_size_y=1,
                micro_size_k=4,
                vector_size=4,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        else:
            return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 1. Check Tensor Core support
        # ...

        # Step 2. Get schedule config.
        config = self.get_configs(target)

        # Step 3: 
        y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y
        x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, config.micro_size_k],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, config.micro_size_k],
            )
            batch, x, y, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_size_y, config.micro_size_y]
        )
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_size_x, config.micro_size_x]
        )
        ko, ki = sch.split(k, factors=[None, config.micro_size_k])

        sch.reorder(by, bx, vy, vx, ko, ty, tx, ki, yi, xi)
        
        by = sch.fuse(batch, by)

        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=config.unroll)
        sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)

        auto_inline_producers(sch, main_block)

        auto_inline_consumers(sch, l2g)

        remaining_consumers = sch.get_consumers(l2g)

        if len(remaining_consumers) != 0:
            # Some blocks have failed to be inlined to the producer cache-write stage.
            # This could be due to another producer block that has not been scheduled.
            for c in remaining_consumers:
                for p in sch.get_producers(c):
                    if sch.get(p) != sch.get(l2g):
                        sch.compute_inline(p)

            # Try inlining into the cache-write stage again, this time it should succeed.
            auto_inline_consumers(sch, l2g)

        msg = "There are some consumers of the cache-write stage that are not properly inlined."
        assert len(sch.get_consumers(l2g)) == 0, msg

        sch.decompose_reduction(main_block, ko)

        return sch