"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CachedCallStack = exports.Memory = void 0;
const support_1 = require("./support");
/**
 * Wasm Memory wrapper to perform JS side raw memory access.
 */
class Memory {
    constructor(memory) {
        this.wasm32 = true;
        this.memory = memory;
        this.buffer = this.memory.buffer;
        this.viewU8 = new Uint8Array(this.buffer);
        this.viewU16 = new Uint16Array(this.buffer);
        this.viewI32 = new Int32Array(this.buffer);
        this.viewU32 = new Uint32Array(this.buffer);
        this.viewF32 = new Float32Array(this.buffer);
        this.viewF64 = new Float64Array(this.buffer);
    }
    loadU8(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        return this.viewU8[ptr >> 0];
    }
    loadU16(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        return this.viewU16[ptr >> 1];
    }
    loadU32(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        return this.viewU32[ptr >> 2];
    }
    loadI32(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        return this.viewI32[ptr >> 2];
    }
    loadI64(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        const base = ptr >> 2;
        // assumes little endian, for now truncate high.
        return this.viewI32[base];
    }
    loadF32(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        return this.viewF32[ptr >> 2];
    }
    loadF64(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        return this.viewF64[ptr >> 3];
    }
    loadPointer(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        if (this.wasm32) {
            return this.loadU32(ptr);
        }
        else {
            return this.loadI64(ptr);
        }
    }
    loadUSize(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        if (this.wasm32) {
            return this.loadU32(ptr);
        }
        else {
            return this.loadI64(ptr);
        }
    }
    sizeofPtr() {
        return this.wasm32 ? 4 /* SizeOf.I32 */ : 8 /* SizeOf.I64 */;
    }
    /**
     * Load raw bytes from ptr.
     * @param ptr The head address
     * @param numBytes The number
     */
    loadRawBytes(ptr, numBytes) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        const result = new Uint8Array(numBytes);
        result.set(this.viewU8.slice(ptr, ptr + numBytes));
        return result;
    }
    /**
     * Load TVMByteArray from ptr.
     *
     * @param ptr The address of the header.
     */
    loadTVMBytes(ptr) {
        const data = this.loadPointer(ptr);
        const length = this.loadUSize(ptr + this.sizeofPtr());
        return this.loadRawBytes(data, length);
    }
    /**
     * Load null-terminated C-string from ptr.
     * @param ptr The head address
     */
    loadCString(ptr) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        // NOTE: the views are still valid for read.
        const ret = [];
        let ch = 1;
        while (ch != 0) {
            ch = this.viewU8[ptr];
            if (ch != 0) {
                ret.push(String.fromCharCode(ch));
            }
            ++ptr;
        }
        return ret.join("");
    }
    /**
     * Store raw bytes to the ptr.
     * @param ptr The head address.
     * @param bytes The bytes content.
     */
    storeRawBytes(ptr, bytes) {
        if (this.buffer != this.memory.buffer) {
            this.updateViews();
        }
        this.viewU8.set(bytes, ptr);
    }
    /**
     * Update memory view after the memory growth.
     */
    updateViews() {
        this.buffer = this.memory.buffer;
        this.viewU8 = new Uint8Array(this.buffer);
        this.viewU16 = new Uint16Array(this.buffer);
        this.viewI32 = new Int32Array(this.buffer);
        this.viewU32 = new Uint32Array(this.buffer);
        this.viewF32 = new Float32Array(this.buffer);
        this.viewF64 = new Float64Array(this.buffer);
    }
}
exports.Memory = Memory;
/**
 * Auxiliary call stack for the FFI calls.
 *
 * Lifecyle of a call stack.
 * - Calls into allocXX to allocate space, mixed with storeXXX to store data.
 * - Calls into ptrFromOffset, no further allocation(as ptrFromOffset can change),
 *   can still call into storeXX
 * - Calls into commitToWasmMemory once.
 * - reset.
 */
class CachedCallStack {
    constructor(memory, allocSpace, freeSpace) {
        /** List of temporay arguments that can be disposed during reset. */
        this.tempArgs = [];
        this.stackTop = 0;
        this.basePtr = 0;
        this.addressToSetTargetValue = [];
        const initCallStackSize = 128;
        this.memory = memory;
        this.cAllocSpace = allocSpace;
        this.cFreeSpace = freeSpace;
        this.buffer = new ArrayBuffer(initCallStackSize);
        this.basePtr = this.cAllocSpace(initCallStackSize);
        this.viewU8 = new Uint8Array(this.buffer);
        this.viewI32 = new Int32Array(this.buffer);
        this.viewU32 = new Uint32Array(this.buffer);
        this.viewF64 = new Float64Array(this.buffer);
        this.updateViews();
    }
    dispose() {
        if (this.basePtr != 0) {
            this.cFreeSpace(this.basePtr);
            this.basePtr = 0;
        }
    }
    /**
     * Rest the call stack so that it can be reused again.
     */
    reset() {
        this.stackTop = 0;
        (0, support_1.assert)(this.addressToSetTargetValue.length == 0);
        while (this.tempArgs.length != 0) {
            this.tempArgs.pop().dispose();
        }
    }
    /**
     * Commit all the cached data to WasmMemory.
     * This function can only be called once.
     * No further store function should be called.
     *
     * @param nbytes Number of bytes to be stored.
     */
    commitToWasmMemory(nbytes = this.stackTop) {
        // commit all pointer values.
        while (this.addressToSetTargetValue.length != 0) {
            const [targetOffset, valueOffset] = this.addressToSetTargetValue.pop();
            this.storePtr(targetOffset, this.ptrFromOffset(valueOffset));
        }
        this.memory.storeRawBytes(this.basePtr, this.viewU8.slice(0, nbytes));
    }
    /**
     * Allocate space by number of bytes
     * @param nbytes Number of bytes.
     * @note This function always allocate space that aligns to 64bit.
     */
    allocRawBytes(nbytes) {
        // always aligns to 64bit
        nbytes = ((nbytes + 7) >> 3) << 3;
        if (this.stackTop + nbytes > this.buffer.byteLength) {
            const newSize = Math.max(this.buffer.byteLength * 2, this.stackTop + nbytes);
            const oldU8 = this.viewU8;
            this.buffer = new ArrayBuffer(newSize);
            this.updateViews();
            this.viewU8.set(oldU8);
            if (this.basePtr != 0) {
                this.cFreeSpace(this.basePtr);
            }
            this.basePtr = this.cAllocSpace(newSize);
        }
        const retOffset = this.stackTop;
        this.stackTop += nbytes;
        return retOffset;
    }
    /**
     * Allocate space for pointers.
     * @param count Number of pointers.
     * @returns The allocated pointer array.
     */
    allocPtrArray(count) {
        return this.allocRawBytes(this.memory.sizeofPtr() * count);
    }
    /**
     * Get the real pointer from offset values.
     * Note that the returned value becomes obsolete if alloc is called on the stack.
     * @param offset The allocated offset.
     */
    ptrFromOffset(offset) {
        return this.basePtr + offset;
    }
    // Store APIs
    storePtr(offset, value) {
        if (this.memory.wasm32) {
            this.storeU32(offset, value);
        }
        else {
            this.storeI64(offset, value);
        }
    }
    storeUSize(offset, value) {
        if (this.memory.wasm32) {
            this.storeU32(offset, value);
        }
        else {
            this.storeI64(offset, value);
        }
    }
    storeI32(offset, value) {
        this.viewI32[offset >> 2] = value;
    }
    storeU32(offset, value) {
        this.viewU32[offset >> 2] = value;
    }
    storeI64(offset, value) {
        // For now, just store as 32bit
        // NOTE: wasm always uses little endian.
        const low = value & 0xffffffff;
        const base = offset >> 2;
        this.viewI32[base] = low;
        this.viewI32[base + 1] = 0;
    }
    storeF64(offset, value) {
        this.viewF64[offset >> 3] = value;
    }
    storeRawBytes(offset, bytes) {
        this.viewU8.set(bytes, offset);
    }
    /**
     * Allocate then set C-String pointer to the offset.
     * This function will call into allocBytes to allocate necessary data.
     * The address won't be set immediately(because the possible change of basePtr)
     * and will be filled when we commit the data.
     *
     * @param offset The offset to set ot data pointer.
     * @param data The string content.
     */
    allocThenSetArgString(offset, data) {
        const strOffset = this.allocRawBytes(data.length + 1);
        this.storeRawBytes(strOffset, (0, support_1.StringToUint8Array)(data));
        this.addressToSetTargetValue.push([offset, strOffset]);
    }
    /**
     * Allocate then set the argument location with a TVMByteArray.
     * Allocate new temporary space for bytes.
     *
     * @param offset The offset to set ot data pointer.
     * @param data The string content.
     */
    allocThenSetArgBytes(offset, data) {
        // Note: size of size_t equals sizeof ptr.
        const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
        const dataOffset = this.allocRawBytes(data.length);
        this.storeRawBytes(dataOffset, data);
        this.storeUSize(headerOffset + this.memory.sizeofPtr(), data.length);
        this.addressToSetTargetValue.push([offset, headerOffset]);
        this.addressToSetTargetValue.push([headerOffset, dataOffset]);
    }
    /**
     * Update internal cache views.
     */
    updateViews() {
        this.viewU8 = new Uint8Array(this.buffer);
        this.viewI32 = new Int32Array(this.buffer);
        this.viewU32 = new Uint32Array(this.buffer);
        this.viewF64 = new Float64Array(this.buffer);
    }
}
exports.CachedCallStack = CachedCallStack;
//# sourceMappingURL=memory.js.map