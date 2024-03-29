
import sys
import json
import argparse
import numpy as np
from enum import Enum


invalid_cost = 1000000000.0


class JsonDataFormat(Enum):
  UNKNOWN = 0
  AUTOTVM = 1
  AUTOSCHEDULER = 2


class FLOPsHelper:
  @staticmethod
  def batch_matmul(batch, M, K, N):
    return batch * 2 * M * K * N
  
  @staticmethod
  def conv2d_gemm(IH, IW, IC, OC, PH, PW, SH, SW):
    return 0


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--filepath", type=str, required=True, help="File path")
  parser.add_argument("--start-line", type=int, default=0, help="Start line")
  parser.add_argument("--line-count", type=int, default=-1, help="Line count")
  parser.add_argument("--use-default-schedule", action="store_true")
  return parser.parse_args()


def load_config_entity(entity):
  out_text = ""
  for e in entity:
    ename = e[0]
    etype = e[1]
    evalues = e[2]
    out_text = out_text + "%s,%s," % (ename, etype)
    if etype == "sp":
      for ev in evalues:
        out_text = out_text + "%d," % ev
    elif etype == "ot":
      out_text = out_text + "%s," % str(evalues)
  return out_text[:-1]


def get_json_data_format(data):
  if "input" in data:
    return JsonDataFormat.AUTOTVM
  elif "i" in data:
    return JsonDataFormat.AUTOSCHEDULER
  else:
    return JsonDataFormat.UNKNOWN


def unify_kernel_type(kernel_type):
  if kernel_type.startswith("dense"):
    return "dense." + kernel_type.split(".")[1]
  else:
    return kernel_type


def parse_json_data(data):
  kernel_identify = ""
  config_entity_text = ""
  mean_cost = 0
  err_code = 0
  total_cost = 0

  json_data_format = get_json_data_format(data)
  if json_data_format == JsonDataFormat.AUTOTVM:
    kernel_type = unify_kernel_type(data["input"][1])
    kernel_size = data["input"][2]
    #print("kernel_type:", kernel_type)
    #print("kernel_size:", kernel_size)
    kernel_identify = "%s,%s" % (kernel_type, kernel_size)

    #print(config)
    config = data["config"]
    #print(result)
    result = data["result"]
    
    config_entity_text = load_config_entity(config["entity"])
    cost_list = result[0]
    mean_cost = np.mean(np.array(cost_list))
    cost_count = len(cost_list)
    err_code = result[1]
    total_cost = result[2]
    timestamp = result[3]
  elif json_data_format == JsonDataFormat.AUTOSCHEDULER:
    kernel_hash_and_size = data["i"][0][0]
    #print("kernel_hash_and_size:", kernel_hash_and_size)
    kernel_identify = kernel_hash_and_size

    config_entity_text = "unknown"

    result = data["r"]
    cost_list = result[0]
    mean_cost = np.mean(np.array(cost_list))
    cost_count = len(cost_list)
    err_code = result[1]
    total_cost = result[2]
    timestamp = result[3]
  
  return kernel_identify, config_entity_text, mean_cost, cost_count, \
      err_code, total_cost, timestamp, json_data_format


class Record(object):
  def __init__(self):
    self.reset()
    self.kernel_identifies = []
    self.min_mean_cost_list = []
    self.min_mean_cost_config_text_list = []
    self.time_and_mean_cost_list_list = []
    self.count_and_mean_cost_list_list = []
  
  def reset(self, kernel_identify=""):
    self.min_mean_cost = sys.float_info.max
    self.min_mean_cost_idx = -1
    self.min_mean_cost_config_text = ""
    self.record_count = 0
    self.timestamp = 0
    self.total_timestamp_list = []
    self.count_and_min_mean_cost_list = [[], []]
    self.time_and_min_mean_cost_list = [[], []]
    self.out_text_list = []
    self.kidx = 0
    self.last_kernel_identify = kernel_identify
    if kernel_identify != "":
      self.kernel_identifies.append(kernel_identify)

  def cur_kernel_idx(self):
    return len(self.min_mean_cost_list)

  def record_cur_min_mean_cost(self):
    if self.min_mean_cost < invalid_cost:
      self.min_mean_cost_list.append(self.min_mean_cost)
    else:
      self.min_mean_cost_list.append(0)
    self.min_mean_cost_config_text_list.append(self.min_mean_cost_config_text)
    self.count_and_min_mean_cost_list[0].append(self.record_count)
    self.count_and_min_mean_cost_list[1].append(self.min_mean_cost)
    self.count_and_mean_cost_list_list.append(self.count_and_min_mean_cost_list)
    self.time_and_min_mean_cost_list[0].append(self.timestamp)
    self.time_and_min_mean_cost_list[1].append(self.min_mean_cost)
    self.time_and_mean_cost_list_list.append(self.time_and_min_mean_cost_list)

  def calc_cost_sum(self, remove_same_kernel_ident=False):
    if not remove_same_kernel_ident:
      return np.sum(np.array(self.min_mean_cost_list))
    else:
      min_mean_cost_dict = {}
      for i in range(len(self.kernel_identifies)):
        kernel_ident = self.kernel_identifies[i]
        min_mean_cost = self.min_mean_cost_list[i]
        if kernel_ident not in min_mean_cost_dict:
          min_mean_cost_dict[kernel_ident] = min_mean_cost
        else:
          if min_mean_cost < min_mean_cost_dict[kernel_ident]:
            min_mean_cost_dict[kernel_ident] = min_mean_cost
      cost_sum = 0
      for kernel_ident in min_mean_cost_dict:
        cost_sum = cost_sum + min_mean_cost_dict[kernel_ident]
      return cost_sum
    
  def print_best_config(self):
    print("\nBest configures:")
    num_kernels = len(self.min_mean_cost_list)
    print("KernelIdx,BestConfig,MinCost")
    for i in range(0, num_kernels):
      min_cost = self.min_mean_cost_list[i]
      min_cost_config = self.min_mean_cost_config_text_list[i]
      print("%d,[%s],%.6f" % (i, min_cost_config, min_cost))

  def print_time_and_min_cost(self):
    print("\nTime and min cost:")
    num_kernels = len(self.min_mean_cost_list)
    for i in range(0, num_kernels):
      count_and_mean_cost_list = self.count_and_mean_cost_list_list[i]
      count_list = count_and_mean_cost_list[0]
      time_and_mean_cost_list = self.time_and_mean_cost_list_list[i]
      time_list = time_and_mean_cost_list[0]
      min_cost_list = time_and_mean_cost_list[1]
      print("Kernel %d" % i)
      print("Count:")
      print(count_list)
      print("Time (ms):")
      print(list(np.array(time_list) * 1000.0))
      #print("Time (sec):")
      #print(list(np.array(time_list)))
      #print("Time (min):")
      #print(list(np.array(time_list) / 60.0))
      #print("Cost (sec):")
      #print(min_cost_list)
      print("Cost (ms):")
      # NOTE(fucheng): For Conv2D (GEMM), there are 4 kernels per op.
      kernel_count_per_op = 1
      print(list(np.array(min_cost_list) * 1000.0 * kernel_count_per_op))
      #print("Performance, GFLOPs:", gflops)
      #flops = FLOPsHelper.batch_matmul(1, 384, 768, 768)
      #gflops = 1.0 * flops / 1e9
      #print(list(gflops / np.array(min_cost_list)))

  def print_total_time_cost(self):
    print("\nTotal time cost:")
    for i in range(1, len(self.total_timestamp_list)):
      total_time_cost = self.total_timestamp_list[i] - self.total_timestamp_list[i - 1]
      print("kernel %d, total_time_cost %.2f sec" % (i, total_time_cost))


def load(json_filepath, start_line=0, line_count=-1, use_default_schedule=False):
  print("filename:", json_filepath)
  print("use_default_schedule:", use_default_schedule)

  record = Record()

  line_idx = 0
  should_update_cost = True
  record.reset()
  for line in open(json_filepath, "r"):
    if line_idx < start_line:
      line_idx = line_idx + 1
      continue
    data = json.loads(line)
    
    kernel_identify, config_entity_text, \
        mean_cost, cost_count, err_code, total_cost, timestamp, \
        json_data_format = parse_json_data(data)
    
    record.total_timestamp_list.append(timestamp)

    #print("last_kernel_type:", record.last_kernel_type)
    #print("last_kernel_size:", record.last_kernel_size)
    if kernel_identify != record.last_kernel_identify:
      if len(record.out_text_list) > 0:
        print("%d,%s" % (record.cur_kernel_idx(), record.out_text_list[record.min_mean_cost_idx]))
        record.record_cur_min_mean_cost()
      record.reset(kernel_identify)
      should_update_cost = True

    out_text = "%s,%s,%f,%f" % (kernel_identify, config_entity_text, mean_cost, total_cost)
    #print(out_text)
    record.out_text_list.append(out_text)

    record.record_count += 1
    #record.timestamp += total_cost
    if mean_cost < invalid_cost:
      #record.timestamp += mean_cost
      record.timestamp += (total_cost - mean_cost * cost_count)

    if should_update_cost and \
        mean_cost < record.min_mean_cost and \
        mean_cost < invalid_cost:
      record.min_mean_cost = mean_cost
      record.min_mean_cost_config_text = config_entity_text
      record.count_and_min_mean_cost_list[0].append(record.record_count)
      record.count_and_min_mean_cost_list[1].append(mean_cost)
      record.time_and_min_mean_cost_list[0].append(record.timestamp)
      record.time_and_min_mean_cost_list[1].append(mean_cost)
      record.min_mean_cost_idx = record.kidx
      if use_default_schedule:
        should_update_cost = False

    #print("kidx:", record.kidx)
    record.kidx = record.kidx + 1

    #break

    line_idx = line_idx + 1
    if line_count > 0 and line_idx - start_line >= line_count:
      break

  if len(record.out_text_list) > 0:
    print("%d,%s" % (record.cur_kernel_idx(), record.out_text_list[record.min_mean_cost_idx]))
    record.record_cur_min_mean_cost()

  record.print_best_config()

  record.print_time_and_min_cost()

  #record.print_total_time_cost()

  if False:
    remove_same_kernel_ident = (json_data_format == JsonDataFormat.AUTOSCHEDULER)
    cost_sum = record.calc_cost_sum(remove_same_kernel_ident)
    print("cost_sum %.6f s = %.3f ms" % (cost_sum, cost_sum * 1000.0))
    print("")


if __name__ == "__main__":
  args = parse_args()
  load(args.filepath, args.start_line, args.line_count, args.use_default_schedule)
