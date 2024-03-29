
import numpy as np
from ..measure import MeasureInput, create_measure_batch


class Runner(object):
    def __init__(self, task, config, option):
        self.task = task
        self.config = config
        self.option = option

    def run(self):
        print("runner.py: config %s" % str(self.config))
        measure_batch = create_measure_batch(self.task, self.option)
        inputs = [MeasureInput(self.task.target, self.task, self.config)]
        results = measure_batch(inputs)
        cost_len = len(results[0].costs)
        mean_cost = np.mean(results[0].costs)
        print("runner.py: results %s" % str(results))
        print("runner.py: cost_len %s" % str(cost_len))
        print("runner.py: mean_cost %s" % str(mean_cost))
        print("runner.py: GFLOPS %s" % str(
                self.task.flop * 1.0e-9 / mean_cost))
        print("runner.py: Peak GFLOPS %s" % str(
                self.task.flop * 1.0e-9 / np.min(results[0].costs)))
        return results[0].costs
