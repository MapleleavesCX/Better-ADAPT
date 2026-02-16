import numpy as np
from adapt.metric.metric import Metric

class StrongNeuronActivationCoverage(Metric):
    def __init__(self, history=20):
        self.history = history
        self._internals = []
        super(StrongNeuronActivationCoverage, self).__init__()

    def covered(self, internals, **kwargs):
        covered = []

        if self._internals == []:
            for layer_output in internals:
                self._internals.append(np.array(layer_output))
                maxs = np.max(layer_output)
                vec = [True if n >= maxs else False for n in layer_output]
                covered.append(np.array(vec))
            return np.array(covered, dtype=object)
        else:
            for index, layer_output in enumerate(internals):
                self._internals[index] = np.vstack((self._internals[index], layer_output))
                self._internals[index][-self.history:]
                maxs = np.max(self._internals[index], axis=0)
                vec = layer_output >= maxs
                covered.append(np.array(vec))
            return np.array(covered, dtype=object)

    def __repr__(self):
        return 'StrongNeuronActivationCoverage()'  



# 示例使用
if __name__ == "__main__":
    # 创建度量对象
    metric = StrongNeuronActivationCoverage()
    old_covered = [False]*11
    old_covered = np.array(old_covered)
    for i in range(5):
        layout1 = np.random.rand(3)
        layout2 = np.random.rand(6)
        layout4 = np.random.rand(2)
        internals_example = [layout1,layout2,layout4]
        print(f"internals_example:{internals_example}")
        # 计算覆盖布尔数组
        new_covered = np.concatenate(metric.covered(internals_example))
        old_covered = np.bitwise_or(old_covered, new_covered)
        print(f"Coverages:{np.mean(old_covered)}\n\n")