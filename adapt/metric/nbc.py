import numpy as np
from adapt.metric.metric import Metric

class NeuronBoundaryCoverage(Metric):
    '''
    Neuron Boundary Coverage (NBC).
    '''

    def __init__(self, history=20):
        '''Create a Neuron Layer Coverage metric.
        
        Args:
            None
        '''
        self.history = history
        self._internals = []
        super(NeuronBoundaryCoverage, self).__init__()


    def covered(self, internals, **kwargs):
        '''Returns a list of neuron layer coverage vectors.
        
        Args:
            internals: A list of the values of internal neurons in each layer.
            kwargs: Not used. Present for the compatibility with the super class.
        
        Returns:
            A list of coverage vectors that identifies which neurons are activated.
        '''
        covered = []

        if self._internals == []:  # 初始化
            for layer_output in internals:
                # 添加行，其中layer_output应为一个一维数组
                self._internals.append(np.array(layer_output))
                # 均值
                average = np.mean(layer_output)
                # 标准差
                variance = np.std(layer_output)
                # 上下界
                up = average + variance
                down = average - variance
                # 根据是否超出范围确定覆盖
                vec = [True if n<down or n>up else False for n in layer_output]
                covered.append(np.array(vec))
            # 返回覆盖率向量
            return np.array(covered, dtype=object)
        else:   
            for index, layer_output in enumerate(internals):
                # 添加新行
                self._internals[index] = np.vstack((self._internals[index], layer_output))
                # 保持记录数目
                self._internals[index][-self.history:]
                # 按列计算每个神经元历史激活值的均值、标准差
                averages = np.mean(self._internals[index], axis=0) # 按列计算
                variances = np.std(self._internals[index], axis=0) # 按列计算
                # 上下界
                ups = averages + variances
                downs = averages - variances
                # 根据是否超出范围确定覆盖
                vec = (layer_output > ups) | (layer_output < downs)
                covered.append(np.array(vec))
            # 返回覆盖率向量
            return np.array(covered, dtype=object)

    def __repr__(self):
        '''Returns a string representation of the object.
        
        Example:
        
        >>> from adapt.metric import NeuronBoundaryCoverage
        >>> metric = NBC()
        >>> metric
        NeuronBoundaryCoverage()
        '''
        return 'NeuronBoundaryCoverage()'
    



# 示例使用
if __name__ == "__main__":
    # 创建度量对象
    metric = NeuronBoundaryCoverage()
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