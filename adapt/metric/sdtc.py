import numpy as np
from adapt.metric.metric import Metric

class StandardDeviationThresholdCoverage(Metric):
    '''Standard Deviation Threshold Coverage (SDTC).
    
    利用均值(Mean)和标准差(Standard Deviation)设定阈值(Threshold)来进行覆盖(Coverage),
    关注的是神经元激活值是否超出了一定的范围(Range).
    '''

    def __init__(self):
        '''Create a Neuron Layer Coverage metric.
        
        Args:
            None
        '''
        super(StandardDeviationThresholdCoverage, self).__init__()


    def covered(self, internals, **kwargs):
        '''Returns a list of neuron layer coverage vectors.
        
        Args:
            internals: A list of the values of internal neurons in each layer.
            kwargs: Not used. Present for the compatibility with the super class.
        
        Returns:
            A list of coverage vectors that identifies which neurons are activated.
        '''
        
        covered = []
        
        for layer_output in internals:
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

    def __repr__(self):
        '''Returns a string representation of the object.
        
        Example:
        
        >>> from adapt.metric import StandardDeviationThresholdCoverage
        >>> metric = SDTC()
        >>> metric
        StandardDeviationThresholdCoverage()
        '''
        return 'StandardDeviationThresholdCoverage()'