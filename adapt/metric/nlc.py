import numpy as np
from scipy.stats import skew, kurtosis
# from adapt.metric.metric import Metric

# class NeuraLCoverage(Metric):
class NeuraLCoverage:
    '''Neural Layer Coverage(NLC).
    动态累积批次数据，并根据批次数量选择使用 NC或 NLC方法。
    '''
    def __init__(self):
        '''
        初始化 NeuralLayerCoverage对象。
        不再需要人为设定初始阈值。
        '''
        super(NeuraLCoverage, self).__init__()
        self.layer_batches = {}  # 用于存储每层的批次数据
        
    def calculate_statistics(self, layer_output):
        '''计算多种统计量'''
        mean = np.mean(layer_output)
        std_dev = np.std(layer_output)
        skewness = skew(layer_output)
        kurt = kurtosis(layer_output)
        return mean, std_dev, skewness, kurt
    
    def calculate_adaptive_threshold(self, stats, factor=1.0):
        '''基于统计量动态调整阈值'''
        _, std_dev, skewness, kurt = stats
        adaptive_threshold = std_dev + abs(skewness) + abs(kurt)
        return adaptive_threshold * factor
    
    def covered(self, internals, **kwargs):
        '''
        返回一个表示哪些神经元被覆盖的布尔数组列表。
        Args:
            internals:当前批次的每一层神经元激活值的列表。
            例如[array([...]), array([...]),...]。
        Returns:
            每一层的覆盖布尔数组。
        '''
        coverages = []
        layer_index = 0
        for layer_output in internals:
            if layer_index not in self.layer_batches:
                self.layer_batches[layer_index] = []
            self.layer_batches[layer_index].append(layer_output)  # 累积当前批次数据

            # 使用累积数据计算统计量
            cumulative_data = np.vstack(self.layer_batches[layer_index])
            stats = self.calculate_statistics(cumulative_data.flatten())
            adaptive_threshold = self.calculate_adaptive_threshold(stats)

            up = stats[0] + stats[1] + adaptive_threshold
            down = stats[0] - stats[1] - adaptive_threshold

            # 打印调试信息
            print(f"Layer {layer_index} - Stats: {stats}, Adaptive Threshold: {adaptive_threshold}, Up: {up}, Down: {down}")

            # 判断每个神经元是否被覆盖
            vec = [True if n < down or n > up else False for n in layer_output]
            coverages.append(np.array(vec))
            layer_index += 1
        return np.array(coverages, dtype=object)
    
    def __repr__(self):
        return 'NeuraLCoverage()'

# 示例使用
if __name__ == "__main__":
    metric = NeuraLCoverage()
    for i in range(5):
        layout1 = np.random.uniform(-1,1,3)
        layout2 = np.random.uniform(-1, 1, 12)
        layout4 = np.random.uniform(-1, 1, 4)
        internals_example = [layout1, layout2, layout4]
        print(f"internals_example:{internals_example}")
        coverages = metric.covered(internals_example)
        print(f"Coverages:{coverages}\n\n")



# import numpy as np
# from adapt.metric.metric import Metric

# class NeuraLCoverage(Metric):
#     '''Neural Layer Coverage (NLC).
    
#     动态累积批次数据，并根据批次数量选择使用NC或NLC方法。
#     '''

#     def __init__(self, threshold=0.1):
#         '''
#         初始化NeuralLayerCoverage对象。
        
#         Args:
#             threshold: 覆盖阈值，用于判断神经元是否被覆盖。
#         '''
#         super(NeuraLCoverage, self).__init__()

#         self.threshold = threshold
#         self.layer_batches = {}  # 用于存储每层的批次数据

#     def covered(self, internals, **kwargs):
#         '''
#         返回一个表示哪些神经元被覆盖的布尔数组列表。
        
#         Args:
#             internals: 当前批次的每一层神经元激活值的列表。
#                        例如 [array([...]), array([...]), ...]。
            
#         Returns:
#             每一层的覆盖布尔数组。
#         '''
#         coverages = []
#         layer_index = 0
        
#         for layer_output in internals:
#             # 将当前批次数据添加到累积数据中
#             if layer_index not in self.layer_batches:
#                 self.layer_batches[layer_index] = []
#             self.layer_batches[layer_index].append(layer_output)
            
#             # 根据累积的批次数量决定使用哪种方法
#             if len(self.layer_batches[layer_index]) < 2:
#                 # 均值
#                 average = np.mean(layer_output)
#                 # 标准差
#                 variance = np.std(layer_output)
#                 # 上下界
#                 up = average + variance
#                 down = average - variance
#                 # 根据是否超出范围确定覆盖
#                 vec = [True if n<down or n>up else False for n in layer_output]
#             else:
#                 # 使用协方差矩阵方法（NLC）
#                 '''这里采用人为规定超参数值，不合理，应该设置为自动拟合一个适合的超参数'''
#                 covariance_matrix = covariance_matrix = np.cov(self.layer_batches[layer_index], rowvar=False)
#                 vec = [np.any(np.abs(row) > self.threshold) for row in covariance_matrix]
            
#             coverages.append(np.array(vec))
#             layer_index += 1
        
#         return np.array(coverages, dtype=object)

#     def __repr__(self):
#         return f'NeuraLCoverage(threshold={self.threshold})'

# # 示例使用
# if __name__ == "__main__":
#     # 创建度量对象
#     metric = NeuraLCoverage(threshold=0.1)

#     for i in range(5):
#         layout1 = np.random.rand(3)
#         layout2 = np.random.rand(12)
#         layout4 = np.random.rand(4)
#         internals_example = [layout1,layout2,layout4]
#         print(f"internals_example:{internals_example}")
#         # 计算覆盖布尔数组
#         coverages = metric.covered(internals_example)
        
#         print(f"Coverages:{coverages}\n\n")