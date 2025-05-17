from math import ceil
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
import numpy as np

from adapt.strategy.strategy import Strategy
from adapt.utils.functional import greedy_max_set


from scipy import stats
from collections import deque

class FeatureMatrix:
  
  # 原特征维度
  CONST_FEATURES = 17
  VARIABLE_FEATURES = 12
  TOTAL_FEATURES = CONST_FEATURES + VARIABLE_FEATURES


  ''' =========================== __init__ 方法 ==========================='''

  def __init__(self, network, window_size=50):
      '''初始化时增加激活历史记录窗口'''
      self.network = network
      self.window_size = window_size
      
      # 初始化原特征矩阵
      self._init_constant_features()
      # 初始化变量特征为0向量
      self.variable_vectors = np.zeros((len(self.const_vectors), self.VARIABLE_FEATURES), dtype=int)


  def _init_constant_features(self):
      '''原常量特征初始化逻辑保持不变'''
      self.const_vectors = []
      weights = []
      
      for li, l in enumerate(self.network.layers[:-1]):
          # 原f0-f3特征
          layer_location = int((li / (len(self.network.layers) - 1)) * 4)
          
          # 原f4-f9特征
          w = l.get_weights()[0] if len(l.get_weights()) > 0 else np.zeros(l.output.shape[1:])
          
          # 原f10-f16特征
          layer_type = self._get_layer_type(l)
          
          for ni in range(l.output.shape[-1]):
              weights.append(np.mean(w[..., ni]))
              vec_c = np.zeros(self.CONST_FEATURES, dtype=int)
              vec_c[layer_location] = 1
              vec_c[layer_type] = 1
              self.const_vectors.append(vec_c)
              
      self.const_vectors = np.array(self.const_vectors)
      self._set_weight_features(weights)
  
  def _get_layer_type(self, layer):
      '''辅助方法：获取层类型编码'''
      if isinstance(layer, BatchNormalization):
          return 10
      elif isinstance(layer, (MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D)):
          return 11
      elif isinstance(layer, (Conv2D, ZeroPadding2D)):
          return 12
      elif isinstance(layer, Dense):
          return 13
      elif isinstance(layer, Activation):
          return 14
      elif isinstance(layer, (Add, Concatenate, Lambda)):
          return 15
      else:
          return 16
  
  def _set_weight_features(self, weights):
      '''设置权重相关特征(f4-f9)'''
      argsort = np.argsort(weights)
      n = len(argsort)
      
      self.const_vectors[argsort[int(n*0.9):], 4] = 1  # top 10%
      self.const_vectors[argsort[int(n*0.8):int(n*0.9)], 5] = 1  # 10-20%
      self.const_vectors[argsort[int(n*0.7):int(n*0.8)], 6] = 1  # 20-30%
      self.const_vectors[argsort[int(n*0.6):int(n*0.7)], 7] = 1  # 30-40%
      self.const_vectors[argsort[int(n*0.5):int(n*0.6)], 8] = 1  # 40-50%
      self.const_vectors[argsort[:int(n*0.5)], 9] = 1  # bottom 50%





  ''' =========================== update 方法 ==========================='''


  def update(self, covered_count, objective_covered):
      '''更新特征矩阵
      covered_count: 这是一行numpy的int数组，记录了每个神经元的激活次数
      objective_covered: 覆盖布尔向量，指示找到对抗样本后的覆盖向量，已经展平为一维的向量
      '''
      
      # 更新原变量特征
      self.variable_vectors = np.zeros(
         (len(self.const_vectors), self.VARIABLE_FEATURES), dtype=int)
      
      # f17: 目标触发时激活
      idx = np.squeeze(np.argwhere(objective_covered > 0))
      self.variable_vectors[idx, 0] = 1
      
      # f18: 从未激活
      idx = np.squeeze(np.argwhere(covered_count < 1))
      self.variable_vectors[idx, 1] = 1
      
      # f19-28: 激活百分位
      sorted_idx = np.setdiff1d(
         np.argsort(covered_count), 
         idx, 
         assume_unique=True
         ) # 激活值排序，返回索引

      n = len(sorted_idx)
      ranges = [
          (0.9, 2), (0.8, 3), (0.7, 4), 
          (0.6, 5), (0.5, 6), (0.4, 7), 
          (0.3, 8), (0.2, 9), (0.1, 10)
      ]
      
      for threshold, col in ranges:
          start = int(n * threshold)
          if threshold == 0.9:
              self.variable_vectors[sorted_idx[start:], col] = 1
          else:
              end = int(n * (threshold + 0.1))
              self.variable_vectors[sorted_idx[start:end], col] = 1
      self.variable_vectors[sorted_idx[:int(n*0.1)], 11] = 1
      
      return self




  @property
  def matrix(self):
      '''合并所有特征'''
      return np.concatenate([
          self.const_vectors, 
          self.variable_vectors  
      ], axis=1)  # 已经完成初始化了
  
  def dot(self, vector):
      '''点积运算需适应新维度'''
      return np.dot(self.matrix, vector[:self.TOTAL_FEATURES])







class ParameterizedStrategy(Strategy):
  '''一种使用参数化选择策略的策略.
  
  参数化神经元选择策略是一种用选择向量对神经元进行参数化和评分的策略。请参阅以下论文了解更多详细信息：

  Effective White-Box Testing for Deep Neural Networks with Adaptive Neuron-Selection Strategy
  http://prl.korea.ac.kr/~pronto/home/papers/issta20.pdf
  '''

  def __init__(self, network, bound=5):
    '''Create a parameterized strategy, and initialize its variables.
    
    Args:
      network: A wrapped Keras model with `adapt.Network`.
      bound: A floating point number indicates the absolute value of minimum
        and maximum bounds.

    Example:

    >>> from adapt import Network
    >>> from adapt.strategy import ParameterizedStrategy
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> model = VGG19()
    >>> network = Network(model)
    >>> strategy = ParameterizedStrategy(network)
    '''

    super(ParameterizedStrategy, self).__init__(network)

    # Initialize feature vectors for each neuron.
    self.matrix = FeatureMatrix(network)

    # Create variables.
    self.bound = bound
    self.label = None
    self.covered_count = None
    self.objective_covered = None

    # Create a random strategy.
    self.strategy = np.random.uniform(-self.bound, self.bound, size=FeatureMatrix.TOTAL_FEATURES)

  def select(self, k):
    '''Select k neurons with highest scores.
    
    Args:
      k: A positive integer. The number of neurons to select.

    Returns:
      A list of locations of selected neurons.
    '''

    # Calculate scores.
    scores = self.matrix.dot(self.strategy)

    # Get k highest neurons and return their location.
    indices = np.argpartition(scores, -k)[-k:]
    return [self.neurons[i] for i in indices]

  def init(self, covered, label, **kwargs):
    '''Initialize the variables of the strategy.

    This method should be called before all other methods in the class.

    Args:
      covered: A list of coverage vectors that the initial input covers.
      label: A label of the initial input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.

    Raises:
      ValueError: When the size of the passed coverage vectors are not matches
        to the network setting.
    '''
    
    # Flatten coverage vectors.
    covered = np.concatenate(covered)
    if len(covered) != len(self.neurons):
      raise ValueError('The number of neurons in network does not matches to the setting.')

    # 初始化每个神经元的覆盖数量.
    self.covered_count = np.zeros_like(covered, dtype=int)
    self.covered_count += covered

    # 设置初始标签.
    self.label = label

    # 当目标满足时初始化覆盖向量.
    self.objective_covered = np.zeros_like(self.covered_count, dtype=bool)

    return self

  def update(self, covered, label,**kwargs):
    '''Update the variable of the strategy.

    Args:
      covered: A list of coverage vectors that a current input covers.
      label: A label of a current input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''
    
    # Flatten coverage vectors.
    covered = np.concatenate(covered)

    # Update the number of covering for each neuron.
    self.covered_count += covered

    # If adversarial input found, update the coverage vector for objective satifaction.
    if self.label != label:
      self.objective_covered = np.bitwise_or(self.objective_covered, covered)

    # Update variable vectors

    # 传入：覆盖计数、目标覆盖
    self.matrix = self.matrix.update(self.covered_count, self.objective_covered)





class BetterStrategy(ParameterizedStrategy):

  def __init__(self, network, bound=5, size=100, history=1500, sigma=1):

    super(BetterStrategy, self).__init__(network, bound)

    # 初始化各种超参数变量.
    self.size = size
    self.history = history
    self.sigma = sigma

    # 随机创建初始策略, strategies为策略池
    self.strategies = [np.random.uniform(
       -self.bound, self.bound, size=FeatureMatrix.TOTAL_FEATURES
       ) for _ in range(self.size)]
    # 从策略池获取队首策略
    self.strategy = self.strategies.pop(0)
    # 为策略创建一个覆盖向量.
    self.strategy_covered = None
    # 储存使用过的策略与其对应的结果.
    self.records = []

  def init(self, covered, label, delta, **kwargs):

    super(BetterStrategy, self).init(covered=covered, label=label, **kwargs)

    # 初始化有关策略向量的变量.
    self.org_label = label
    self.strategy_distance = delta # 初始距离，与外界白盒测试主程序一致
    self.strategy_covered = np.zeros(len(self.neurons), dtype=bool)

    return self

  def update(self, covered, label, distance, **kwargs):

    super(BetterStrategy, self).update(covered=covered, label=label, **kwargs)

    # 仅当为对抗样本时更新平均距离.
    if self.org_label != label:
      self.strategy_distance = min(self.strategy_distance, distance)
    # 更新覆盖.
    covered = np.concatenate(covered) # 展平为一维数组
    self.strategy_covered = np.bitwise_or(self.strategy_covered, covered)

    return self

  '''=================================================================================='''

  def next(self):

    # 将当前完成测试的策略与对应结果添加记录
    # strategy_covered 是一个布尔向量，在update中已经展平为一维数组
    # strategy_distance 是一个整数，表示当前策略在测试过程中生成的对抗样本与初始样本的最小距离
    self.records.append((self.strategy, self.strategy_covered, self.strategy_distance))
    
    # 若策略池里的策略还没测完，则获取下一个策略继续测试
    if len(self.strategies) > 0:
      return self._get_next_strategy()

    # 从过去记录中获取超参数 self.history 个最近记录
    self.records = self.records[-self.history:]
    # 这里把records解压后，得到一组策略向量和策略向量对应的覆盖向量、对抗样本的最小平均距离
    strategies, covereds, distances = zip(*self.records)

    ''' =========================== 改进的核心部分 =========================== '''
    ''' =========================== 改进的核心部分 =========================== '''
    ''' =========================== 改进的核心部分 =========================== '''
    ''' =========================== 改进的核心部分 =========================== '''
    
    '''根据覆盖率、平均距离计算得分，归一化为概率分布，用于指示策略向量挑选'''
    # 计算每个策略对应覆盖向量的覆盖期望
    coverages = np.mean(covereds, axis=1)
    # 设计相关公式: scores = coverages - distances, 得分越高则说明该策略越优秀
    scores = coverages - np.array(distances)
    # 计算特征重要性
    feat_importance = _calculate_feature_importance(X=strategies, y=scores)


    '''归一化 特征重要性 到[0, 1]范围，作为指示交叉的特征重要性概率分布'''
    # 计算特征重要性的最小值和最大值
    feat_min = np.min(feat_importance)
    feat_max = np.max(feat_importance)

    # 如果最大值等于最小值，则直接设置为均匀分布
    if feat_max == feat_min:
      probabilities = np.ones_like(feat_importance) / len(feat_importance)  # 均匀分布
    else:
      probabilities = (feat_importance - feat_min) / (feat_max - feat_min)

    # 确保 probabilities 的值在 [0, 1] 范围内，并避免浮点数误差
    probabilities = np.clip(probabilities, 0, 1)
    
    # 基于重要性生成新策略
    new_strategies = []
    for _ in range(self.size):
      
      """选择特征分布互补的策略"""
      # 按照策略的综合得分计算选择概率
      scores_distribution = scores - np.min(scores)  # 将最小值平移到 0
      if np.sum(scores_distribution) == 0:
          # 如果所有得分都相同，则使用均匀分布
          scores_distribution = np.ones_like(scores) / len(scores)
      else:
          # 否则进行归一化
          scores_distribution /= np.sum(scores_distribution)
      l_idx = np.random.choice(len(strategies), p=scores_distribution)
      left = strategies[l_idx]
      r_idx = np.random.choice(len(strategies), p=scores_distribution)
      right = strategies[r_idx]
      
      """基于特征重要性的智能交叉"""
      # 生成 [0, 1] 范围内的随机数
      random_values = np.random.rand(len(feat_importance))  
      # 比较随机数与重要性概率
      mask = random_values < probabilities
      # 根据 mask 选择策略: 重要的特征有更高概率保留优势策略的值
      new_strategy = np.where(mask, left, right)

      """基于特征重要性概率地控制变异幅度"""
      # 定向变异（重要性越高变异幅度越小）
      mutation = np.random.normal(
          scale=np.exp(-feat_importance) * self.sigma)
      # 应用变异
      new_strategy += mutation
      new_strategy = np.clip(new_strategy, -self.bound, self.bound)
      new_strategies.append(new_strategy)
    
    self.strategies = new_strategies
    return self._get_next_strategy()

  def _get_next_strategy(self):
    """获取下一个策略并初始化其覆盖记录"""
    self.strategy = self.strategies.pop(0)
    self.strategy_covered = np.zeros_like(self.strategy_covered, dtype=bool)
    return self



def _calculate_feature_importance(X, y):
    """辅助函数: 计算各特征维度的重要性"""

    # 使用 回归方法 获取稀疏重要性，自动选择alpha值
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    model = LassoCV(cv=5).fit(X, y)

    return np.abs(model.coef_) + 1e-6  # 防止除零