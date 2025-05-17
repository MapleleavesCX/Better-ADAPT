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
      covered_count: 这是一行numpy的布尔数组，数组大小表示神经元总个数，值为True表示该神经元被覆盖
      objective_covered: 目标覆盖，指示对抗样本的覆盖
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
  '''A strategy that uses a parameterized selection strategy.
  
  Parameterized neuron selection strategy is a strategy that parameterized
  neurons and scores with a selection vector. Please see the following paper
  for more details:

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

    # Initialize the number of covering for each neuron.
    self.covered_count = np.zeros_like(covered, dtype=int)
    self.covered_count += covered

    # Set the initial label.
    self.label = label

    # Initialize the coverage vector when objective satisfies.
    self.objective_covered = np.zeros_like(self.covered_count, dtype=bool)

    return self

  def update(self, covered, label, **kwargs):
    '''Update the variable of the strategy.

    Args:
      covered: A list of coverage vectors that a current input covers.
      label: A label of a current input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''
    
    # 这里把具有层结构的covered转为一行
    '''例如：
    >>> covered = [
      np.array([False, True, True]),
      np.array([True, False, True, True]),
      np.array([False, True])
    ]
    >>> covered = np.concatenate(covered)
    >>> covered
    [ False  True  True  True  False  True  True  False  True]
    '''
    covered = np.concatenate(covered)

    # 对每个神经元激活次数进行计数. 这里self.covered_count为int类型
    self.covered_count += covered

    # 如果发现对抗性输入，更新目标满意度的覆盖向量.
    if self.label != label:
      self.objective_covered = np.bitwise_or(self.objective_covered, covered)

    # 这里更新参数特征
    self.matrix = self.matrix.update(self.covered_count, self.objective_covered)













class AdaptiveParameterizedStrategy(ParameterizedStrategy):
  '''A adaptive and parameterized neuron selection strategy.
  
  Adaptive and parameterized neuron selection strategy is a strategy that changes
  the parameterized neuron selection strategy adaptively with respect to the model,
  data, or even time. These updates are done in online; in other words, the strategies
  are updated while testing. Please see the following paper for detail:

  Effective White-Box Testing for Deep Neural Networks with Adaptive Neuron-Selection Strategy
  http://prl.korea.ac.kr/~pronto/home/papers/issta20.pdf
  '''

  def __init__(self, network, bound=5, size=100, history=300, remainder=0.5, sigma=1):
    '''Create a adaptive parameterized strategy, and initialize its variables.
    
    Args:
      network: A wrapped Keras model with `adapt.Network`.
      bound: 一个浮点数，策略向量数值边界，表示最小和最大边界的绝对值.
      size: 一个正整数. 一次创建的策略数量.
      history: 一个正整数. 策略学习和生成下一个策略时要创建的策略数量.
      remainder: [0,1]中的浮点数. 生成下一个策略的策略部分.
      sigma: 一个非负浮点数。正态分布的标准差，增加了策略的多样性.

    Raises:
      ValueError: When arguments are not in proper range.

    Example:

    >>> from adapt import Network
    >>> from adapt.strategy import AdaptiveParameterizedStrategy
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> model = VGG19()
    >>> network = Network(model)
    >>> strategy = AdaptiveParameterizedStrategy(network)
    '''

    super(AdaptiveParameterizedStrategy, self).__init__(network, bound)

    # Initialize variables.
    self.size = size
    self.history = history
    self.remainder = remainder
    self.sigma = sigma

    # Create initial stratagies randomly.
    self.strategies = [np.random.uniform(-self.bound, self.bound, size=FeatureMatrix.TOTAL_FEATURES) for _ in range(self.size)]
    self.strategy = self.strategies.pop(0)

    # Create a coverage vector for a strategy.
    self.strategy_covered = None

    # Storage for used strategies and their result.
    self.records = []

  def init(self, covered, label, **kwargs):
    '''Initialize the variables of the strategy.

    This method should be called before all other methods in the class.

    Args:
      covered: A list of coverage vectors that the initial input covers.
      label: A label of the initial input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''

    super(AdaptiveParameterizedStrategy, self).init(covered=covered, label=label, **kwargs)

    # Initialize coverage vector for one strategy.
    self.strategy_covered = np.zeros(len(self.neurons), dtype=bool)

    return self

  def update(self, covered, label, **kwargs):
    '''Update the variable of the strategy.

    Args:
      covered: A list of coverage vectors that a current input covers.
      label: A label of a current input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''

    super(AdaptiveParameterizedStrategy, self).update(covered=covered, label=label, **kwargs)

    # Flatten coverage vectors.
    covered = np.concatenate(covered)
    self.strategy_covered = np.bitwise_or(self.strategy_covered, covered)

    return self

  def next(self):
    '''Get the next parameterized strategy.
    
    Get the next parameterized strategy. If the generated parameterized strategies
    are all used, generate new parameterized strategies.

    Returns:
      Self for possible call chains.
    '''

    # Finish current strategy.
    self.records.append((self.strategy, self.strategy_covered))

    # 当策略数大于0，弹出已经测试过的队首策略，清空该策略的覆盖，然后返回继续测下一个.
    if len(self.strategies) > 0:
      self.strategy = self.strategies.pop(0)
      self.strategy_covered = np.zeros_like(self.strategy_covered, dtype=bool)
      return self

    
    # records = self.records[-self.history:]
    # strategies, covereds = tuple(zip(*records))
    '''
    这里有个不是很好的地方: 获取了最近的h个记录,
    但是却不抛弃其他记录, 然而其他记录在整个代码没
    有任何地方用到, 这会无端占用大量内存, 应改为:
    '''
    # 当 self.strategy 里的策略测完了，开始生成新策略.
    # 从过去记录中获取超参数 self.history 个最近记录
    self.records = self.records[-self.history:]
    # 这里把records解压后，得到一组策略向量和策略向量对应的覆盖
    strategies, covereds = tuple(zip(*self.records))

    # 这里 n 是如何规定的？为什么是 self.size * self.remainder ？
    n = int(self.size * self.remainder)
    
    # 使用贪心算法找到最大覆盖的最小集S1中元素索引 indices.
    # greedy_max_set 返回最大集 _ 和该集合元素在 covereds 中的索引 indices
    _, indices = greedy_max_set(covereds, n=n)

    # 从剩下部分找最大覆盖的记录集合S2的元素索引，|S2|=n-|S1|，然后合并到 indices.
    n = n - len(indices)
    if n > 0:
      coverages = list(map(np.mean, covereds))
      indices = indices + list(np.argpartition(coverages, -n)[-n:])
    
    # 按照索引从strategies中挑选重新合成一个新的数组selected.
    selected = np.array(strategies)[indices]

    # 采用随机交叉.
    n = len(selected)
    generation = ceil(1 / self.remainder) # generation 指示每一对的交叉次数
    left = selected[np.random.permutation(n)] # 随机选了一组，共n个待交叉的策略向量，为左组
    right = selected[np.random.permutation(n)] # 随机选了一组，共n个待交叉的策略向量，为右组
    # PS：以上选组是可能会出现左右选到同一个策略向量的

    # 开始交叉，对向量的每一个分量，随机选择左组或右组
    for l, r in zip(left, right):
      for _ in range(generation):# 对每一对随机交叉 generation 次

        # 生成新策略向量.
        s = np.array([l[i] if np.random.choice([True, False]) else r[i] for i in range(FeatureMatrix.TOTAL_FEATURES)])

        # 添加一个随机噪声.
        s = s + np.random.normal(0, self.sigma, size=FeatureMatrix.TOTAL_FEATURES)

        # 修剪该向量的每个分量，值大小不能超出正负 self.bound 范围.
        s = np.clip(s, -self.bound, self.bound)

        self.strategies.append(s)

    # 从以上已经生成的系列策略向量中，仅取最新的 self.size 个，其余抛弃
    self.strategies = self.strategies[:self.size]

    # 获取下一个策略.
    self.strategy = self.strategies.pop(0)

    return self
