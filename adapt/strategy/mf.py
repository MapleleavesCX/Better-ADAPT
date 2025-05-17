from math import ceil
import tensorflow as tf
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

class EnhancedFeatureMatrix:
  '''增强版特征矩阵，新增神经元动态行为特征'''
    
  # 原特征维度
  CONST_FEATURES = 17
  VARIABLE_FEATURES = 12
  EXTRA_FEATURES = 13
  TOTAL_FEATURES = CONST_FEATURES + VARIABLE_FEATURES + EXTRA_FEATURES


  ''' =========================== __init__ 方法 ==========================='''

  def __init__(self, network, history=50):
      '''初始化时增加激活历史记录窗口'''
      self.network = network
      self.history = history # 历史记录最大存储数目
      self._internals = [] # 存储历史记录
      
      # 初始化原特征矩阵
      self._init_constant_features()
      self.variable_vectors = np.zeros(
         (len(self.const_vectors), self.VARIABLE_FEATURES), dtype=int)
      self.extra_vectors = np.zeros(
         (len(self.const_vectors), self.EXTRA_FEATURES), dtype=int)


  ''' =========================== update 方法 ==========================='''


  def update(self, covered_count, objective_covered, internals, gradients):
      '''更新特征矩阵
      covered_count: 这是一行整数数组，统计每个神经元激活次数
      objective_covered: 目标覆盖，指示对抗样本的覆盖
      internals: 这是具有层间关系的激活值向量
      gradients: 这是具有层间关系的每层详细梯度
      '''

      self.update_variable_features(
         covered_count=covered_count, objective_covered=objective_covered)   
      
      self.update_extra_features(internals=internals, gradients=gradients)

      return self

  @property
  def matrix(self):
      '''合并所有特征'''
      return np.concatenate([
          self.const_vectors, 
          self.variable_vectors,
          self.extra_vectors  
      ], axis=1)  # 已经完成初始化了
  
  def dot(self, vector):
      '''点积运算需适应新维度'''
      return np.dot(self.matrix, vector[:self.TOTAL_FEATURES])


# **************************************************************************************************
# **************************************************************************************************
  def update_extra_features(self, internals, gradients):
    '''=========== 额外特征更新 =========='''
    
    ''' ==================== 梯度类特征 ==================== '''
    # 将当前梯度展平
    gradients = np.concatenate(gradients)
    
    # 清零额外特征
    self.extra_vectors = np.zeros(
    (len(self.const_vectors), self.EXTRA_FEATURES), dtype=int)

    # 计算梯度相关的变量特征
    # f29 梯度非零
    idx = np.squeeze(np.argwhere(gradients == 0))
    self.extra_vectors[idx, 0] = 1
    # f30 梯度为正
    idx = np.squeeze(np.argwhere(gradients > 0))
    self.extra_vectors[idx, 1] = 1
    # f31 梯度为负
    idx = np.squeeze(np.argwhere(gradients < 0))
    self.extra_vectors[idx, 2] = 1
    # 设置梯度排序百分比特征 (f32-f41)
    self._set_gradient_features(gradients)


    ''' ==================== 激活值类特征 ==================== '''
    # # 将当前激活值展平
    # internals = np.concatenate(internals)
    # # 记录历史激活值
    # self._internals.append(internals)
    # if len(self._internals) < 2:  
    #   # 因为只有一行记录，无法统计，特征不更新
    #   return 
    # # 存储记录只保留 history 条
    # self._internals = self._internals[-self.history:]


# **************************************************************************************************
# **************************************************************************************************
  def update_variable_features(self, covered_count, objective_covered):
    '''=========== 变量特征更新 ==========='''
    
    # 清零原变量特征
    self.variable_vectors = np.zeros(
        (len(self.const_vectors), self.VARIABLE_FEATURES), dtype=int)
    
    # f17: 目标触发时激活
    idx = np.squeeze(np.argwhere(objective_covered > 0))
    self.variable_vectors[idx, 0] = 1
    
    # f18: 从未激活
    idx = np.squeeze(np.argwhere(covered_count < 1))
    self.variable_vectors[idx, 1] = 1
    
    # f19-28: 激活次数百分位排序
    self._set_covered_features(covered_count, idx)


# **************************************************************************************************
# **************************************************************************************************
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

  def _set_gradient_features(self, gradients):
    '''设置梯度相关特征 (f32-f41)'''
    argsort = np.argsort(np.abs(gradients))  # 根据梯度绝对值排序
    n = len(argsort)
    
    # 梯度绝对值的分位数特征
    self.extra_vectors[argsort[int(n*0.9):], 3] = 1            # top 10%
    self.extra_vectors[argsort[int(n*0.8):int(n*0.9)], 4] = 1  # 10%-20%
    self.extra_vectors[argsort[int(n*0.7):int(n*0.8)], 5] = 1  # 20%-30%
    self.extra_vectors[argsort[int(n*0.6):int(n*0.7)], 6] = 1  # 30%-40%
    self.extra_vectors[argsort[int(n*0.5):int(n*0.6)], 7] = 1  # 40%-50%
    self.extra_vectors[argsort[int(n*0.4):int(n*0.5)], 8] = 1  # 50%-60%
    self.extra_vectors[argsort[int(n*0.3):int(n*0.4)], 9] = 1  # 60%-70%
    self.extra_vectors[argsort[int(n*0.2):int(n*0.3)], 10] = 1 # 70%-80%
    self.extra_vectors[argsort[int(n*0.1):int(n*0.2)], 11] = 1 # 80%-90%
    self.extra_vectors[argsort[:int(n*0.1)], 12] = 1           # 90%-100%

  def _set_covered_features(self, covered_count, idx_never_covered):
    '''f19-28: 激活次数百分位排序'''
    
    argsort = np.setdiff1d(
       np.argsort(covered_count), idx_never_covered, assume_unique=True
       )# 这里要去掉从未激活的神经元
    n = len(argsort)

    self.variable_vectors[argsort[int(n*0.9):], 2] = 1            # top 10%
    self.variable_vectors[argsort[int(n*0.8):int(n*0.9)], 3] = 1  # 10%-20%
    self.variable_vectors[argsort[int(n*0.7):int(n*0.8)], 4] = 1  # 20%-30%
    self.variable_vectors[argsort[int(n*0.6):int(n*0.7)], 5] = 1  # 30%-40%
    self.variable_vectors[argsort[int(n*0.5):int(n*0.6)], 6] = 1  # 40%-50%
    self.variable_vectors[argsort[int(n*0.4):int(n*0.5)], 7] = 1  # 50%-60%
    self.variable_vectors[argsort[int(n*0.3):int(n*0.4)], 8] = 1  # 60%-70%
    self.variable_vectors[argsort[int(n*0.2):int(n*0.3)], 9] = 1  # 70%-80%
    self.variable_vectors[argsort[int(n*0.1):int(n*0.2)], 10] = 1 # 80%-90%
    self.variable_vectors[argsort[:int(n*0.1)], 11] = 1           # 90%-100%

# **************************************************************************************************
# **************************************************************************************************
# **************************************************************************************************



# class EnhancedFeatureMatrix:
#   '''增强版特征矩阵，新增其他变量特征'''
    
#   # 原特征维度
#   CONST_FEATURES = 17
#   VARIABLE_FEATURES = 12
#   EXTRA_FEATURES = 13
#   TOTAL_FEATURES = CONST_FEATURES + VARIABLE_FEATURES + EXTRA_FEATURES


#   ''' =========================== __init__ 方法 ==========================='''

#   def __init__(self, network, history=50):
#       '''初始化时增加激活历史记录窗口'''
#       self.network = network
#       self.history = history # 历史记录最大存储数目
#       self._internals = [] # 存储历史激活记录
#       self.last_gradients = [] # 存储上一条梯度记录
#       self.gradients_count = None # 存储梯度变化计数
      
#       # 初始化原特征矩阵
#       self._init_constant_features()
#       self.variable_vectors = np.zeros(
#          (len(self.const_vectors), self.VARIABLE_FEATURES), dtype=int)
#       self.extra_vectors = np.zeros(
#          (len(self.const_vectors), self.EXTRA_FEATURES), dtype=int)


#   ''' =========================== update 方法 ==========================='''


#   def update(self, covered_count, objective_covered, internals, gradients):
#       '''更新特征矩阵
#       covered_count: 这是一行整数数组，统计每个神经元激活次数
#       objective_covered: 目标覆盖，指示对抗样本的覆盖
#       internals: 这是具有层间关系的激活值向量
#       gradients: 这是具有层间关系的每层详细梯度
#       '''

#       self.update_variable_features(
#          covered_count=covered_count, objective_covered=objective_covered)   
      
#       self.update_extra_features(internals=internals, gradients=gradients)

#       return self

#   @property
#   def matrix(self):
#       '''合并所有特征'''
#       return np.concatenate([
#           self.const_vectors, 
#           self.variable_vectors,
#           self.extra_vectors  
#       ], axis=1)  # 已经完成初始化了
  
#   def dot(self, vector):
#       '''点积运算需适应新维度'''
#       return np.dot(self.matrix, vector[:self.TOTAL_FEATURES])


# # **************************************************************************************************
# # **************************************************************************************************
#   def update_extra_features(self, internals, gradients):
#     '''=========== 额外特征更新 =========='''
    
#     ''' ==================== 梯度类特征 ==================== '''
    
       
#     # 将当前梯度展平
#     gradients = np.concatenate(gradients)

#     # 梯度变号次数记录
#     if self.last_gradients == []:
#       self.last_gradients = gradients
#       self.gradients_count = np.zeros_like(gradients, dtype=int)
#     else:
#       self.gradients_count += (np.sign(self.last_gradients) != np.sign(gradients)).astype(int)
#       self.last_gradients = gradients

    
#     # 清零额外特征
#     self.extra_vectors = np.zeros(
#     (len(self.const_vectors), self.EXTRA_FEATURES), dtype=int)

#     # 计算梯度相关的变量特征

#     # f29 梯度为正
#     idx = np.squeeze(np.argwhere(gradients > 0))
#     self.extra_vectors[idx, 1] = 1
#     # f30 梯度为负
#     idx = np.squeeze(np.argwhere(gradients < 0))
#     self.extra_vectors[idx, 2] = 1
#     # f31: 从未变号
#     idx = np.squeeze(np.argwhere(self.gradients_count < 1))
#     self.variable_vectors[idx, 1] = 1
#     # 设置梯度变号排序百分比特征 (f32-f41)
#     self._set_gradient_features(self.gradients_count, idx)


#     ''' ==================== 激活值类特征 ==================== '''
#     # # 将当前激活值展平
#     # internals = np.concatenate(internals)
#     # # 记录历史激活值
#     # self._internals.append(internals)
#     # if len(self._internals) < 2:  
#     #   # 因为只有一行记录，无法统计，特征不更新
#     #   return 
#     # # 存储记录只保留 history 条
#     # self._internals = self._internals[-self.history:]


# # **************************************************************************************************
# # **************************************************************************************************
#   def update_variable_features(self, covered_count, objective_covered):
#     '''=========== 变量特征更新 ==========='''
    
#     # 清零原变量特征
#     self.variable_vectors = np.zeros(
#         (len(self.const_vectors), self.VARIABLE_FEATURES), dtype=int)
    
#     # f17: 目标触发时激活
#     idx = np.squeeze(np.argwhere(objective_covered > 0))
#     self.variable_vectors[idx, 0] = 1
    
#     # f18: 从未激活
#     idx = np.squeeze(np.argwhere(covered_count < 1))
#     self.variable_vectors[idx, 1] = 1
    
#     # f19-28: 激活次数百分位排序
#     self._set_covered_features(covered_count, idx)


# # **************************************************************************************************
# # **************************************************************************************************
#   def _init_constant_features(self):
#       '''原常量特征初始化逻辑保持不变'''
#       self.const_vectors = []
#       weights = []
      
#       for li, l in enumerate(self.network.layers[:-1]):
#           # 原f0-f3特征
#           layer_location = int((li / (len(self.network.layers) - 1)) * 4)
          
#           # 原f4-f9特征
#           w = l.get_weights()[0] if len(l.get_weights()) > 0 else np.zeros(l.output.shape[1:])
          
#           # 原f10-f16特征
#           layer_type = self._get_layer_type(l)
          
#           for ni in range(l.output.shape[-1]):
#               weights.append(np.mean(w[..., ni]))
#               vec_c = np.zeros(self.CONST_FEATURES, dtype=int)
#               vec_c[layer_location] = 1
#               vec_c[layer_type] = 1
#               self.const_vectors.append(vec_c)
              
#       self.const_vectors = np.array(self.const_vectors)
#       self._set_weight_features(weights)
  
#   def _get_layer_type(self, layer):
#       '''辅助方法：获取层类型编码'''
#       if isinstance(layer, BatchNormalization):
#           return 10
#       elif isinstance(layer, (MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D)):
#           return 11
#       elif isinstance(layer, (Conv2D, ZeroPadding2D)):
#           return 12
#       elif isinstance(layer, Dense):
#           return 13
#       elif isinstance(layer, Activation):
#           return 14
#       elif isinstance(layer, (Add, Concatenate, Lambda)):
#           return 15
#       else:
#           return 16
  
#   def _set_weight_features(self, weights):
#       '''设置权重相关特征(f4-f9)'''
#       argsort = np.argsort(weights)
#       n = len(argsort)
      
#       self.const_vectors[argsort[int(n*0.9):], 4] = 1  # top 10%
#       self.const_vectors[argsort[int(n*0.8):int(n*0.9)], 5] = 1  # 10-20%
#       self.const_vectors[argsort[int(n*0.7):int(n*0.8)], 6] = 1  # 20-30%
#       self.const_vectors[argsort[int(n*0.6):int(n*0.7)], 7] = 1  # 30-40%
#       self.const_vectors[argsort[int(n*0.5):int(n*0.6)], 8] = 1  # 40-50%
#       self.const_vectors[argsort[:int(n*0.5)], 9] = 1  # bottom 50%

#   def _set_gradient_features(self, gradients_count, idx_never_change_sign):
#     '''设置梯度变化相关特征 (f32-f41)'''
#     argsort = np.setdiff1d(
#        np.argsort(gradients_count), idx_never_change_sign, assume_unique=True
#        )# 这里要去掉从未变号的梯度对应的神经元id
#     n = len(argsort)
    
#     # 梯度变化次数的分位数特征
#     self.extra_vectors[argsort[int(n*0.9):], 3] = 1            # top 10%
#     self.extra_vectors[argsort[int(n*0.8):int(n*0.9)], 4] = 1  # 10%-20%
#     self.extra_vectors[argsort[int(n*0.7):int(n*0.8)], 5] = 1  # 20%-30%
#     self.extra_vectors[argsort[int(n*0.6):int(n*0.7)], 6] = 1  # 30%-40%
#     self.extra_vectors[argsort[int(n*0.5):int(n*0.6)], 7] = 1  # 40%-50%
#     self.extra_vectors[argsort[int(n*0.4):int(n*0.5)], 8] = 1  # 50%-60%
#     self.extra_vectors[argsort[int(n*0.3):int(n*0.4)], 9] = 1  # 60%-70%
#     self.extra_vectors[argsort[int(n*0.2):int(n*0.3)], 10] = 1 # 70%-80%
#     self.extra_vectors[argsort[int(n*0.1):int(n*0.2)], 11] = 1 # 80%-90%
#     self.extra_vectors[argsort[:int(n*0.1)], 12] = 1           # 90%-100%

#   def _set_covered_features(self, covered_count, idx_never_covered):
#     '''f19-28: 激活次数百分位排序'''
    
#     argsort = np.setdiff1d(
#        np.argsort(covered_count), idx_never_covered, assume_unique=True
#        )# 这里要去掉从未激活的神经元
#     n = len(argsort)

#     self.variable_vectors[argsort[int(n*0.9):], 2] = 1            # top 10%
#     self.variable_vectors[argsort[int(n*0.8):int(n*0.9)], 3] = 1  # 10%-20%
#     self.variable_vectors[argsort[int(n*0.7):int(n*0.8)], 4] = 1  # 20%-30%
#     self.variable_vectors[argsort[int(n*0.6):int(n*0.7)], 5] = 1  # 30%-40%
#     self.variable_vectors[argsort[int(n*0.5):int(n*0.6)], 6] = 1  # 40%-50%
#     self.variable_vectors[argsort[int(n*0.4):int(n*0.5)], 7] = 1  # 50%-60%
#     self.variable_vectors[argsort[int(n*0.3):int(n*0.4)], 8] = 1  # 60%-70%
#     self.variable_vectors[argsort[int(n*0.2):int(n*0.3)], 9] = 1  # 70%-80%
#     self.variable_vectors[argsort[int(n*0.1):int(n*0.2)], 10] = 1 # 80%-90%
#     self.variable_vectors[argsort[:int(n*0.1)], 11] = 1           # 90%-100%














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
    self.matrix = EnhancedFeatureMatrix(network)

    # Create variables.
    self.bound = bound
    self.label = None
    self.covered_count = None
    self.objective_covered = None

    # Create a random strategy.
    self.strategy = np.random.uniform(-self.bound, self.bound, size=EnhancedFeatureMatrix.TOTAL_FEATURES)

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

  def update(self, covered, label, internals, gradients, **kwargs):
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
    
    # 更新特征矩阵
    self.matrix.update(
        covered_count=self.covered_count,
        objective_covered=self.objective_covered,
        internals=internals,
        gradients=gradients
    )

class MoreFeaturesStrategy(ParameterizedStrategy):


  def __init__(self, network, bound=5, size=100, history=300, remainder=0.5, sigma=1):


    super(MoreFeaturesStrategy, self).__init__(network, bound)

    # Initialize variables.
    self.size = size
    self.history = history
    self.remainder = remainder
    self.sigma = sigma

    # Create initial stratagies randomly.
    self.strategies = [np.random.uniform(-self.bound, self.bound, size=EnhancedFeatureMatrix.TOTAL_FEATURES) for _ in range(self.size)]
    self.strategy = self.strategies.pop(0)

    # Create a coverage vector for a strategy.
    self.strategy_covered = None

    # Storage for used strategies and their result.
    self.records = []

  def init(self, covered, label, **kwargs):

    super(MoreFeaturesStrategy, self).init(
       covered=covered, label=label, **kwargs)

    # Initialize coverage vector for one strategy.
    self.strategy_covered = np.zeros(len(self.neurons), dtype=bool)

    return self

  def update(self, covered, label, internals, gradients, **kwargs):

    super(MoreFeaturesStrategy, self).update(
       covered=covered, label=label, internals=internals, gradients=gradients, **kwargs)

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

    # Get the next strategy.
    if len(self.strategies) > 0:
      self.strategy = self.strategies.pop(0)
      self.strategy_covered = np.zeros_like(self.strategy_covered, dtype=bool)
      return self

    # Generate next strategies from the past records.
    records = self.records[-self.history:]

    # Find a set of strategies that maximizes the coverage.
    n = int(self.size * self.remainder)
    strategies, covereds = tuple(zip(*records))
    _, indices = greedy_max_set(covereds, n=n)

    # Find the maximum coverages for remaining part.
    n = n - len(indices)
    if n > 0:
      coverages = list(map(np.mean, covereds))
      indices = indices + list(np.argpartition(coverages, -n)[-n:])
    
    # Get strategies.
    selected = np.array(strategies)[indices]

    # Mix strategies randomly.
    n = len(selected)
    generation = ceil(1 / self.remainder)
    left = selected[np.random.permutation(n)]
    right = selected[np.random.permutation(n)]

    for l, r in zip(left, right):
      for _ in range(generation):

        # Generate new strategy.
        s = np.array([l[i] if np.random.choice([True, False]) else r[i] for i in range(EnhancedFeatureMatrix.TOTAL_FEATURES)])

        # Add little distortion.
        s = s + np.random.normal(0, self.sigma, size=EnhancedFeatureMatrix.TOTAL_FEATURES)

        # Clip the ranges.
        s = np.clip(s, -self.bound, self.bound)

        self.strategies.append(s)

    self.strategies = self.strategies[:self.size]

    # Get the next strategy.
    self.strategy = self.strategies.pop(0)

    return self
