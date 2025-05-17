import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class Network:
  '''用于封装 Keras 模型的类。
  
  该类可以帮助获取内部神经元的值和梯度。所有在 ADAPT 中使用的模型都应使用此类进行封装。
  '''

  def __init__(self, model, skippable=None):
    '''
    通过 Keras 模型创建一个封装类。
      参数:
        model: Keras 模型，此参数是必需的。
        skippable: 可跳过的 Keras 层类列表。默认情况下，所有由 `tensorflow.keras.layers.Flatten`
            和 `tensorflow.keras.layers.InputLayer` 创建的层都会被跳过。    
    Example:

    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> from adapt import Network
    >>> model = VGG19()
    >>> network = Network(model)
    '''

    self.model = model

    # 如果未指定 skippable，则使用默认的可跳过层。
    if not skippable:
      skippable = [InputLayer, Flatten]
    self.skippable = skippable

    # 定义一个返回非跳过层输出的函数。
    self.functors = Model(
      inputs = self.model.input, 
      outputs = [l.output for l in self.model.layers if type(l) not in self.skippable])
    
  def predict(self, x, get_grad=False):
    '''
    计算输入的内部值、logits 和梯度。
      参数:
        x: 要处理的输入。目前 Network 类不支持批量处理，因此输入的第一个维度必须为 1。
      返回:
        一个三元组，包含：
          - 每一层内部神经元的值列表（internals）。
          - logits。
          - 每一层相对于输入的梯度列表（gradients），保留层次关系。
    '''
    if get_grad == True:
      # 确保输入是 TensorFlow 张量
      if not isinstance(x, tf.Tensor):
          x = tf.convert_to_tensor(x)

      # 使用 GradientTape 记录前向传播过程，并设置为持久模式
      with tf.GradientTape(persistent=True) as tape:
          tape.watch(x)  # 监控输入以便计算梯度
          outs = self.functors(x)  # 获取每一层的输出
          logits = outs[-1]  # 最后一层为 logits

      # 计算每一层输出相对于输入的梯度
      gradients = [tape.gradient(out, x) for out in outs]
      # 显式删除 tape 以释放资源
      del tape

      # 对内部输出进行归一化处理
      internals = [K.mean(K.reshape(l, (-1, l.shape[-1])), axis=0) for l in outs[:-1]]

      return internals, logits, gradients
    
    else:
      # Get output and normalize to get the output of the neurons.
      outs = [K.mean(K.reshape(l, (-1, l.shape[-1])), axis = 0) for l in self.functors(x)]
      # Return internal outputs and logits.
      internals = outs[:-1]
      logits = outs[-1]
      return internals, logits, None

  @property
  def layers(self):
    '''A list of layers that is not skippable.
    
    Example:
    
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> from adapt import Network
    >>> model = VGG19()
    >>> network = Network(model)
    >>> len(network.layers)
    24
    '''

    # Return a list of layers which are not skippable.
    return [l for l in self.model.layers if type(l) not in self.skippable]

