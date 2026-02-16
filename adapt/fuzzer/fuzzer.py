import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from adapt.network import Network
from adapt.fuzzer.archive import Archive
from adapt.metric import NeuronCoverage
from adapt.strategy import RandomStrategy
from adapt.utils.functional import coverage
from adapt.utils.timer import Timeout
from adapt.utils.timer import Timer

class WhiteBoxFuzzer:
  '''A white-box fuzzer for deep neural network.
  
  White-box testing is a technique that utilizes internal values to generate
  inputs. This class will uses the gradients to generate next input for testing.
  This fuzzer will test one input.
  '''

  def __init__(self, network, input, metric=None, strategy=None, k=10, delta=0.5, class_weight=0.5, neuron_weight=0.5, lr=0.1, trail=3, decode=None):
    '''Create a fuzzer.
    
    Create a white-box fuzzer. All parameters except for the time budget, should
    be set.

    Args:
      network: A wrapped Keras model with `adapt.Network`. Wrap if not wrapped.
      input: An input to test.
      metric: A coverage metric for testing. By default, the fuzzer will use
        a neuron coverage with threshold 0.5.
      strategy: A neuron selection strategy. By default, the fuzzer will use
        the `adapt.strategy.RandomStrategy`.
      k: A positive integer. The number of the neurons to select.
      delta: A positive floating point number. Limits of distance of created
        inputs. By default, use 0.5.
      class_weight: A floating point number. A weight for the class term in
        optimization equation. By default, use 0.5.
      neuron_weight: A floating point number. A weight for the neuron term in
        optimization equation. By default, use 0.5.
      lr: A floating point number. A learning rate to apply when generating
        the next input using gradients. By default, use 0.1.
      trail: A positive integer. Trails to apply one set of selected neurons.
      decode: A function that gets logits and return the label. By default,
        uses `np.argmax`.

    Raises:
      ValueError: When arguments are not in their proper range.
    '''

    # Store variables.
    if not isinstance(network, Network):
      network = Network(network)
    self.network = network

    self.input = np.array(input)

    if not metric:
      metric = NeuronCoverage(0.5)
    self.metric = metric

    if not strategy:
      strategy = RandomStrategy(self.network)
    self.strategy = strategy

    if k < 1:
      raise ValueError('The argument k is not positive.')
    self.k = int(k)

    if delta <= 0:
      raise ValueError('The argument delta is not positive.')
    self.delta = float(delta)

    self.class_weight = float(class_weight)
    self.neuron_weight = float(neuron_weight)
    self.lr = float(lr)

    if trail < 1:
      raise ValueError('The argument trails is not positive.')
    self.trail = trail

    if not decode:
      decode = np.argmax
    self.decode = decode

    # Variables that are set during (or after) testing.
    self.archive = None

    self.start_time = None
    self.time_consumed = None

    self.label = None
    self.orig_coverage = None
    self.covered = None
    self.coverage = None

  def start(self, hours=0, minutes=0, seconds=0, append='meta', verbose=0):
    '''Start fuzzing for the given time budget.

    Start fuzzing for a time budget.

    Args:
      hours: A non-negative integer which indicates the time budget in hours.
        0 for the default value.
      minutes: A non-negative integer which indicates the time budget in minutes.
        0 for the defalut value.
      seconds: A non-negative integer which indicates the time budget in seconds.
        0 for the defalut value. If hours, minutes, and seconds are set to be 0,
        the time budget will automatically set to be 10 seconds.
      append: An option that specifies the data that archive stores. Should be one
        of "meta", "min_dist", or "all". By default, "meta" will be used.
      verbose: An option that print out logs or not. Pass 1 for printing and 0 for
        not printing. Be default, set to be 0.
    '''

    # Get the original properties.
    internals, logits, gradients = self.network.predict(np.array([self.input]), get_grad=True)

    orig_index = np.argmax(logits)
    orig_norm = np.linalg.norm(self.input)
    self.label = self.decode(np.array([logits]))
    self.covered = self.metric(internals=internals, logits=logits, gradients=gradients)
    self.orig_coverage = coverage(self.covered)

    # Initialize variables.
    self.archive = Archive(self.input, self.label, append=append)

    # Initialize the strategy.
    self.strategy = self.strategy.init(
      covered=self.covered, 
      label=self.label, 
      delta=self.delta
      )

    # Set timer.
    timer = Timer(hours, minutes, seconds)
    if verbose > 0:
      print('Fuzzing started. Press ctrl+c to quit.')

    '''==========================从这里开始，进入三重循环，耗时重灾区，或许可以优化？====================='''

    # Loop until timeout, or interrupted by user.
    try:
      while True:

        # Create worklist.
        worklist = [tf.identity(np.array([self.input]))]

        # While worklist is not empty:
        while len(worklist) > 0:

          # Get input
          input = worklist.pop(0)

          # Select neurons.
          neurons = self.strategy(k=self.k)
          
          # Try trail times.
          for _ in range(self.trail):

            # Get original coverage
            orig_cov = coverage(self.covered)


            # Calculate gradients.        
            with tf.GradientTape() as t:
              t.watch(input)
              internals, logits, _ = self.network.predict(input)
              loss = self.neuron_weight * K.sum([internals[li][ni] for li, ni in neurons]) - self.class_weight * logits[orig_index]
            dl_di = t.gradient(loss, input)

            # Generate the next input using gradients.
            input += self.lr * dl_di

            # Get the properties of the generated input.
            '''这里的internals是具体神经元的激活值, logits是最终输出, gradients是梯度'''
            internals, logits, gradients = self.network.predict(input, get_grad=True)

            '''这里传入当前神经网络模型的输出进入指标，返回一个覆盖向量'''
            covered = self.metric(internals=internals, logits=logits, gradients=gradients)
            
            label = self.decode(np.array([logits]))

            distance = np.linalg.norm(input - self.input) / orig_norm

            '''将获得的覆盖向量与旧覆盖向量或运算'''
            # Update varaibles in fuzzer
            self.covered = np.bitwise_or(self.covered, covered)
            '''然后把更新后的覆盖向量算一个均值 *覆盖向量为布尔数组，所以最后算出来的一定在0~1之间'''
            new_cov = coverage(self.covered)
            '''当新的覆盖向量均值大于旧的且生成样本距离小于设定值，把该输入加入工作区'''
            # If coverage increased.
            if new_cov > orig_cov and distance < self.delta:
              worklist.append(tf.identity(input))

            '''当前覆盖向量用于学习更新策略'''
            # Feedback to strategy.
            self.strategy.update(covered=covered, 
                                 label=label, 
                                 internals=internals, 
                                 distance=distance,
                                 gradients=gradients
                                 )

            # Add created input.
            self.archive.add(input, label, distance, timer.elapsed.total_seconds(), new_cov)

        # Check timeout.
        timer.check_timeout()

        # Update strategy.
        self.strategy.next()

    except Timeout:
      pass
    except KeyboardInterrupt:
      if verbose > 0:
        print('Stopped by the user.')

    # Update meta variables.
    self.coverage = coverage(self.covered)
    self.start_time = timer.start_time
    self.time_consumed = timer.elapsed.total_seconds()
    self.archive.timestamp.append((self.time_consumed, self.coverage))

    if verbose > 0:
      print('Done!')

    return self.archive
