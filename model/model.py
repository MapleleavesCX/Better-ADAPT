from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import MobileNetV2, NASNetMobile

from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Input
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import os


def SVGG():
    '''简化版VGG: 只包含少量卷积层和池化层的网络'''
    # 输入层.
    input_tensor = Input(shape=(32, 32, 3))  # 调整为CIFAR-10的输入尺寸
    
    # 第一个卷积块
    x = Convolution2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)
    
    # 第二个卷积块
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)
    
    # 全连接层
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    output_tensor = Dense(10, activation='softmax', name='predictions')(x)  # CIFAR-10有10个类别
    
    # 返回模型
    return Model(inputs=input_tensor, outputs=output_tensor)


def SAlexNet():
    '''小型AlexNet: 基于经典的AlexNet,但规模大幅缩小'''
    # 输入层.
    input_tensor = Input(shape=(32, 32, 3))  # 调整为CIFAR-10的输入尺寸
    
    # 第一个卷积块
    x = Convolution2D(96, (5, 5), strides=(1, 1), activation='relu', padding='same', name='block1_conv1')(input_tensor)  # 减小卷积核尺寸并调整步长
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # 第二个卷积块
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)  # 减小卷积核尺寸
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    # 全连接层
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    output_tensor = Dense(10, activation='softmax', name='predictions')(x)  # CIFAR-10有10个类别
    
    # 返回模型
    return Model(inputs=input_tensor, outputs=output_tensor)


def LeNet5():
    # 输入层.
    input_tensor = Input(shape=(28, 28, 1))
    # 块 1
    x = Convolution2D(6, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    # 块 2
    x = Convolution2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    # 全连接层
    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='redictions')(x)
    # 返回tensorflow定义模型
    return Model(input_tensor, x)


def LeNet4():
    # 输入层.
    input_tensor = Input(shape=(28, 28, 1))
    # 块 1
    x = Convolution2D(6, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    # 块 2
    x = Convolution2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    # 全连接层
    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    # 相比于LeNet-5，这里直接从120维到输出类别数（假设为10类），省略了一个中间的全连接层。
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    # 返回tensorflow定义模型
    return Model(input_tensor, x)


if __name__ == "__main__":

    # 创建并打印模型结构
    '''第一类:采用数据集:MNIST    28*28*1  '''

    model1 = LeNet4()
    model1.summary()

    model2 = LeNet5()
    model2.summary()

    '''第二类:采用数据集:CIFAR-10   32*32*3  '''

    model3 = SVGG()
    model3.summary()

    model4 = SAlexNet()
    model4.summary()

    '''第三类:采用数据集:ImageNet   '''

    model5 = VGG19(weights='imagenet') # 548MB，层数最少（19层）
    model5.summary()

    model6 = ResNet50(weights='imagenet') # 97MB，层数第三多
    model6.summary()

    model7 = MobileNetV2(weights='imagenet') # 13MB，层数次多
    model7.summary()

    model8 = NASNetMobile(weights='imagenet')  # 20MB，层数非常多
    model8.summary()