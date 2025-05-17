


def find_inhomogeneous_parts(covered):
    shapes = [arr.shape for arr in covered]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print("Inhomogeneous parts detected:")
        for i, shape in enumerate(shapes):
            if shape != shapes[0]:  # 假设第一个子数组的形状是标准形状
                print(f"Element {i}: shape={shape}, content={covered[i]}")
    else:
        print("All elements have the same shape.")



def print_inhomogeneous_details(covered):
    shapes = [arr.shape for arr in covered]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print("Inhomogeneous parts detected. Details:")
        for i, arr in enumerate(covered):
            print(f"Element {i}: shape={arr.shape}, dtype={arr.dtype}, content={arr}")
    else:
        print("All elements have the same shape.")




import numpy as np

def check_and_fix_inhomogeneous(covered):
    # 检测不规则的部分
    shapes = [arr.shape for arr in covered]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print("Inhomogeneous parts detected. Fixing...")
        
        # 填充数组
        max_length = max(len(arr) for arr in covered)
        padded_covered = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=False) for arr in covered]
        
        return padded_covered
    else:
        print("All elements have the same shape.")
        return covered
    





# 以下来自 NLC ,原本采用为import torch,改写为使用tensorflow
'''PS:未证明改写后功能相同'''

import tensorflow as tf

class Estimator:
    def __init__(self, feature_num, num_class=1):
        '''Initialize the Estimator.
        
        Args:
            feature_num: Number of features (neurons) in each layer.
            num_class: Number of classes (default is 1).
        '''
        self.num_class = num_class
        self.feature_num = feature_num

        # Initialize variables for mean, covariance, and amount
        self.CoVariance = tf.Variable(
            tf.zeros((num_class, feature_num, feature_num)), dtype=tf.float32
        )
        self.Ave = tf.Variable(
            tf.zeros((num_class, feature_num)), dtype=tf.float32
        )
        self.Amount = tf.Variable(
            tf.zeros(num_class), dtype=tf.float32
        )
        self.CoVarianceInv = tf.Variable(
            tf.zeros((num_class, feature_num, feature_num)), dtype=tf.float32
        )

    def calculate(self, features, labels=None):
        '''Calculate the mean, covariance, and amount for the given features.
        
        Args:
            features: A tensor of shape (N, A), where N is the number of samples
                    and A is the number of features.
            labels: A tensor of shape (N,) indicating the class of each sample.
                If None, all samples are assumed to belong to class 0.
        
        Returns:
            A dictionary containing the updated mean, covariance, and amount.
        '''
        N = tf.shape(features)[0]  # Number of samples
        C = self.num_class         # Number of classes

        if labels is None:
            labels = tf.zeros(N, dtype=tf.int32)

        # Convert labels to one-hot encoding
        onehot = tf.one_hot(labels, depth=C, dtype=tf.float32)  # Shape: (N, C)

        # Expand features to match the shape (N, C, A)
        features_expanded = tf.expand_dims(features, axis=1)  # Shape: (N, 1, A)
        NxCxFeatures = tf.tile(features_expanded, [1, C, 1])  # Shape: (N, C, A)

        # Multiply features by one-hot encoding to get features_by_sort
        features_by_sort = NxCxFeatures * tf.expand_dims(onehot, axis=-1)  # Shape: (N, C, A)

        # Calculate the amount of samples for each class and feature
        Amount_CxA = tf.reduce_sum(
            tf.expand_dims(onehot, axis=-1), axis=0
        )  # Shape: (C, A)
        Amount_CxA = tf.where(
            tf.equal(Amount_CxA, 0), tf.ones_like(Amount_CxA), Amount_CxA
        )

        # Calculate the mean for each class and feature
        ave_CxA = tf.reduce_sum(features_by_sort, axis=0) / Amount_CxA  # Shape: (C, A)

        # Calculate the variance
        var_temp = features_by_sort - tf.expand_dims(ave_CxA, axis=0) * tf.expand_dims(onehot, axis=-1)
        var_temp = tf.matmul(
            tf.transpose(var_temp, perm=[1, 2, 0]),  # Shape: (C, A, N)
            tf.transpose(var_temp, perm=[1, 0, 2])   # Shape: (C, N, A)
        ) / tf.expand_dims(Amount_CxA, axis=-1)  # Shape: (C, A, A)

        # Calculate weights for updating the covariance and mean
        sum_weight_CV = tf.reduce_sum(onehot, axis=0)  # Shape: (C,)
        sum_weight_CV = tf.expand_dims(tf.expand_dims(sum_weight_CV, axis=-1), axis=-1)  # Shape: (C, 1, 1)
        sum_weight_AV = tf.reduce_sum(onehot, axis=0)  # Shape: (C,)
        sum_weight_AV = tf.expand_dims(sum_weight_AV, axis=-1)  # Shape: (C, 1)

        weight_CV = sum_weight_CV / (sum_weight_CV + tf.expand_dims(self.Amount, axis=-1))
        weight_CV = tf.where(tf.math.is_nan(weight_CV), tf.zeros_like(weight_CV), weight_CV)

        weight_AV = sum_weight_AV / (sum_weight_AV + tf.expand_dims(self.Amount, axis=-1))
        weight_AV = tf.where(tf.math.is_nan(weight_AV), tf.zeros_like(weight_AV), weight_AV)

        # Calculate additional covariance
        additional_CV = weight_CV * (1 - weight_CV) * tf.matmul(
            tf.expand_dims(self.Ave - ave_CxA, axis=-1),  # Shape: (C, A, 1)
            tf.expand_dims(self.Ave - ave_CxA, axis=-2)   # Shape: (C, 1, A)
        )

        # Update covariance, mean, and amount
        new_CoVariance = (self.CoVariance * (1 - weight_CV) + var_temp * weight_CV) + additional_CV
        new_Ave = self.Ave * (1 - weight_AV) + ave_CxA * weight_AV
        new_Amount = self.Amount + tf.reduce_sum(onehot, axis=0)

        return {
            'Ave': new_Ave,
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }

    def update(self, dic):
        '''Update the internal state with new values.
        
        Args:
            dic: A dictionary containing 'Ave', 'CoVariance', and 'Amount'.
        '''
        self.Ave.assign(dic['Ave'])
        self.CoVariance.assign(dic['CoVariance'])
        self.Amount.assign(dic['Amount'])

    def invert(self):
        '''Calculate the inverse of the covariance matrix.'''
        self.CoVarianceInv.assign(tf.linalg.inv(self.CoVariance))

    def transform(self, features, labels):
        '''Transform the features using the covariance matrix.
        
        Args:
            features: A tensor of shape (N, A), where N is the number of samples
                      and A is the number of features.
            labels: A tensor of shape (N,) indicating the class of each sample.
        
        Returns:
            Transformed features of shape (N, A).
        '''
        CV = tf.gather(self.CoVariance, labels)  # Shape: (N, A, A)
        transformed = tf.matmul(
            tf.nn.l2_normalize(CV, axis=-1),
            tf.expand_dims(features, axis=-1)
        )  # Shape: (N, A, 1)
        return tf.squeeze(transformed, axis=-1)  # Shape: (N, A)


if __name__ == '__main__':
    pass
