from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from functools import partial, update_wrapper

import tensorflow.keras.backend as K
import tensorflow_addons as tfa

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def glu(act, n_units):
    """Generalized linear unit nonlinear activation."""
    return act[:, :n_units] * tf.keras.activations.sigmoid(act[:, n_units:])


def reduce_entropy(mask_values, epsilon=0.001, num_decision_steps=6):
    entropy = K.mean(K.sum(-mask_values * K.log(mask_values + epsilon), axis=1),
                     keepdims=True) / (num_decision_steps - 1)
    return entropy

def ones_like_layer(layer):
    return tf.keras.backend.ones_like(layer)


class EntropyRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, entropy_function, sparsity_loss_weight=0.0001):
        self.entropy_fuction = entropy_function
        self.sparsity_loss_weight = sparsity_loss_weight

    def __call__(self, x):
        return self.sparsity_loss_weight * self.entropy_fuction(x)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "entropy_fuction" : self.entropy_fuction,
                "sparsity_loss_weight": self.sparsity_loss_weight}


def build_tabnet(num_features,
                 feature_dim,
                 output_dim,
                 num_decision_steps,
                 relaxation_factor,
                 batch_momentum,
                 virtual_batch_size,
                 num_classes,
                 is_training,
                 epsilon=0.00001):
    raw_glu = wrapped_partial(glu, n_units=feature_dim)
    raw_reduce_entropy = wrapped_partial(reduce_entropy,
                                         epsilon=epsilon,
                                         num_decision_steps=num_decision_steps)
    normalized_glu = lambda x: np.sqrt(0.5) * (raw_glu(x[1]) + x[0])
    decision_func = lambda x: x[:, :output_dim]

    inputs = tf.keras.layers.Input(shape=(num_features,), dtype='float32')
    features = tf.keras.layers.BatchNormalization(
        momentum=batch_momentum)(inputs)

    # Initializes decision-step dependent variables.
    output_aggregated = []
    masked_features = features
    complementary_aggregated_mask_values = \
        tf.keras.layers.Lambda(ones_like_layer)(inputs)

    if is_training:
        v_b = virtual_batch_size
    else:
        v_b = 1

    for ni in range(num_decision_steps):

        # Feature transformer with two shared and two decision step dependent
        # blocks is used below.
        reuse_flag = (ni > 0)

        if not reuse_flag:
            transform_f1_base = tf.keras.layers.Dense(feature_dim * 2,
                                                      use_bias=False)
            transform_f2_base = tf.keras.layers.Dense(feature_dim * 2,
                                                      use_bias=False)

        transform_f1 = transform_f1_base(masked_features)
        transform_f1 = tf.keras.layers.BatchNormalization(
            momentum=batch_momentum, virtual_batch_size=v_b)(transform_f1)
        transform_f1 = tf.keras.layers.Lambda(raw_glu)(transform_f1)

        transform_f2 = transform_f2_base(transform_f1)
        transform_f2 = tf.keras.layers.BatchNormalization(
            momentum=batch_momentum,
            virtual_batch_size=v_b)(transform_f2)
        transform_f2 = normalized_glu([transform_f1, transform_f2])

        transform_f3 = tf.keras.layers.Dense(feature_dim * 2,
                                             use_bias=False)(transform_f2)
        transform_f3 = tf.keras.layers.BatchNormalization(
            momentum=batch_momentum,
            virtual_batch_size=v_b)(transform_f3)
        transform_f3 = normalized_glu([transform_f2, transform_f3])

        transform_f4 = tf.keras.layers.Dense(feature_dim * 2,
                                             use_bias=False)(transform_f3)
        transform_f4 = tf.keras.layers.BatchNormalization(
            momentum=batch_momentum,
            virtual_batch_size=v_b)(transform_f4)
        transform_f4 = normalized_glu([transform_f3, transform_f4])

        if ni > 0:
            decision_out = tf.keras.layers.Lambda(decision_func)(transform_f4)
            decision_out = tf.keras.layers.Activation('relu')(decision_out)

            # Decision aggregation.
            output_aggregated.append(decision_out)

        features_for_coef = tf.keras.layers.Lambda(decision_func)(transform_f4)

        if ni < num_decision_steps - 1:
            # Determines the feature masks via linear and nonlinear
            # transformations, taking into account of aggregated feature use.
            mask_values = tf.keras.layers.Dense(num_features,
                                                use_bias=False)(features_for_coef)
            mask_values = tf.keras.layers.BatchNormalization(
                momentum=batch_momentum,
                virtual_batch_size=v_b)(mask_values)
            mask_values = tf.keras.layers.Multiply()(
                [mask_values, complementary_aggregated_mask_values])
            mask_values = tf.keras.layers.Activation(tfa.activations.sparsemax,
                                                     activity_regularizer=EntropyRegularizer(
                                                         raw_reduce_entropy))(mask_values)

            # Relaxation factor controls the amount of reuse of features between
            # different decision blocks and updated with the values of
            # coefficients.
            complementary_aggregated_mask_values = tf.keras.layers.Multiply()(
                [complementary_aggregated_mask_values,
                 relaxation_factor - mask_values])

            # Feature selection.
            masked_features = tf.keras.layers.Multiply()([mask_values, features])

    output_aggregated = tf.keras.layers.Add()(output_aggregated)

    logits = tf.keras.layers.Dense(num_classes,
                                   use_bias=False,
                                   name='logits')(output_aggregated)

    logits = tf.keras.layers.Activation('softmax')(logits)

    model = tf.keras.Model(inputs, logits)

    return model