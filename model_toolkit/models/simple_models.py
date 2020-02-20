import tensorflow as tf
from typing import Union, Callable, Dict
from pathlib import Path
from collections import defaultdict
from ..utils.io import load_pickle
from tensorflow.python.training.tracking.tracking import AutoTrackable

LoadedKerasModel = AutoTrackable


class SingleLayerNetwork(tf.keras.Sequential):
    """Implements a single layer dense network in Keras that defaults to be a
    logistic regression. Optional l1 and l2 regularization

    :param n_inputs: number of features
    :param activation: activation function for final output
    :param name: optional: name for model
    :param l1_lambda: lambda if applying l1 regularization. 0.0 for none.
    :param l2_lambda: lambda if applying l2 regularization. 0.0 for none."""

    def __init__(self,
                 n_inputs: int = None,
                 activation: Union[Callable, str] = tf.sigmoid,
                 l1_lambda: int = 0.0,
                 l2_lambda: int = 0.0,
                 name: str = None):
        logit_layer = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_lambda,
                                                          l2=l2_lambda),
            input_shape=(n_inputs,))
        activation_layer = tf.keras.layers.Activation(activation)
        super().__init__([logit_layer, activation_layer], name)


class MultiLayerNetwork(tf.keras.Sequential):
    """Implements a multi layer dense network in Keras with mulitclass
    support. Optional l1 and l2 regularization

    :param n_inputs: number of features
    :param n_outputs: number of target classes
    :param n_layers: number of layers
    :param activation: activation function for final output
    :param name: optional: name for model
    :param l1_lambda: lambda if applying l1 regularization. 0.0 for none.
    :param l2_lambda: lambda if applying l2 regularization. 0.0 for none."""

    def __init__(self,
                 n_inputs: int = None,
                 n_outputs: int = None,
                 n_layers: int = 1,
                 l1_lambda: int = 0.0,
                 l2_lambda: int = 0.0,
                 name: str = None):
        layers_list = []
        for _ in range(n_layers - 1):
            next_layer = tf.keras.layers.Dense(n_inputs,
                                               kernel_initializer=tf.keras.initializers.GlorotUniform(
                                                   seed=42),
                                               kernel_regularizer=tf.keras.regularizers.L1L2(
                                                   l1=l1_lambda,
                                                   l2=l2_lambda),
                                               input_shape=(n_inputs,))
            layers_list.append(next_layer)

        logit_layer = tf.keras.layers.Dense(
            n_outputs,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_lambda,
                                                          l2=l2_lambda),
            activation='sigmoid',
            input_shape=(n_inputs,))
        layers_list.append(logit_layer)
        super().__init__(layers_list, name)


class TensorFlowLogistic(tf.Module):
    """Implements a logisitic regression in base TensorFlow. Used to test
    base TF
    functionality

    :param n_inputs: number of features
    :param activation: activation function for final output
    """

    def __init__(
        self,
        n_inputs: int,
        activation: Callable = tf.sigmoid,
        name: str = None,
    ):
        super(TensorFlowLogistic, self).__init__(name)
        self.weights = tf.Variable(
            tf.random.normal(shape=(n_inputs, 1), dtype=tf.float32))
        self.biases = tf.Variable(
            tf.random.normal(shape=(1,), dtype=tf.float32))
        self.activation = activation

    @tf.function
    def __call__(self, x):
        logits = tf.add(tf.matmul(x, self.weights), self.biases)
        return self.activation(logits)

    @property
    def trainable_variables(self):
        return [self.weights, self.biases]


class SVM(tf.keras.Sequential):
    """Implements a Support Vector Machine in Keras

    :param n_inputs: number of features
    :param c: lambda for soft margin
    :param name: optional: name for model
    """

    def __init__(self, n_inputs: int, c: int = 0.1, name: str = None):
        l2_layer = tf.keras.layers.Dense(
            1,
            activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(c),
            input_shape=(n_inputs,))
        activation_layer = tf.keras.layers.Activation('sigmoid')
        super().__init__([l2_layer, activation_layer], name)


_model_class_map: Dict[str, Callable] = defaultdict(
    lambda: None,
    dict(
        SingleLayerNetwork=SingleLayerNetwork,
        SVM=SVM,
    ),
)


def from_config(config: dict = None, **kwargs):
    if config is None:
        config = kwargs

    if 'model_path' in config:
        model_path = config['model_path']
        return load_model(model_path)

    if 'model_type' not in config:
        raise ConfigError("Missing `model_type` parameter.")
    model_type = config['model_type']
    model_class = _model_class_map[model_type]
    if model_class is None:
        raise ValueError(
            f"Unrecognized value '{model_type}' for parameter `model_type`.")

    del config['model_type']
    model = model_class(**config)

    if 'weights_path' in config:
        weights_path = config['weights_path']
        load_weights(model, weights_path)

    return model


def save_model(model: Callable, model_dir: Path, name: str = 'model'):
    """Save model to file system.

    Args:
        model: A `Model` object. Currently supports `tensorflow.keras`
        models, `tensorflow` models,
            and `xgboost` models.
        model_dir: The directory in which to save the model.
        name: The name of the model. Default is 'model'.

    Returns:
        The path to the saved model. For `tensorflow.keras` models, format
        is 'MODEL_DIR/NAME.h5py.
        For `tensorflow` models, format is `MODEL_DIR/NAME/`. For `xgboost`
        models, format is
        'MODEL_DIR/NAME.pkl`.
    """
    if isinstance(model, tf.keras.Model) or isinstance(model,
                                                       LoadedKerasModel):
        model_path = (model_dir / f'{name}.h5py').absolute()
        model.save(str(model_path))
    elif isinstance(model, tf.Module):
        model_path = (model_dir / name).absolute()
        tf.saved_model.save(model, str(model_path))
    else:
        raise ValueError(f"Unsupported model type '{type(model).__name__}'.")
    return model_path


def save_weights(model: tf.keras.Model,
                 weights_dir: Path,
                 name: str = 'model') -> Path:
    """Save model weights to file system.

    Args:
        model: A `Model` object. Currently supports tensorflow.keras models.
        weights_dir: The directory in which to save the model weights.
        name: The name of the model. Default is 'model'.

    Returns:
        The path to the saved model. Format is 'WEIGHTS_DIR/NAME_weights.h5'.
    """
    if isinstance(model, tf.keras.Model):
        weights_path = (weights_dir / f'{name}_weights.h5').absolute()
        model.save_weights(str(weights_path))
    else:
        raise ValueError(f"Saving model weights not supported for "
                         f"specified model of type '{type(model).__name__}'.")
    return weights_path


def load_model(model_path: Path) -> Union[tf.Module, tf.keras.Model]:
    """Load model from file system.

    Args:
        model_path: The path to the model. Supported extensions: `h5py`,
        `h5`, `pkl`.
            Also supports folders containing `tensorflow` models.
    """
    model_ext = model_path.suffix[1:]
    model_path = str(model_path)
    if model_ext:
        if model_ext in {'h5py', 'h5'}:
            model = tf.keras.models.load_model(model_path)
        elif model_ext == 'pkl':
            model = load_pickle(model_path)
        else:
            raise ValueError(
                f"Unsupported model file extension '{model_ext}'.")
    else:  # no suffix => specified path is folder
        model = tf.saved_model.load(model_path)
    return model


def load_weights(model: tf.keras.Model, weights_path: Path):
    """Load model weights from file system.

    Args:
        model: A model, currently only supports `tf.keras.Model` objects
        weights_path: The path to the model weights. Supported extensions:
        `h5`.
    """
    weights_ext = weights_path.suffix[1:]
    weights_path = str(weights_path)
    if weights_ext == 'h5':
        if isinstance(model, tf.keras.Model):
            model.load_weights(weights_path)
        else:
            raise ValueError(
                f"Cannot load weights into model of type '"
                f"{type(model).__name__}'."
            )
    else:
        raise ValueError(
            f"Unsupported weights file extension '{weights_ext}'.")
    return model


class ConfigError(Exception):
    pass
