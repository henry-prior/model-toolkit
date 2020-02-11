from typing import List, Dict, Tuple, Callable
import pickle
import os
import pandas as pd
from xgboost import XGBClassifier
from pathlib import Path

from .adversarial_functions import *
from .databunch import *
from ..models import *
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from tensorflow.python.training.tracking.tracking import AutoTrackable
LoadedKerasModel = AutoTrackable

__all__ = ["Trainer", "XGBClassifierWrapper"]


def get_adversarial_function(model: Union[tf.Module, tf.keras.Model]):
    if isinstance(model, tf.keras.Model):
        if len(model.get_weights()) == 1:
            return adv_perturbation_closed_form
        else:
            return adv_perturbation_pgd
    else:
        raise TypeError(
            f"Adversarial training not supported for specified model of type '{type(model)}'."
        )


class XGBClassifierWrapper:
    def __init__(self, n_inputs=None, *args, **kwargs):
        self._classifier = XGBClassifier(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._classifier.predict_proba(*args, **kwargs)[:, [1]]

    def fit(self, *args, **kwargs):
        return self._classifier.fit(*args, **kwargs)

    def get_booster(self, *args, **kwargs):
        return self._classifier.get_booster(*args, **kwargs)

    
@dataclass
class Trainer:
    optimizer: str = 'adam'
    loss: Union[str, tf.keras.losses.Loss] = 'BinaryCrossentropy'
    metrics: List[str] = field(default_factory=lambda: ['BinaryAccuracy'])

    train_percent: float = 0.8
    test_percent: float = 0.2

    num_epochs: int = 30
    batch_size: int = 32
    drop_remainder_batch: bool = False
    min_delta: float = 0
    patience: int = 10
    adversarial_args: dict = None

    verbose: bool = True
    log_dir: str = None

    _trained: bool = False
    _converged: bool = False
    _adversarial_training: bool = False
    _train_auc: tf.keras.metrics.AUC = None
    _test_auc: tf.keras.metrics.AUC = None
    _train_metrics: List = None
    _test_metrics: List = None

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.optimizer, self.loss, self.metrics,
                    self.train_percent, self.test_percent,
                    self.num_epochs, self.batch_size, self.drop_remainder_batch,
                    self.min_delta, self.patience, self.adversarial_args,
                    self.verbose, self.log_dir, self._trained, self._converged) == \
                   (other.optimizer, other.loss, other.metrics,
                    other.train_percent, other.test_percent,
                    other.num_epochs, other.batch_size, other.drop_remainder_batch,
                    other.min_delta, other.patience, other.adversarial_args,
                    other.verbose, other.log_dir, other._trained, other._converged)
        return False

    def __hash__(self):
        return (hash(self.optimizer) ^ hash(self.loss)
                ^ hash(tuple(self.metrics)) ^ hash(self.train_percent)
                ^ hash(self.test_percent) ^ hash(self.num_epochs)
                ^ hash(self.batch_size) ^ hash(self.drop_remainder_batch)
                ^ hash(self.min_delta) ^ hash(self.patience)
                ^ hash_dict(self.adversarial_args) ^ hash(self.verbose)
                ^ hash(self.log_dir) ^ hash(self._trained)
                ^ hash(self._converged))

    def __post_init__(self):
        """

        :param loss
        """

        optimizer_name = self.optimizer
        self.optimizer = tf.keras.optimizers.get(optimizer_name)

        self._init_metrics()

        if self.train_percent + self.test_percent > 1:
            raise Exception("`train_percent + test_percent` > 1")

        if self.adversarial_args is not None and self.adversarial_args[
            'eps'] > 0:
            self._adversarial_training = True
        else:
            self._adversarial_training = False

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_dir = Path(self.log_dir)
            self.log_file = self.log_dir/'log.csv'
            self.roc_file = self.log_dir/'roc.pkl'

    def _init_metrics(self):
        loss_name = self.loss
        self.loss = tf.keras.losses.get(loss_name)

        self._train_loss = tf.keras.metrics.get(loss_name)
        self._train_loss._name = "Train Loss"
        self._train_auc = tf.keras.metrics.AUC(name="Train AUC")
        self._train_metrics = [self._train_loss, self._train_auc]

        self._test_loss = tf.keras.metrics.get(loss_name)
        self._test_loss._name = "Test Loss"
        self._test_auc = tf.keras.metrics.AUC(name="Test AUC")
        self._test_metrics = [self._test_loss, self._test_auc]
        if self.metrics is not None:
            if not isinstance(self.metrics, list):
                raise ConfigError(f"Expected type of parameter `metrics` "
                                  f"to be 'list', not '{type(self.metrics)}'.")

            for metric_name in self.metrics:
                train_metric = tf.keras.metrics.get(metric_name)
                train_metric._name = f"Train {metric_name}"
                self._train_metrics.append(train_metric)

                test_metric = tf.keras.metrics.get(metric_name)
                test_metric._name = f"Test {metric_name}"
                self._test_metrics.append(test_metric)

    @tf.function
    def train_step(self, model: tf.keras.Model, observations: np.ndarray,
                   labels: np.ndarray):
        with tf.GradientTape() as tape:
            predictions = tf.squeeze(model(observations))
            loss = self.loss(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        self.update_train_metrics(labels, predictions)

    @tf.function
    def test_step(self, model, observations, labels):
        predictions = tf.squeeze(model(observations))
        self.update_test_metrics(labels, predictions)

    def update_train_metrics(self, labels, predictions):
        for metric in self._train_metrics:
            metric.update_state(labels, predictions)

    def update_test_metrics(self, labels, predictions):
        for metric in self._test_metrics:
            metric.update_state(labels, predictions)

    def get_metrics(self):
        return {**self.get_train_metrics(), **self.get_test_metrics()}

    def get_train_metrics(self):
        return compile_metrics(self._train_metrics)

    def get_test_metrics(self):
        return compile_metrics(self._test_metrics)

    def reset_metrics(self):
        reset_metrics(self._train_metrics)
        reset_metrics(self._test_metrics)

    def get_roc_metrics(self):
        train_true_positives = self._train_auc.true_positives.numpy()[::-1]
        train_tpr = train_true_positives/train_true_positives[-1]

        train_false_positives = self._train_auc.false_positives.numpy()[::-1]
        train_fpr = train_false_positives/train_false_positives[-1]

        test_true_positives = self._test_auc.true_positives.numpy()[::-1]
        test_tpr = test_true_positives/test_true_positives[-1]

        test_false_positives = self._test_auc.false_positives.numpy()[::-1]
        test_fpr = test_false_positives/test_false_positives[-1]

        roc_metrics = dict(train_tpr=train_tpr,
                           train_fpr=train_fpr,
                           test_tpr=test_tpr,
                           test_fpr=test_fpr)
        return roc_metrics

    def fit(self, model: tf.keras.Model, databunch: DataBunch):

        self.history_df = pd.DataFrame()
        self.reset_metrics()

        if isinstance(model, (tf.Module, tf.keras.Model, LoadedKerasModel)):
            return self.tf_fit(model, databunch)
        elif isinstance(model, XGBClassifierWrapper):
            return self.xgb_fit(model, databunch)
        else:
            raise TypeError(f"Unsupported model type {type(model).__name__}")

    def xgb_fit(self, model: XGBClassifierWrapper, databunch: DataBunch):
        self.reset_metrics()
        X, y = databunch.as_numpy()
        X_train, X_test, y_train, y_test = \
            train_test_split(
                X, y,
                train_size=self.train_percent,
                test_size=self.test_percent,
                random_state=42
            )
        model.fit(X_train, y_train)
        train_predictions = predict(model, X_train)
        self.update_train_metrics(y_train, train_predictions)
        test_predictions = predict(model, X_test)
        self.update_test_metrics(y_test, test_predictions)
        self.update_history(0)
        self.log()
        if self.verbose:
            self.print_most_recent_metrics()
        return self.get_metrics()

    def tf_fit(self, model: tf.keras.Model, databunch: DataBunch):
        if self._converged:
            return self.get_metrics()

        adversarial_function = (get_adversarial_function(model)
                                if self._adversarial_training else None)

        X, y = databunch.as_numpy()
        X_train, X_test, y_train, y_test = \
            train_test_split(
                X, y,
                train_size=self.train_percent,
                test_size=self.test_percent,
                random_state=42
            )
        train_batches, test_batches = Trainer.batch(X_train, y_train, X_test,
                                                    y_test, self.batch_size,
                                                    self.drop_remainder_batch)

        patience_count = 0
        for epoch in range(self.num_epochs):
            patience_count, stop = \
                early_stopping_check(
                    self.history_df,
                    self.min_delta,
                    patience_count,
                    self.patience
                )

            if stop:
                print("Early stopping at epoch {}".format(epoch))
                self._converged = True
                self.update_history(epoch)
                self.log()
                break

            self.reset_metrics()

            # train
            for observations, labels in train_batches:
                # ADVERSARIAL PERTURBATION of each MINI-BATCH
                if self._adversarial_training:
                    observations = adversarial_function(
                        model,
                        x=observations,
                        y=labels,
                        **self.adversarial_args)
                self.train_step(model, observations, labels)

            # test
            for observations, labels in test_batches:
                self.test_step(model, observations, labels)

            self.update_history(epoch)
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                self.log()
                if self.verbose:
                    self.print_most_recent_metrics()
        self._trained = True
        return self.get_metrics()

    def eval(self, model, databunch):

        self.reset_metrics()

        if isinstance(model, tf.keras.Model):
            return self.tf_eval(model, databunch)
        else:
            raise TypeError(f"Unsupported model type {type(model).__name__}")

    def tf_eval(self, model, databunch):
        X, y = databunch.as_numpy()
        X_train, X_test, y_train, y_test = \
            train_test_split(
                X, y,
                train_size=self.train_percent,
                test_size=self.test_percent,
                random_state=42
            )
        _, test_batches = Trainer.batch(X_train, y_train, X_test, y_test,
                                        self.batch_size,
                                        self.drop_remainder_batch)

        for observations, labels in test_batches:
            self.test_step(model, observations, labels)

        return self.get_test_metrics()

    def update_history(self, epoch):
        metrics = self.get_metrics()
        metrics['epoch'] = epoch
        self.history_df = self.history_df.append(metrics, ignore_index=True)

    def log(self):
        if self.log_dir:
            self.history_df.to_csv(self.log_file)
            with open(self.roc_file, 'w+b') as roc_pkl:
                pickle.dump(self.get_roc_metrics(), roc_pkl)

    def print_most_recent_metrics(self):
        history = self.history_df.iloc[-1]
        print(f"Epoch: {history['epoch']:g}")
        for train_metric, test_metric in zip(self._train_metrics,
                                             self._test_metrics):
            print(f"{train_metric.name}: {history[train_metric.name]:.3f}\t"
                  f"{test_metric.name}: {history[test_metric.name]:.3f}")
        print()

    @property
    def trained(self):
        return self._trained

    @property
    def converged(self):
        return self._converged

    @staticmethod
    def batch(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        drop_remainder_batch: bool = False
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        train_ds = \
            tf.data.Dataset.from_tensor_slices(
                (X_train, y_train)
            ).shuffle(10000).batch(batch_size=batch_size, drop_remainder=drop_remainder_batch)

        test_ds = \
            tf.data.Dataset.from_tensor_slices(
                (X_test, y_test)
            ).batch(batch_size=batch_size, drop_remainder=drop_remainder_batch)

        return train_ds, test_ds

    def get_config(self):
        return dict(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            train_percent=self.train_percent,
            test_percent=self.test_percent,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            drop_remainder_batch=self.drop_remainder_batch,
            min_delta=self.min_delta,
            patience=self.patience,
            adversarial_args=self.adversarial_args,
            verbose=self.verbose,
            log_dir=self.log_dir,
        )

    @classmethod
    def from_config(cls, config: Dict = None, **kwargs):
        if config is None:
            config = kwargs
        return cls(**config)


def create_metric_name(name):
    return ' '.join(name.split('_')).title()


def compile_metrics(metrics: List) -> Dict:
    metric_dict = dict()
    for metric in metrics:
        metric_dict[metric.name] = metric.result().numpy()
    return metric_dict


def reset_metrics(metrics):
    for metric in metrics:
        metric.reset_states()


def hash_dict(d: Dict) -> int:
    return hash(tuple(sorted(d.items()))) if d is not None else hash(None)


def early_stopping_check(history_df, min_delta, patience_count, patience):
    if history_df.shape[0] > 1:
        if history_df.iloc[-2]['Test AUC'] - history_df.iloc[-1][
            'Test AUC'] > min_delta:
            patience_count = 0
        else:
            patience_count += 1
    return patience_count, (patience_count > patience)


def predict(model: Callable, x: np.ndarray, *args, **kwargs):
    """Calls the model on an example.

    Args:
        model: A `Model` object.
        x: A 1-dimensional or 2-dimensional array.

    Returns:
        If input `x` is a 1-dimensional array, a float is returned. If input `x` is a
            2-dimensional array of length N, a 1-dimensional array of length N is returned.
    """
    if isinstance(x, np.ndarray):
        x = x.astype(np.float32)
        res = predict_np(model, x, *args, **kwargs)
        if isinstance(res, tf.Tensor):
            res = res.numpy()
        return res
    elif isinstance(x, tf.Tensor):
        x = tf.cast(x, tf.float32)
        return predict_tf(model, x, *args, **kwargs)
    else:
        raise TypeError(
            f"Invalid type '{type(x).__name__}' for input `x`, expected np.ndarray or tf.Tensor"
        )


def predict_np(model: Callable, x: np.ndarray, *args, **kwargs):
    x_dim = x.ndim
    if x_dim == 1:
        x = np.array([x], dtype=np.float32)
    out = np.squeeze(np.max(model(x, *args, **kwargs)))
    return out


def predict_tf(model: Callable, x: tf.Tensor, *args, **kwargs):
    x_dim = len(x.shape)
    if x_dim == 1:
        x = tf.convert_to_tensor([x], dtype=tf.float32)
        return model(x, *args, **kwargs)[0]
    else:
        return tf.squeeze(model(x, *args, **kwargs))
