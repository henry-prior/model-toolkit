from pathlib import Path
from typing import Union, Tuple

import sklearn.preprocessing as pre
import pandas as pd
import tensorflow as tf
import numpy as np
from functools import lru_cache

__all__ = ["DataBunch"]


class DataBunch:
    data: pd.DataFrame
    target: str
    feature_names: list = None
    categorical_columns: list = None
    label_strings: dict = None
    domain: dict = None

    def __init__(self,
                 path: Union[str, Path] = None,
                 data: pd.DataFrame = None,
                 target: str = None,
                 feature_names: list = None,
                 categorical_columns: list = None,
                 label_strings: dict = None,
                 domain: dict = None):
        if path is not None and data is None:
            data = load_data(path)
        elif data is not None and path is not None:
            raise ValueError(
                f"Exactly one of `path` or `data` must be specified, you specified both."
            )
        elif data is None and path is None:
            raise ValueError(
                f"Exactly one of `path` or `data` must be specified, you specified neither."
            )

        if target is None:
            raise ValueError("Must specify `target`.")

        self.path = path
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.categorical_columns = categorical_columns
        self.label_strings = label_strings
        self.domain = domain

        if self.feature_names is None:
            self.feature_names = [
                col for col in self.data.columns if col != self.target
            ]

        if self.categorical_columns is None:
            self.categorical_columns = infer_categorical_columns(self.data)

        if self.label_strings is None:
            self.label_strings = {
                val: str(val)
                for val in set(data[target].values)
            }

        self.encoders = get_feature_encoders(self.data, self.feature_names,
                                             self.categorical_columns)
        self.scalers = get_scalers(self.data, self.feature_names)

    def __call__(self):
        return self.data

    @property
    @lru_cache()
    def n_inputs(self):
        return len(self.feature_names)

    @property
    @lru_cache()
    def n_classes(self):
        return self.data[self.target].nunique()

    @lru_cache()
    def as_df(self,
              encode=True,
              scale=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.data.copy()
        data = self.transform(data, encode, scale)

        X_df = data[self.feature_names]
        y_df = data[self.target]
        return X_df, y_df

    @lru_cache()
    def as_numpy(self,
                 encode=True,
                 scale=True) -> Tuple[np.ndarray, np.ndarray]:
        X_df, y_df = self.as_df(encode, scale)
        X_np = X_df.values.astype(np.float32)
        y_np = y_df.values.astype(np.float32)
        return X_np, y_np

    @lru_cache()
    def as_tensor(self,
                  encode=True,
                  scale=True) -> Tuple[tf.Tensor, tf.Tensor]:
        X_np, y_np = self.as_numpy(encode, scale)
        X_tf = tf.constant(X_np, dtype=tf.float32)
        y_tf = tf.constant(y_np, dtype=tf.float32)
        return X_tf, y_tf

    def transform(self, X, encode=True, scale=True):
        if encode:
            X = self.encode(X)
        if scale:
            X = self.scale(X)
        return X

    def encode(self, X):
        if isinstance(X, tf.Tensor):
            raise ValueError(
                f"Inputs of type 'tf.Tensor' are not yet supported.")
        if X.ndim == 1:  # if input is single array
            return self.encode(np.array([X]))[0]
        X = X.copy()
        for i, feature in enumerate(self.feature_names):
            encoder = self.encoders[feature]
            if encoder is not None:
                if isinstance(X, pd.DataFrame):
                    X[feature] = encoder.transform(X[feature].values)
                elif isinstance(X, np.ndarray):
                    X[:, i] = encoder.transform(X[:, i])
        return X

    def scale(self, X):
        if isinstance(X, tf.Tensor):
            raise ValueError(
                f"Inputs of type 'tf.Tensor' are not yet supported.")
        if X.ndim == 1:  # if input is single array
            return self.scale(np.array([X]))[0]
        X = X.copy()
        for i, feature in enumerate(self.feature_names):
            scaler = self.scalers[feature]
            if scaler is not None:
                if isinstance(X, pd.DataFrame):
                    X[feature] = scaler.transform(X[feature].values.reshape(
                        -1, 1)).squeeze()
                elif isinstance(X, np.ndarray):
                    X[:, i] = scaler.transform(X[:, i].reshape(-1,
                                                               1)).squeeze()
        return X

    @classmethod
    def from_config(cls, config: dict = None, **kwargs):
        if config is None:
            config = kwargs
        return cls(**config)


def load_data(path: Union[str, Path]):
    data_path = Path(path) if not isinstance(path, Path) else path
    file_ext = data_path.suffix[1:]
    if file_ext in ['csv', 'txt']:
        data = pd.read_csv(data_path)
    elif file_ext == 'feather':
        data = pd.read_feather(data_path)
    else:
        raise DataFileTypeError(f"Unrecognized data file type '{file_ext}'.")
    return data


def infer_categorical_columns(data_df):
    categorical_columns = set()
    feature_list = data_df.columns.tolist()
    dtype_list = data_df.dtypes.tolist()
    n_unique_list = data_df.nunique().tolist()
    max_list = data_df.max().tolist()
    for feature, dtype, n_unique, max_ in zip(feature_list, dtype_list,
                                              n_unique_list, max_list):
        if dtype == np.object:  # feature is string
            categorical_columns.add(feature)
        elif np.issubdtype(dtype, np.number):  # feature is numerical
            if n_unique == max_ - 1:  # feature values are [0, 1, 2, ..., n-1]
                categorical_columns.add(feature)
    return categorical_columns


def get_feature_encoders(data_df, features, categorical_columns):
    encoders = dict()
    for feature in features:
        if feature in categorical_columns:
            encoder = pre.LabelEncoder()
            feature_values = data_df[feature].values
            encoder.fit(feature_values)
            encoders[feature] = encoder
        else:
            encoders[feature] = None
    return encoders


def get_scalers(data_df, features):
    scalers = dict()
    for feature in features:
        scaler = get_scaler('standard')
        feature_values = data_df[feature].values.reshape(-1, 1)
        scaler.fit(feature_values)
        scalers[feature] = scaler
    return scalers


def get_scaler(scaler_type):
    if scaler_type == 'standard':
        return pre.StandardScaler()
    elif scaler_type == 'min_max':
        return pre.MinMaxScaler()
    elif scaler_type == 'robust':
        return pre.RobustScaler()
    elif scaler_type == 'normalize':
        return pre.Normalizer()
    else:
        raise ValueError(f"Scaler type '{scaler_type}' not recognized.")

class DataFileTypeError(Exception):
    pass