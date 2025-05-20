"""
Model validation framework for comparing models.

This module provides a structured approach for validating models using various validation strategies.
It defines abstract base classes for models and validation methods, along with concrete implementations for common
validation techniques such as k-fold cross-validation and leave-future-out validation.

Key Features:
- Abstract Model interface for creating compatible models
- Customizable validation strategies through the Validation base class
- K-fold cross-validation implementation (KFoldValidation)
- Leave-future-out validation for time series data (LFOValidation)
- Common loss functions (MSE, MAPE) for performance evaluation
- Type-safe implementation using generics for handling different types of input data
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar, Any
import pandas as pd
import random

DataPoint = TypeVar("DataPoint")  # For input data (x, x_test)


class Model(Generic[DataPoint], ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, x_train: list[DataPoint], y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x: list[DataPoint]) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class Validation(Generic[DataPoint], ABC):
    def __init__(
        self, loss_function: Callable[[np.ndarray, np.ndarray], float]
    ) -> None:
        self.loss_function = loss_function
        self.df_results = None

    def validate(
        self,
        model_names: list[str],
        models: list[Model],
        x: list[DataPoint],
        y: np.ndarray,
    ) -> pd.DataFrame:
        if len(model_names) != len(models):
            raise ValueError("Length of model_names must match length of models")

        if len(x) != len(y):
            raise ValueError(
                "Length of input data (x) must match length of target data (y)"
            )

        out = []
        folds = self.dataset_folds(len(x))

        for i, fold in enumerate(folds):
            for j in range(len(models)):
                x_train = [x[ix] for ix in fold[0]]
                y_train = y[fold[0]]
                x_test = [x[ix] for ix in fold[1]]
                y_test = y[fold[1]]

                df = self.unit_validate(
                    model_names[j], models[j], x_train, y_train, x_test, y_test
                )
                df["fold_n"] = i
                out.append(df)

        self.df_results = pd.concat(out, ignore_index=True)
        return self.df_results.copy()

    def unit_validate(
        self,
        model_name: str,
        model: Model,
        x_train: list[DataPoint],
        y_train: np.ndarray,
        x_test: list[DataPoint],
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        model.reset()
        model.train(x_train, y_train)
        y_pred = model.predict(x_test)
        err = self.loss_function(y_test, y_pred)
        return pd.DataFrame([{"model": model_name, "error": err}])

    @abstractmethod
    def dataset_folds(self, length: int) -> list[tuple[list[int], list[int]]]:
        pass


def loss_MSE(real: np.ndarray, pred: np.ndarray) -> float:
    return ((pred - real) ** 2).mean()


def loss_MAPE(real: np.ndarray, pred: np.ndarray) -> float:
    epsilon = 1e-10
    return (np.abs((real - pred) / (np.abs(real) + epsilon))).mean()


class KFoldValidation(Validation):
    def __init__(
        self, loss_function: Callable[[np.ndarray, np.ndarray], float], n_folds: int
    ) -> None:
        super().__init__(loss_function)
        self.n_folds = n_folds

    def dataset_folds(self, length: int) -> list[tuple[list[int], list[int]]]:
        """
        Create train-test splits for k-fold cross-validation.

        Randomly divides a dataset of given length into self.n_folds folds,
        then creates train-test pairs where each fold serves as the test set once
        while all other folds combined form the training set.

        Args:
            length: Total number of samples in the dataset

        Returns:
            A list of (train_indices, test_indices) tuples, where each tuple
            represents one fold's train-test split of indices.
        """
        if length < self.n_folds:
            raise ValueError(
                f"Dataset length ({length}) must be at least as large as the number of folds ({self.n_folds})"
            )
        arr = list(range(length))
        random.shuffle(arr)

        part_size = len(arr) // self.n_folds
        remainder = len(arr) % self.n_folds
        folds: list[list[int]] = []
        start = 0

        for i in range(self.n_folds):
            end = start + part_size + (1 if i < remainder else 0)
            folds.append(arr[start:end])
            start = end

        out = []
        for f in range(self.n_folds):
            test = folds[f]
            train = []
            for i in range(self.n_folds):
                if i != f:
                    train += folds[i]
            out.append((train, test))

        return out


class LFOValidation(Validation):
    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        min_train_len: int,
        test_len: int,
    ) -> None:
        super().__init__(loss_function)
        self.min_train_len = min_train_len
        self.test_len = test_len

    def dataset_folds(self, length: int) -> list[tuple[list[int], list[int]]]:
        arr = list(range(length))
        out = []
        for i in range(self.min_train_len, length - self.test_len):
            train = arr[:i]
            test = arr[i : i + self.test_len]
            out.append((train, test))
        return out
