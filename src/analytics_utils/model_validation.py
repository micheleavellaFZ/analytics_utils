import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar, Any
import pandas as pd

X = TypeVar("X")  # For input data (x, x_test)


class Model(Generic[X], ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, x_test: X, y_test: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x: X) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class Validation(Generic[X]):
    def __init__(
        self, loss_function: Callable[[np.ndarray, np.ndarray], float]
    ) -> None:
        self.loss_function = loss_function
        self.df_results = None

    def validate(
        self,
        model_names: list[str],
        models: list[Model],
        x_train: list[X],
        y_train: list[np.ndarray],
        x_test: list[X],
        y_test: list[np.ndarray],
    ) -> pd.DataFrame:
        if len(model_names) != len(models):
            raise ValueError("Length of model_names must match length of models")

        if (
            len(x_train) != len(y_train)
            or len(x_test) != len(y_test)
            or len(x_test) != len(x_train)
        ):
            raise ValueError(
                "Length of input data (x) must match length of target data (y)"
            )

        out = []
        for j in range(len(models)):
            for i in range(len(x_train)):
                df = self.unit_validate(
                    model_names[j],
                    models[j],
                    x_train[i],
                    y_train[i],
                    x_test[i],
                    y_test[i],
                )
                df["dataset_n"] = i
                out.append(df)

        self.df_results = pd.concat(out, ignore_index=True)
        return self.df_results.copy()

    def unit_validate(
        self,
        model_name: str,
        model: Model,
        x_train: X,
        y_train: np.ndarray,
        x_test: X,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        model.reset()
        model.train(x_train, y_train)
        y_pred = model.predict(x_test)
        err = self.loss_function(y_test, y_pred)
        return pd.DataFrame([{"model": model_name, "error": err}])


def loss_MSE(real: np.ndarray, pred: np.ndarray) -> float:
    return ((pred - real) ** 2).mean()


def loss_MAPE(real: np.ndarray, pred: np.ndarray) -> float:
    epsilon = 1e-10
    return (np.abs((real - pred) / (np.abs(real) + epsilon))).mean()
