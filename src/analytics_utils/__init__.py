from .database_connection import DatabaseConnection
from .data_source import DataSource
from .dataframe_comparer import compare_dataframe, ComparisonResult
from .run_python_script import run_python_script
from .send_email import send_email
from .model_validation import (
    Model,
    Validation,
    loss_MSE,
    loss_MAPE,
    KFoldValidation,
    LFOValidation,
)
import logging

logging.basicConfig(
    format="{asctime} | [{filename}:{lineno} - {funcName}()] | {levelname} | {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
my_logger = logging.getLogger(__name__)
