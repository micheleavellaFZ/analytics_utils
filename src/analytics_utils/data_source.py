from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
import pandera as pa
import tempfile


class DataSource(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.local_db_path = Path("data/partner.db")

    def run_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path: Path = self.fetch_data(Path(temp_dir))
            df_in: pd.DataFrame = self.load_data(data_path)
        df_fixed_col: pd.DataFrame = self.remove_blank_from_column_names(df_in)
        df_cleaned = self.clean_data(df_fixed_col)
        df_validated = self.validate_data(df_cleaned)
        self.upload_data(df_validated)

    @staticmethod
    @abstractmethod
    def fetch_data(temp_dir: Path) -> Path:
        pass

    @staticmethod
    def load_data(file_path: Path) -> pd.DataFrame:
        file_type = get_file_type(file_path)
        match file_type:
            case "csv":
                df_in = pd.read_csv(file_path, dtype=str)
            case "xlsx":
                df_in = pd.read_excel(file_path, dtype=str)
            case "pq":
                df_in = pd.read_parquet(file_path)
            case _:
                raise NotImplementedError(f"Not supported file type: {file_type}")
        return df_in

    @staticmethod
    def remove_blank_from_column_names(df_in: pd.DataFrame) -> pd.DataFrame:
        names_dict = {c: c.replace(" ", "_") for c in df_in.columns}
        return df_in.rename(columns=names_dict)

    @staticmethod
    @abstractmethod
    def clean_data(df_in: pd.DataFrame) -> pd.DataFrame:
        pass

    def validate_data(self, df_in: pd.DataFrame) -> pd.DataFrame:
        validated_df = self.schema.validate(df_in)
        # validated_df filter the columns BUT cast data
        # We want the filtered columns but the str type
        df_out = df_in[validated_df.columns].copy()
        assert isinstance(df_out, pd.DataFrame)
        return df_out

    @property
    @abstractmethod
    def schema(self) -> pa.DataFrameSchema:
        pass

    @staticmethod
    @abstractmethod
    def upload_data(df_in: pd.DataFrame):
        pass


def is_path_valid(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    if not file_path.is_file():
        raise ValueError(f"'{file_path}' is not a regular file.")


def get_file_type(file_path: Path) -> str:
    is_path_valid(file_path)
    file_type = file_path.suffix.lower()
    if file_type == "":
        raise ValueError(f"{file_path} has no file extension.")
    return file_type.replace(".", "")
