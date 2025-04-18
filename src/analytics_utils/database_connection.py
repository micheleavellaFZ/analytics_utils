import os
import json
import psycopg2
from pathlib import Path
import pandas as pd
from psycopg2.extras import execute_batch


class DatabaseConnection:
    def __init__(self, config_file: str | Path) -> None:
        with open(config_file) as f:
            self.config = json.load(f)
        self.connection: None | psycopg2.extensions.connection = None

    def connect(self) -> None:
        try:
            self.connection = psycopg2.connect(**self.config)
        except psycopg2.Error:
            self.connection = None
            raise

    def commit(self) -> None:
        if self.connection is None:
            raise ValueError("No database connection")
        self.connection.commit()

    def rollback(self) -> None:
        if self.connection is None:
            raise ValueError("No database connection")
        self.connection.rollback()

    def disconnect(self) -> None:
        if self.connection is None:
            raise ValueError("No database connection")
        self.connection.close()
        self.connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            # if something bad happend inside the with
            if exc_type is not None:
                self.connection.rollback()
            else:
                self.connection.commit()
            self.connection.close()
            self.connection = None
        return False

    def run_query(self, query: str | Path) -> None:
        if self.connection is None:
            raise ValueError("No database connection")
        query_to_run = self.__path_to_string(query)
        with self.connection.cursor() as cur:
            cur.execute(query_to_run)

    def get_df_from_query(self, query: str | Path) -> pd.DataFrame:
        if self.connection is None:
            raise ValueError("No database connection")
        query_to_run = self.__path_to_string(query)
        with self.connection.cursor() as cur:
            cur.execute(query_to_run)
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])

    def upload_df_into_db(
        self, table_name: str, df: pd.DataFrame, conflict: str | None = None
    ):
        if self.connection is None:
            raise ValueError("No database connection")
        with self.connection.cursor() as cur:
            df_up = self.__clean_df_nulls(df)
            self.__fix_int_cols(df_up)
            columns = df_up.columns.tolist()
            rows = df_up.values.tolist()
            column_str = ", ".join(columns)
            placeholder_str = ", ".join(["%s"] * len(columns))
            query = (
                f"INSERT INTO {table_name} ({column_str}) VALUES ({placeholder_str})"
            )
            if conflict is not None:
                query += conflict
            chunk_size = 10_000
            chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
            for chunk in chunks:
                data = [tuple(row) for row in chunk]
                execute_batch(cur, query, data)

    @staticmethod
    def __path_to_string(path: str | Path) -> str:
        if os.path.isfile(path):
            with open(path) as f:
                return f.read()
        else:
            return str(path)

    @staticmethod
    def __clean_df_nulls(df: pd.DataFrame) -> pd.DataFrame:
        return df.where(pd.notnull(df), None)

    @staticmethod
    def __fix_int_cols(df: pd.DataFrame):
        for c in df.columns:
            col = df[c].copy()
            if pd.api.types.is_float_dtype(col.dtype):
                col_notna = col.dropna()
                if (col_notna % 1 == 0).all():
                    df[c] = df[c].astype("Int64")
