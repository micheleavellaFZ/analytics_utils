"""
PostgreSQL database connection utility for data operations.

This module provides a DatabaseConnection class that simplifies interaction with PostgreSQL databases.
It handles connection management, query execution, and data transfer between pandas DataFrames and
database tables, with automatic handling of transactions and proper data type conversion.

Key Features:
- Connection management with context manager support for automatic transaction handling
- Configuration via JSON files for database connection parameters
- Query execution from strings or files
- Conversion of query results to pandas DataFrames
- Bulk uploading of pandas DataFrames to database tables
- Automatic handling of NULL values and integer data types
- Support for conflict resolution during data uploads
"""

import os
import json
import psycopg2
from pathlib import Path
import pandas as pd
from psycopg2.extras import execute_batch
from typing import Iterator
from datetime import datetime
import uuid


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
            return pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])  # type: ignore

    def get_df_from_query_chunk(
        self, query: str | Path, chunk_size: int = 50_000
    ) -> Iterator[pd.DataFrame]:
        if self.connection is None:
            raise ValueError("No database connection")
        query_to_run = self.__path_to_string(query)
        cursor_name = f"chunked_cursor_{uuid.uuid4().hex[:8]}"
        with self.connection.cursor(cursor_name) as cur:
            cur.execute(query_to_run)
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                yield pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])  # type: ignore

    def upload_df_into_db(
        self, table_name: str, df: pd.DataFrame, conflict: str | None = None
    ):
        if self.connection is None:
            raise ValueError("No database connection")
        with self.connection.cursor() as cur:
            df_up = df.copy()
            self.__clean_df_nulls(df_up)
            self.__fix_int_cols(df_up)
            columns = df_up.columns.tolist()
            rows = df_up.values.tolist()
            column_str = ", ".join([f'"{col}"' for col in columns])
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
    def __clean_df_nulls(df: pd.DataFrame) -> None:
        for c in df.columns:
            mask = df[c].isna()
            df[c] = df[c].astype(object)
            df.loc[mask, c] = None

    @staticmethod
    def __fix_int_cols(df: pd.DataFrame):
        for c in df.columns:
            col = df[c].copy()
            if pd.api.types.is_float_dtype(col.dtype):
                col_notna = col.dropna()
                if (col_notna % 1 == 0).all():
                    df[c] = df[c].astype("Int64")


if __name__ == "__main__":
    with DatabaseConnection(
        "/Users/michele.avellafiscozen.it/analytics_data_warehouse/config/postgres_db.json"
    ) as conn:
        print("Start", datetime.now())
        for df in conn.get_df_from_query_chunk("select * from fiscozen_task"):
            print(len(df), datetime.now())
