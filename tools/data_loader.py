from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


@dataclass
class DataLoadResult:
    dataframe: pd.DataFrame
    shape: tuple[int, int]
    columns: list[str]
    head: pd.DataFrame
    source_name: str


def load_dataset(file_source: str | Path | BinaryIO, file_name: str | None = None) -> DataLoadResult:
    extension = _resolve_extension(file_source, file_name)

    if hasattr(file_source, "seek"):
        file_source.seek(0)

    if extension == ".csv":
        dataframe = pd.read_csv(file_source)
    elif extension == ".xlsx":
        dataframe = pd.read_excel(file_source, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file extension: {extension}. Please upload CSV or XLSX.")

    if dataframe.empty:
        raise ValueError("The uploaded dataset is empty.")

    return DataLoadResult(
        dataframe=dataframe,
        shape=dataframe.shape,
        columns=list(dataframe.columns),
        head=dataframe.head(),
        source_name=file_name or str(file_source),
    )


def _resolve_extension(file_source: str | Path | BinaryIO, file_name: str | None) -> str:
    if file_name:
        return Path(file_name).suffix.lower()
    if isinstance(file_source, (str, Path)):
        return Path(file_source).suffix.lower()
    name = getattr(file_source, "name", "")
    return Path(name).suffix.lower()
