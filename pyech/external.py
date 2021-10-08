import pandas as pd
from pandas.tseries.offsets import MonthEnd

from pyech.utils import NXR_URL, CPI_URL


def get_cpi() -> pd.DataFrame:
    """Download and process CPI data by division.

    Returns
    -------
    pd.DataFrame
    """
    file = pd.ExcelFile(CPI_URL)
    sheet_1 = pd.read_excel(
        file, sheet_name=file.sheet_names[0], skiprows=10, nrows=13
    ).T.reset_index()
    sheet_1 = sheet_1.dropna(how="all", axis=1).iloc[3:, :].dropna(how="any")
    sheet_1.index = pd.date_range(start="1997-03-31", periods=len(sheet_1), freq="M")
    sheet_1 = sheet_1.rename_axis(None, axis=1)
    sheet_2 = pd.read_excel(
        file, sheet_name=file.sheet_names[1], skiprows=9, nrows=14
    ).T.reset_index()
    sheet_2 = sheet_2.dropna(how="all", axis=1).iloc[4:, :].dropna(how="any")
    sheet_2.index = pd.date_range(start="2016-01-31", periods=len(sheet_2), freq="M")
    sheet_2 = sheet_2.rename_axis(None, axis=1)
    sheet_2.columns = [
        "general",
        "food",
        "alcohol",
        "clothing",
        "dwelling",
        "furniture",
        "health",
        "transportation",
        "communications",
        "entertainment",
        "education",
        "accomodation",
        "other",
    ]
    sheet_1.columns = sheet_2.columns
    return pd.concat([sheet_1, sheet_2], axis=0)


def get_nxr() -> pd.DataFrame:
    """Download and process USDUYU nominal exchange rate.

    Returns
    -------
    pd.DataFrame
    """
    raw = pd.read_excel(NXR_URL, skiprows=6, index_col=0).dropna(how="all")
    output = raw.iloc[:, [4]]
    output.columns = ["Promedio, venta"]
    output.index = pd.to_datetime(output.index) + MonthEnd(0)
    return output
