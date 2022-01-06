import pandas as pd
from pandas.tseries.offsets import MonthEnd

from pyech.utils import NXR_URL, CPI_URL


def get_cpi() -> pd.DataFrame:
    """Download and process CPI data.

    Returns
    -------
    pd.DataFrame
    """
    cpi = pd.read_excel(
        CPI_URL, skiprows=10, usecols="A:B", index_col=0,
    ).dropna()

    cpi.index = pd.date_range(start="1937-07-31", periods=len(cpi), freq="M")
    cpi = cpi.rename_axis(None, axis=1)
    cpi.columns = ["Índice de Precios al Consumidor"]
    return cpi


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
