import tempfile
import re
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
from urllib.request import urlretrieve
from warnings import warn

import numpy as np
import pandas as pd
import patoolib
from pandas_weighting import weight
from pyreadstat import metadata_container, read_sav
from pyreadstat._readstat_parser import PyreadstatError

from pyech.utils import (
    DICTIONARY_URLS,
    PATH,
    OPTIONAL_STR_LIST,
    STR_LIST,
    DATE,
    SURVEY_URLS,
)
from pyech.external import get_cpi, get_nxr


class ECH(object):
    def __init__(
        self,
        dirpath: PATH = ".",
        categorical_threshold: int = 50,
    ):
        self.dirpath = dirpath
        self.categorical_threshold = categorical_threshold
        self.data: pd.DataFrame = pd.DataFrame()
        self.metadata: Optional[metadata_container] = None
        self.weights: Optional[str] = None
        self.dictionary: pd.DataFrame = pd.DataFrame()
        self.cpi = pd.DataFrame()
        self.nxr = pd.DataFrame()

    def load(
        self,
        year: int,
        weights: Optional[str] = None,
        missing: Optional[str] = r"\s+\.",
        missing_regex: bool = True,
        lower: bool = True,
        dictionary: bool = True,
    ):
        try:
            self._read(Path(self.dirpath, f"{year}.sav"))
        except PyreadstatError:
            warn("Could not read data. It could be missing. Attempting download...")
            self.download(dirpath=self.dirpath, year=year)
            self._read(Path(self.dirpath, f"{year}.sav"))
        if missing is not None:
            self.data = self.data.replace(missing, np.nan, regex=missing_regex)
        if lower:
            self._lower_variable_names()
        if dictionary:
            self.get_dictionary(year=year)
        if not weights:
            warn(
                "No column selected for `weights`. Be sure to set the property before using other methods."
            )
        elif weights and weights in self.data.columns:
            self.weights = weights
        else:
            warn("Selected `weights` not available in dataset.")
        return

    def _read(self, path: PATH):
        self.data, self.metadata = read_sav(path)
        return

    def _lower_variable_names(self):
        self.data.columns = self.data.columns.str.lower()
        self.metadata.column_names = [x.lower() for x in self.metadata.column_names]
        for attr in [
            "variable_value_labels",
            "column_names_to_labels",
            "variable_to_label",
            "original_variable_types",
            "readstat_variable_types",
            "variable_alignment",
            "variable_storage_width",
            "variable_display_width",
            "variable_measure",
        ]:
            setattr(
                self.metadata,
                attr,
                {k.lower(): v for k, v in getattr(self.metadata, attr).items()},
            )
        return

    @staticmethod
    def download(dirpath: PATH, year: int):
        full_path = Path(dirpath, f"{year}.sav")
        if full_path.exists():
            warn(f"{year} survey already exists in {dirpath}")
            return
        if not full_path.parent.exists():
            Path(dirpath).mkdir(parents=True, exist_ok=True)
        url = SURVEY_URLS[year]
        temp_file = tempfile.NamedTemporaryFile(suffix=".rar", delete=False).name
        with open(temp_file, "wb") as f:
            urlretrieve(url, f.name)
        with tempfile.TemporaryDirectory() as d:
            patoolib.extract_archive(temp_file, outdir=d, verbosity=-1)
            survey = [list(Path(d).glob(f"**/*{x}*.sav")) for x in ["h*p", "fusionado"]]
            survey = [x[0] for x in survey if x][0]
            survey.rename(Path(dirpath, f"{year}.sav"))
        return

    def get_dictionary(self, year: int):
        url = DICTIONARY_URLS[year]
        excel = pd.ExcelFile(url)
        sheets = []
        for sheet in excel.sheet_names:
            sheet_data = pd.read_excel(excel, sheet_name=sheet, skiprows=7)
            if sheet_data.empty:
                continue
            sheet_data.columns = ["Nombre", "Variable", "Código", "Descripción"]
            sheet_data.dropna(thresh=2, inplace=True)
            sheet_data[["Nombre", "Variable"]] = sheet_data[
                ["Nombre", "Variable"]
            ].fillna(method="ffill")
            sheet_data["Variable"] = sheet_data["Variable"].str.lower()
            sheet_data.insert(0, "Nivel", sheet)
            sheets.append(sheet_data)
        concat = pd.concat(sheets)
        concat.reset_index(drop=True, inplace=True)
        self.dictionary = concat
        return

    def search_dictionary(
        self, term: str, ignore_case: bool = False, regex: bool = False
    ):
        matches = []
        if ignore_case:
            flags = re.IGNORECASE
            if not regex:
                warn("`ignore_case=True` requires `regex=True`.")
        else:
            flags = 0
        for col in self.dictionary.columns:
            match = self.dictionary.loc[
                self.dictionary[col].str.contains(
                    term, flags=flags, na=False, regex=regex
                )
            ]
            matches.append(match)
        matches = pd.concat(matches).sort_index()
        matches = matches.drop_duplicates(keep="first")
        return matches

    def summarize(
        self,
        values: str,
        by: OPTIONAL_STR_LIST = None,
        is_categorical: Optional[bool] = None,
        aggfunc: Union[str, Callable] = "mean",
        apply_labels: bool = True,
        household_level: bool = False,
        dropna: bool = False,
    ):
        if not self.weights:
            raise AttributeError("Summarization requires that `weights` is defined.")
        if household_level:
            data = self.data.loc[self.data["nper"] == 1]
        else:
            data = self.data.copy()
        if not isinstance(by, Sequence) or isinstance(by, str):
            by_array = [by]
        else:
            by_array = by
        if is_categorical is None:
            categorical = self._guess_categorical(values)
        else:
            categorical = is_categorical
        if categorical:
            if aggfunc != "count":
                warn(f"'{values}' is categorical. Summarizing by count.")
            if by is not None:
                output = (
                    data.groupby(by_array + [values], dropna=dropna)[self.weights]
                    .sum()
                    .reset_index()
                )
            else:
                output = (
                    data.groupby([values], dropna=dropna)[self.weights]
                    .sum()
                    .reset_index()
                )
            output.rename({self.weights: "Recuento"}, axis=1, inplace=True)
        else:
            if aggfunc == "mean":
                if by is not None:
                    output = data.groupby(by_array, dropna=dropna).apply(
                        lambda x: np.average(x[values], weights=x[self.weights])
                    )
                else:
                    output = [np.average(data[values], weights=data[self.weights])]
                output = pd.DataFrame(output)
                output.columns = [values]
                output.reset_index(inplace=True)
            elif aggfunc in ["sum", sum, "count"]:
                data["wtd_val"] = data[values] * data[self.weights]
                if by is not None:
                    output = (
                        data.groupby(by_array, dropna=dropna)[["wtd_val", self.weights]]
                        .sum()
                        .reset_index()
                    )
                else:
                    output = data[["wtd_val", self.weights]].sum().to_frame().T
                if aggfunc == "sum" or aggfunc == sum:
                    output["results"] = output["wtd_val"]
                    output.drop(["wtd_val", self.weights], axis=1, inplace=True)
                    output.rename({"results": values}, axis=1, inplace=True)
                elif aggfunc == "count":
                    output["results"] = output[self.weights]
                    output.drop(["wtd_val", self.weights], axis=1, inplace=True)
                    output.rename({"results": values}, axis=1, inplace=True)
            else:
                pd.DataFrame.weight = weight
                pd.Series.weight = weight
                if by is not None:
                    weighted = data[[values] + by_array].weight(data[self.weights])
                    output = (
                        weighted.groupby(by_array, dropna=False)
                        .agg(aggfunc)
                        .reset_index()
                    )
                else:
                    weighted = data[[values]].weight(data[self.weights])
                    output = weighted.apply(aggfunc)
        if apply_labels:
            replace_names = {
                group: self.metadata.variable_value_labels[group]
                for group in by_array
                if group in self.metadata.variable_value_labels
            }
            if categorical and values in self.metadata.variable_value_labels:
                replace_names.update(
                    {values: self.metadata.variable_value_labels[values]}
                )
            output.replace(replace_names, inplace=True)
        return output

    def _guess_categorical(self, variable):
        if self.data[variable].dtype in ["object", "category"]:
            return True
        if (
            self.data[variable].dtype.kind in "iuf"
            and self.data[variable].nunique() <= self.categorical_threshold
        ):
            return True
        else:
            return False

    def percentile(
        self,
        variable: str,
        n: int,
        labels: Union[bool, Sequence[str]] = False,
        by: OPTIONAL_STR_LIST = None,
        result_weighted: bool = False,
        name: Optional[str] = None,
        household_level: bool = False,
    ):
        pd.DataFrame.weight = weight
        pd.Series.weight = weight
        if household_level:
            data = self.data.loc[self.data["nper"] == 1]
        else:
            data = self.data.copy()
        if not isinstance(by, Sequence) or isinstance(by, str):
            by_array = [by]
        else:
            by_array = by
        valid = [v for v in [variable] + by_array if v]
        weighted = data[valid].weight(data[self.weights])
        if by:
            output = weighted.groupby(by)[variable].transform(
                func=pd.qcut, q=n, labels=labels
            )
        else:
            output = pd.qcut(weighted[variable], q=n, labels=labels)
        if result_weighted:
            return output
        else:
            if not name:
                name = f"{variable}_{n}"
            if not household_level:
                self.data[name] = output.loc[~output.index.duplicated(keep="first")]
            else:
                self.data.loc[self.data["nper"] == 1, name] = output.loc[
                    ~output.index.duplicated(keep="first")
                ]
            return

    def convert_real(
        self,
        variables: STR_LIST,
        division: str = "general",
        start: DATE = None,
        end: DATE = None,
    ):
        with pd.option_context("mode.chained_assignment", None):
            if self.cpi.empty:
                self.cpi = get_cpi()
            if start and not end:
                ref = self.cpi.iloc[self.cpi.index.get_loc(start, method="nearest")][
                    division
                ]
            elif not start and end:
                ref = self.cpi.iloc[self.cpi.index.get_loc(end, method="nearest")][
                    division
                ]
            elif start and end:
                ref = self.cpi.loc[start:end].mean()[division]
            else:
                ref = 1

            survey_cpi = self.cpi.loc[
                (
                    (self.cpi.index.year == int(self.data.anio[0]))
                    & (self.cpi.index.month != 12)
                )
                | (
                    (self.cpi.index.year == int(self.data.anio[0]) - 1)
                    & (self.cpi.index.month == 12)
                ),
                [division],
            ]
            survey_cpi.loc[:, "mes"] = survey_cpi.index.month

            if isinstance(variables, str):
                variables = [variables]
            output = self.data.loc[:, ["mes"] + variables]
            output["mes"] = output.loc[:, "mes"] - 1
            output["mes"] = output.loc[:, "mes"].where(output.loc[:, "mes"] == -1, 12)
            output = output.merge(survey_cpi, on="mes")
            output = output.div(output[division], axis=0) * ref
            self.data[[f"{x}_real" for x in variables]] = output.loc[:, variables]
            return

    def convert_usd(self, variables: STR_LIST):
        if self.nxr.empty:
            self.nxr = get_nxr()

        survey_nxr = self.nxr.loc[
            (
                (self.nxr.index.year == int(self.data.anio[0]))
                & (self.nxr.index.month != 12)
            )
            | (
                (self.nxr.index.year == int(self.data.anio[0]) - 1)
                & (self.nxr.index.month == 12)
            ),
            :,
        ]
        survey_nxr.loc[:, "mes"] = survey_nxr.index.month

        if isinstance(variables, str):
            variables = [variables]
        output = self.data.loc[:, ["mes"] + variables]
        output["mes"] = output.loc[:, "mes"] - 1
        output["mes"] = output.loc[:, "mes"].where(output.loc[:, "mes"] == -1, 12)
        output = output.merge(survey_nxr, on="mes")
        output = output.div(output["Promedio, venta"], axis=0)
        self.data[[f"{x}_usd" for x in variables]] = output.loc[:, variables]
        return
