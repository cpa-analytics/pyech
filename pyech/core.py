from __future__ import annotations
import tempfile
import re
import shutil
from os import getcwd
from pathlib import Path
from typing import Callable, Optional, Sequence, Union, List
from urllib.request import urlretrieve
from warnings import warn
from datetime import datetime, date

import numpy as np
import pandas as pd
import patoolib
from pandas_weighting import weight
from pyreadstat import metadata_container, read_sav
from pyreadstat._readstat_parser import PyreadstatError

from pyech.utils import DICTIONARY_URLS, SURVEY_URLS
from pyech.external import get_cpi, get_nxr


class ECH(object):
    """Download, read and process the 2006-2020 Encuesta Continua de Hogares survey carried out by
    Uruguay's Instituto Nacional de Estadística.

    Parameters
    ----------
    dirpath :
        Path where to download new surveys or read existing ones, by default ".".
    categorical_threshold :
        Number of unique values below which the variable is considered categorical, by default 50.

    Attributes
    ----------
    data : pd.DataFrame
        Survey data, by default pd.DataFrame().
    metadata : metadata_container
        Survey metadata, by default None.
    weights : Optional[str]
        Column in :attr:`data` used to weight cases. Generally "pesoano" for annual weighting, by
        default None
    grouping : Union[str, List[str]]
        Variable(s) to use for grouping in methods (:mod:`~pyech.core.ECH.summarize`,
        :mod:`~pyech.core.ECH.assign_ptile`), by default [].
    dictionary : pd.DataFrame
        Variable dictionary, by default pd.DataFrame().
    cpi : pd.DataFrame
        Monthly CPI data, by default pd.DataFrame().
    nxr : pd.DataFrame
        Monthly nominal exchange rate data, by default pd.DataFrame().
    """

    def __init__(
        self,
        dirpath: Union[Path, str] = getcwd(),
        categorical_threshold: int = 50,
    ):
        self.dirpath = dirpath
        self.categorical_threshold = categorical_threshold

    @classmethod
    def from_sav(cls, data: pd.DataFrame, metadata: metadata_container) -> ECH:
        """Build :class:`~pyech.core.ECH` from :attr:`data` and
        :attr:`metadata` as created by `pyreadstat.read_sav()`.

        Parameters
        ----------
        data :
            Survey data.
        metadata :
            Survey metadata.

        Returns
        -------
        :class:`~pyech.core.ECH`
        """
        svy = ECH()
        svy.data = data
        svy.metadata = metadata
        return svy

    @property
    def grouping(self):
        return self._grouping

    @grouping.setter
    def grouping(self, group):
        if group is None:
            group = []
        elif not isinstance(group, Sequence) or isinstance(group, str):
            group = [group]
        self._grouping = group

    def load(
        self,
        year: int,
        weights: Optional[str] = None,
        grouping: Union[str, List[str]] = [],
        missing: Optional[str] = r"\s+\.",
        missing_regex: bool = True,
        lower: bool = True,
        dictionary: bool = True,
    ) -> None:
        """Load a ECH survey and dictionary from a specified year.

        First attempt to read a survey by looking for "`year`.sav" in :attr:`dirpath`. If it cannot
        be found, download the .rar file, extract it to a temporary directory, move the renamed
        .sav file to :attr:`dirpath` and then read. Optionally replace missing values with
        `numpy.nan`, lower all variable names and download the corresponding variable dictonary.

        Parameters
        ----------
        year :
            Survey year
        weights :
            Variable used for weighting cases, by default None.
        grouping :
            Variable(s) to use for grouping in methods (:mod:`~pyech.core.ECH.summarize`,
            :mod:`~pyech.core.ECH.assign_ptile`), by default []
        missing :
            Missing values to replace with `numpy.nan`. Can be a regex with `missing_regex=True`,
            by default r"\s+\.".
        missing_regex :
            Whether to parse `missing` as regex, by default True.
        lower :
            Whether to turn variable names to lower case. This helps with analyzing surveys for
            several years, by default True.
        dictionary :
            Whether to download the corresponding variable dictionary, by default True.
        """
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
        self.weights = weights
        if not weights:
            warn(
                "No column selected for `weights`. Be sure to set the property before using other methods."
            )
        elif weights and weights not in self.data.columns:
            warn("Selected `weights` not available in dataset.")
        self.grouping = grouping
        return

    def _read(self, path: Union[Path, str]):
        self.data, self.metadata = read_sav(path)
        return

    def _lower_variable_names(self) -> None:
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
    def download(dirpath: Union[Path, str], year: int) -> None:
        """Download a ECH survey, unpack the .rar, extract the .sav, rename as "`year`.sav" and
        place in :attr:`dirpath`.

        Parameters
        ----------
        dirpath :
            Download location.
        year :
            Survey year.
        """
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
        dl_dir = Path(dirpath, "dl")
        dl_dir.mkdir(exist_ok=True)
        patoolib.extract_archive(temp_file, outdir=dl_dir, verbosity=-1)
        survey = [
            list(Path(dirpath).glob(f"**/*{x}*.sav"))
            for x in ["h*p", "Fusionado", "FUSIONADO", "H*P"]
        ]
        survey = [x[0] for x in survey if x][0]
        survey.rename(Path(dirpath, f"{year}.sav"))
        shutil.rmtree(dl_dir)
        return

    def get_dictionary(self, year: int) -> None:
        """Download and process variable dictionary for a specified year.

        Parameters
        ----------
        year :
            Survey year.
        """
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
    ) -> pd.DataFrame:
        """Return rows in :attr:`dictionary` with matching terms.

        Parameters
        ----------
        term :
            Search term.
        ignore_case :
            Whether to search for upper and lower case. Requires `regex=True`, by default False.
        regex :
            Whether to parse `term` as regex, by default False.

        Returns
        -------
        pd.DataFrame
            DataFrame containing matching rows.
        """
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
        variable: str,
        by: Optional[Union[str, List[str]]] = None,
        is_categorical: Optional[bool] = None,
        aggfunc: Union[str, Callable] = "mean",
        apply_labels: bool = True,
        household_level: bool = False,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """Summarize a variable in :attr:`data`.

        Parameters
        ----------
        variable :
            Variable to summarize.
        by :
            Summarize by these groups, as well as those in :attr:`grouping`, by default None.
        is_categorical :
            Whether `value` should be treated as categorical. If None, compare with
            :attr:`categorical_threshold`, by default None.
        aggfunc :
            Aggregating function. Possible values are "mean", "sum", "count", or any function that
            works with pd.DataFrame.apply. If `values` is categorical will force `aggfunc="count"`,
            by default "mean".
        apply_labels :
            Whether to use value labels from :attr:`metadata`, by default True.
        household_level :
            If True, summarize at the household level (i.e. consider only :attr:`data` ["nper"] == 1),
            by default False.
        dropna :
            Whether to drop groups with no observations, by default False.

        Returns
        -------
        pd.DataFrame
            Summarized variable.

        Raises
        ------
        AttributeError
            If :attr:`weights` is not defined.
        """
        if not self.weights:
            raise AttributeError("Summarization requires that `weights` is defined.")
        if household_level:
            data = self.data.loc[self.data["nper"] == 1]
        else:
            data = self.data.copy()
        if by is None:
            by_array = []
        elif not isinstance(by, Sequence) or isinstance(by, str):
            by_array = [by]
        else:
            by_array = by
        all_groups = self.grouping + by_array
        groups = True if len(all_groups) > 0 else False
        if is_categorical is None:
            categorical = self._guess_categorical(variable)
        else:
            categorical = is_categorical
        if categorical:
            if aggfunc != "count":
                warn(f"'{variable}' is categorical. Summarizing by count.")
            output = (
                data.groupby(all_groups + [variable], dropna=dropna)[self.weights]
                .sum()
                .reset_index()
            )
            output.rename({self.weights: "Recuento"}, axis=1, inplace=True)
        else:
            if aggfunc == "mean":
                if groups:
                    output = data.groupby(all_groups, dropna=dropna).apply(
                        lambda x: np.average(x[variable], weights=x[self.weights])
                    )
                else:
                    output = [np.average(data[variable], weights=data[self.weights])]
                output = pd.DataFrame(output)
                output.columns = [variable]
                if groups: output.reset_index(inplace=True)
            elif aggfunc in ["sum", sum, "count"]:
                data["wtd_val"] = data[variable] * data[self.weights]
                if groups:
                    output = (
                        data.groupby(all_groups, dropna=dropna)[
                            ["wtd_val", self.weights]
                        ]
                        .sum()
                        .reset_index()
                    )
                else:
                    output = data[["wtd_val", self.weights]].sum().to_frame().T
                if aggfunc == "sum" or aggfunc == sum:
                    output["results"] = output["wtd_val"]
                    output.drop(["wtd_val", self.weights], axis=1, inplace=True)
                    output.rename({"results": variable}, axis=1, inplace=True)
                elif aggfunc == "count":
                    output["results"] = output[self.weights]
                    output.drop(["wtd_val", self.weights], axis=1, inplace=True)
                    output.rename({"results": variable}, axis=1, inplace=True)
            else:
                pd.DataFrame.weight = weight
                pd.Series.weight = weight
                if groups:
                    weighted = data[[variable] + all_groups].weight(data[self.weights])
                    output = (
                        weighted.groupby(self.grouping + by_array, dropna=False)
                        .agg(aggfunc)
                        .reset_index()
                    )
                else:
                    weighted = data[[variable]].weight(data[self.weights])
                    output = weighted.apply(aggfunc).to_frame().T
        if apply_labels:
            replace_names = {
                group: self.metadata.variable_value_labels[group]
                for group in all_groups
                if group in self.metadata.variable_value_labels
            }
            if categorical and variable in self.metadata.variable_value_labels:
                replace_names.update(
                    {variable: self.metadata.variable_value_labels[variable]}
                )
            output.replace(replace_names, inplace=True)

        return output

    def _guess_categorical(self, variable):
        if self.data[variable].dtype.name in ["object", "category"]:
            return True
        if (
            self.data[variable].dtype.kind in "iuf"
            and self.data[variable].nunique() <= self.categorical_threshold
        ):
            return True
        else:
            return False

    def assign_ptile(
        self,
        variable: str,
        n: int,
        labels: Union[bool, Sequence[str]] = False,
        by: Optional[Union[str, List[str]]] = None,
        result_weighted: bool = False,
        name: Optional[str] = None,
        household_level: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Calculate n-tiles for a variable. By default add as new column to :attr:`data`.

        Parameters
        ----------
        variable :
            Reference variable.
        n :
            Number of bins to calculate.
        labels :
            Passed to `pandas.qcut`. If False, use `int` labels for the resulting bins. If True,
            name bins by their edges. Otherwise pass a sequence of length equal to `n`, by default
            False.
        by :
            Calculate bins for each of the groups, as well as those in :attr:`grouping`, by default
            None.
        result_weighted :
            If True, return a pd.DataFrame with the weighted result. Else, add as a column to
            :attr:`data`, by default False
        name :
            Name for the new column. If None, set as "`variable`_`n`", by default None:
        household_level :
            If True, calculate at the household level (i.e. consider only :attr:`data` ["nper"] == 1),
            by default False.

        Returns
        -------
        Optional[pd.DataFrame]

        Raises
        ------
        AttributeError
            If :attr:`weights` is not defined.
        """
        if not self.weights:
            raise AttributeError(
                "Assigning percentiles requires that `weights` is defined."
            )
        pd.DataFrame.weight = weight
        pd.Series.weight = weight
        if household_level:
            data = self.data.loc[self.data["nper"] == 1]
        else:
            data = self.data.copy()
        if by is None:
            by_array = []
        elif not isinstance(by, Sequence) or isinstance(by, str):
            by_array = [by]
        else:
            by_array = by
        all_groups = self.grouping + by_array
        groups = True if len(all_groups) > 0 else False
        valid = [v for v in [variable] + all_groups if v]
        weighted = data[valid].weight(data[self.weights])
        if groups:
            output = weighted.groupby(all_groups)[variable].transform(
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
        variables: Union[str, List[str]],
        division: str = "general",
        start: Optional[Union[str, datetime, date]] = None,
        end: Optional[Union[str, datetime, date]] = None,
    ) -> None:
        """Convert selected monetary variables to real terms.

        Parameters
        ----------
        variables :
            Column(s) in :attr:`data`. Can be a string or a sequence of strings for multiple columns.
        division :
            CPI index to use for deflating. Can be one of "general", "food", "alcohol", "clothing",
            "dwelling", "furniture", "health", "transportation", "communications", "entertainment",
            "education", "accomodation" and "other", by default "general".
        start, end :
            Set prices to either of these dates or the mean between them, by default None.
        """
        try:
            self.cpi
        except:
            self.cpi = get_cpi()
        if start and not end:
            ref = self.cpi.iloc[self.cpi.index.get_loc(start, method="nearest")][
                division
            ]
        elif not start and end:
            ref = self.cpi.iloc[self.cpi.index.get_loc(end, method="nearest")][division]
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
        output["mes"] = np.where(output["mes"] - 1 == 0, 12, output["mes"] - 1)
        output = output.merge(survey_cpi, on="mes", how="left")
        output = output[variables].div(output[division], axis=0) * ref
        self.data[[f"{x}_real" for x in variables]] = output.astype(
            "float"
        )
        return

    def convert_usd(self, variables: Union[str, List[str]]) -> None:
        """Convert selected monetary variables to USD.

        Parameters
        ----------
        variables :
            Column(s) in :attr:`data`. Can be a string or a sequence of strings for multiple columns.
        """
        try:
            self.nxr
        except AttributeError:
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
        output["mes"] = np.where(output["mes"] - 1 == 0, 12, output["mes"] - 1)
        output = output.merge(survey_nxr, on="mes", how="left")
        output = output[variables].div(output["Promedio, venta"], axis=0)
        self.data[[f"{x}_usd" for x in variables]] = output.astype(
            "float"
        )
        return

    def apply_weights(self, variables: Union[str, List[str]]) -> pd.DataFrame:
        """Repeat rows as many times as defined in :attr:`weights`.

        Parameters
        ----------
        variables :
            Columns for which weights should be applied. In general it is a good idea to avoid
            applying weights to all columns since this can result in a large DataFrame.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        AttributeError
            If :attr:`weights` is not defined.
        """
        pd.DataFrame.weight = weight
        if not isinstance(variables, Sequence) or isinstance(variables, str):
            variables = [variables]
        return self.data[variables].weight(self.data[self.weights])
