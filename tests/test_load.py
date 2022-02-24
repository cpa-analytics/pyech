from random import randint
from pathlib import Path

import pandas as pd

from pyech import ECH


def test_load():
    year = randint(2006, 2020)
    survey = ECH("tests")
    survey.load(year, weights="pesoano", splitter="e26")
    Path(f"tests/{year}.sav").unlink()
    assert isinstance(survey.data, pd.DataFrame)


def test_repo_load():
    year = randint(2006, 2020)
    survey = ECH("tests")
    survey.load(year, weights="pesoano", from_repo=True)
    Path(f"tests/{year}.h5").unlink()
    Path(f"tests/{year}.json").unlink()
    assert isinstance(survey.data, pd.DataFrame)


def test_dictionary():
    survey = ECH("tests")
    survey.load(2019, weights="pesoano")
    variables = "nper|numero|anio|dpto|nomdpto"
    search = survey.search_dictionary(variables, regex=True, ignore_case=True)
    assert all(x in search["Variable"].unique() for x in variables.split("|"))


def test_apply_weights():
    survey = ECH("tests")
    survey.load(2019, weights="pesoano")
    output = survey.apply_weights(["nper", "d8_4"])
    assert output.shape[0] == survey.data["pesoano"].sum()
