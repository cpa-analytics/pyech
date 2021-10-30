import numpy as np
import pytest

from pyech import ECH


survey = ECH("tests")
survey.load(2019, weights="pesoano")
cat_arrays = [
    [3209725.0, 308206.0],
    [0.08761002],
    [78600.0, 81014.0, 1626098.0, 1732219.0],
]
noncat_arrays = [
    [82965.39028371],
    [3926.0, 3902.0, 4321.5, 3526.0],
    [8.05198985e10, 1.93731241e09],
]


@pytest.mark.parametrize(
    "variable,is_categorical,aggfunc,by,column,result",
    [
        ("pobre06", None, "count", None, "Recuento", cat_arrays[0]),
        ("pobre06", False, "mean", None, "pobre06", cat_arrays[1]),
        ("e26", None, "count", "d8_4", "Recuento", cat_arrays[2]),
    ],
)
def test_cat(variable, is_categorical, aggfunc, by, column, result):
    output = survey.summarize(
        variable,
        aggfunc=aggfunc,
        by=by,
        is_categorical=is_categorical,
        apply_labels=False,
    )
    assert np.allclose(output[column], result, atol=0, rtol=0.01, equal_nan=True)


@pytest.mark.parametrize(
    "variable,aggfunc,by,result,household_level",
    [
        ("ht11", "mean", None, noncat_arrays[0], False),
        ("yhog", np.median, ["e26", "d8_4"], noncat_arrays[1], True),
        ("ysvl", "sum", "pobre06", noncat_arrays[2], True),
    ],
)
def test_noncat(variable, aggfunc, by, result, household_level):
    output = survey.summarize(
        variable,
        aggfunc=aggfunc,
        by=by,
        household_level=household_level,
        apply_labels=False,
    )
    assert np.allclose(output[variable], result, atol=0, rtol=0.01, equal_nan=True)
