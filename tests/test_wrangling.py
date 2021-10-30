import pytest
import numpy as np

from pyech import ECH


survey = ECH("tests")
survey.load(2019, weights="pesomen")


def test_real():
    survey.convert_real("ht11", start="2019-01-01", end="2019-12-31")
    real = survey.summarize("ht11_real", by="mes", aggfunc="mean")
    nominal = survey.summarize("ht11", by="mes", aggfunc="mean")
    assert real.iloc[0, 1] >= nominal.iloc[0, 1]
    assert real.iloc[-1, 1] <= nominal.iloc[-1, 1]


def test_usd():
    survey.convert_usd("ht11")
    usd = survey.summarize("ht11_usd", by="mes", aggfunc="mean", variable_labels=False)
    peso = survey.summarize("ht11", by="mes", aggfunc="mean", variable_labels=False)
    xr = survey.nxr.loc["2019-01-01":"2019-12-31"].set_index(peso.index)
    assert np.allclose(
        usd["ht11_usd"],
        peso[["ht11"]].div(xr.squeeze(), axis=0).squeeze(),
        rtol=0.05,
        atol=0,
    )
