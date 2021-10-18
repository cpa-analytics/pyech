<img src="https://github.com/CPA-Analytics/pyech/raw/master/logo.png" width=400 style="margin-bottom:60px;display:block;margin:0 auto">

![Build status](https://github.com/CPA-Analytics/pyech/actions/workflows/main.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyech/badge/?version=latest)](https://pyech.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pyech.svg)](https://badge.fury.io/py/pyech)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

# Overview
A simple package that streamlines the download-read-wrangling process needed to analyze the [Encuesta Continua de Hogares](https://www.ine.gub.uy/encuesta-continua-de-hogares1) survey carried out by the Instituto Nacional de Estadística (Uruguay).

Here's what PyECH can do:
* Download survey compressed files.
* Unrar, rename and move the SAV (SPSS) file to a specified path.
* Read surveys from SAV files, keeping variable and value labels.
* Download and process variable dictionaries.
* Search through variable dictionaries.
* Summarize variables.
* Calculate variable n-tiles.
* Convert variables to real terms or USD.

PyECH does not attempt to estimate any indicators in particular, or facilitate any kind of modelling, or concatenate surveys from multiple years. Instead, it aims at providing a hassle-free experience with as simple a syntax as possible.

Surprisingly, PyECH covers a lot of what people tend to do with the ECH survey without having to deal with software licensing.

For R users, check out [ech](https://github.com/calcita/ech).

# Installation
```bash
pip install pyech
```

## Dependencies

In order to unpack downloaded survey files you will need to have unrar in your system. This should be covered if you have WinRAR or 7zip installed. Otherwise `sudo apt-get install unrar` or what's appropiate for your system.

# Usage

* [Full documentation, including this readme](https://pyech.readthedocs.io).
* Run the examples notebook in your browser | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CPA-Analytics/pyech/blob/master/examples/example.ipynb)

Loading a survey is as simple as using `ECH.load`, which will download it if it cannot be found at `dirpath` (by default the current working directory).
```python
from pyech import ECH

survey = ECH()
survey.load(year=2019, weights="pesoano")
```
`ECH.load` also downloads the corresponding variable dictionary, which can be easily searched.
```python
survey.search_dictionary("ingreso", ignore_case=True, regex=True)
 ```
 This will return a pandas DataFrame where every row matches the search term in any of its columns.

 Calculating aggregations is as simple as using `ECH.summarize`.
 ```python
 survey.summarize("ht11", by="dpto", aggfunc="mean", household_level=True)
 ```
 Which returns a pandas DataFrame with the mean of "ht11" grouped by `ECH.grouping` and `by` (both are optional). Cases are weighted by the column defined in `ECH.load`.