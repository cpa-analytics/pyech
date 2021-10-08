<img src="logo.png" width=400 style="margin-bottom:60px;display:block;margin:0 auto">

# Overview
A simple package that streamlines the download-read-wrangling process needed to analyze the Encuesta Continua de Hogares survey carried out by the Instituto Nacional de Estadística (Uruguay).

PyECH does not attempt to estimate any indicators in particular, or facilitate any kind of modelling, or concatenate surveys from multiple years. Instead, it aims at providing a hassle-free experience with as simple a syntax as possible.

Here's what PyECH can do:
* Download survey compressed files.
* Unrar, rename and move the SAV (SPSS) file to a specified path.
* Read surveys from SAV files, keeping variable and value labels.
* Download and process variable dictionaries.
* Search through variable dictionaries.
* Summarize variables.
* Calculate variable n-tiles.
* Convert variables to real terms or USD.

Surprisingly, this covers a lot of what people tend to do with the ECH survey without having to deal with software licensing.

# Installation
```bash
pip install pyech
```

## Dependencies

In order to unpack downloaded survey files you will need to have unrar in your system. This should be covered if you have WinRAR or 7zip installed, or can do `sudo apt install unrar` in Ubuntu for example.

# Usage

[Full documentation, including this readme](https://pyech.readthedocs.io).

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