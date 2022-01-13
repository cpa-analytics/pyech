import os
from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path("pyech/_version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

packages = find_packages(".", exclude=["*.test", "*.test.*"])

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyech",
    version=main_ns["__version__"],
    description="Process INE's ECH surveys in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CPA Ferrere | Data Analytics",
    license="MIT",
    url="https://github.com/cpa-analytics/pyech",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Other Audience",
        "Topic :: Sociology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    keywords=["uruguay", "survey", "ech", "statistics", "ine", "data"],
    install_requires=[
        "pandas>=1.2.0",
        "patool",
        "pyreadstat",
        "pandas-weighting",
        "openpyxl>=2.6.0",
        "xlrd>=1.2.0",
    ],
    include_package_data=True,
    packages=packages,
    python_requires=">=3.6",
)
