### Python Environment for Bayesian Transport Maps

The examples in this repository depend on a Python environment supporting the [BatRam](https://github.com/katzfuss-group/batram) library and its dependencies, which include `gpytorch`. A virtual environment or Anaconda environment can be used for this implementation.

**Virtual Environment (Successful on Mac M1)**

* [Basic background notes](https://mnzel.medium.com/how-to-activate-python-venv-on-a-mac-a8fa1c3cb511) on Python virtural environments on Mac systems
* `brew install python-tk@3.10`
* `python3.10 -m pip install --user --upgrade pip`
* `python3.10 -m pip install --user virtualenv`  
`/Users/user/Library/Python/3.10/bin`
* Set up virtual environment: `python3.10 -m venv venv`
* Activate: `source venv/bin/activate`
* Clone batram: `git clone https://github.com/katzfuss-group/batram`
* PIP install: `pip install -e .`

**Anaconda Environment (Successful on Ubuntu Linux)**

* Python 3.10 or higher: `conda create -n py310 python=3.10 anaconda`
* Clone batram: `git clone https://github.com/katzfuss-group/batram`
* PIP install: `pip install -e .`

**Other packages for land surface model analysis and visualization**

* netCDF4
* pandas
* cartopy
