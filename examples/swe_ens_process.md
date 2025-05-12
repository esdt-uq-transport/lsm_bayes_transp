## Snow Ensemble Preprocessing 

This example describes several preprocessing steps for implementing Bayesian transport maps for land surface model ensembles. A quantile transformation is applied to the ensemble before fitting the BTM. Supporting routines for the quantile and inverse tranformations are found in `lib/quantile_supp.py`. Scripts for each processing step are found in the `script` directory. BTM fitting also requires the [BatRam](https://github.com/katzfuss-group/batram) library. The procedure below highlights the contemporary climate experiment from the large ensemble, and the steps can be repeated for other experiments, such as the end-of-century RCP 8.5 experiment. 

* Setup/activate Python environment for using BatRam
    - [Environment setup notes](../batram_environment.md)
* Download relevant large ensemble datasets from the [NCAR Research Data Archive](https://rda.ucar.edu/datasets/d651027/dataaccess/)
    - Snow water equivalent (SWE) output can be found in the CESM land daily files collection. The SWE variable name is H2OSNO
    - For the contemporary climate experiment, files containing `*B20TRC5CNBDRD*` are used. [File listing](../config/LENS_H2OSNO_FileList_20TRC.txt)
    - For the RCP 8.5 scenario, files containing `*BRCP85C5CNBDRD*` are used. [File listing](../config/LENS_H2OSNO_FileList_RCP85.txt)
    - Files with `OIC` in the file name do not need to be included in the ensemble
* Compute quantiles and implement location masking: `lens_h2osno_quantile.py`  
Script produces output dataset with location-specific quantiles, along with summary maps  
Needed input data:
    - Downloaded ensemble output
    - Probability grid values `config/Probs.csv`
    - Preliminary land mask and location info: `config/LENS_NAmer_Locs.csv`
* Optional: Plot selected city empirical CDFs: `lens_city_ecdf.py`  
Needed input data:
    - List of selected cities and locations: `config/LENS_Cities_LocIdx.csv`
* Optional: K-means cluster analysis for SWE ensemble: `lens_h2osno_kmeans.py`  
Needed input data:
    - Downloaded ensemble output 

