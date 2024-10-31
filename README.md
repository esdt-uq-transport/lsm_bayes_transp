# lsm_bayes_transp
**Bayesian transport maps for land surface model output**

This repository implements [Bayesian transport maps (BTM, Katzfuss and Schäfer, 2023)](https://doi.org/10.1080/01621459.2023.2197158) in Python for output from the NCAR large ensemble, [Kay et al., 2015](https://doi.org/10.1175/BAMS-D-13-00255.1)

***

### BTM for Snow

This implementation executes the BTM fit and simulation for snow water equivalent (SWE) from the large ensemble over North America. A quantile transformation is applied to the ensemble before fitting the BTM. Supporting routines for the quantile and inverse tranformations are found in `lib/quantile_supp.py`. Scripts for each processing step are found in the `script` directory. BTM fitting also requires the [BatRam](https://github.com/katzfuss-group/batram) library. The procedure below highlights the contemporary climate experiment from the large ensemble, and the steps can be repeated for other experiments, such as the end-of-century RCP 8.5 experiment. 

* Setup/activate Python environment for using BatRam
    - [Environment setup notes](batram_environment.md)
* Download relevant large ensemble datasets from the [Climate Data Gateway]((https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.cesmLE.html))
    - [Daily SWE output listing](https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.cesmLE.lnd.proc.daily_ave.H2OSNO/file.html)
    - For the contemporary climate experiment, the string `*B20TRC5CNBDRD*` can be entered in the *Filter by File Name* box  
    For the RCP 8.5 scenario, the string `*BRCP85C5CNBDRD*` can be used.
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
* Fit BTM and generate new samples: `lens_h2osno_trsnf_batram.py`  
Needed input data:
    - Downloaded ensemble output
    - Quantiles by location
    - List of selected cities and locations

***

### Dependencies

The BTM implementations are in Python. Additional Python packages required include

* `batram`
    - [Environment setup notes](batram_environment.md)
* `netCDF4`
    - Reading ensemble datasets
* `numpy`, `pandas`, `datetime`
    - Data processing/analysis
* `quantile_supp`
    - Supporting routines in this repository
* `matplotlib`, `cartopy`
    - Visualization

***

CESM/CLM variables from large ensemble experiment

* FSNO, fraction of ground covered by snow
* H2OSNO, snow depth (liquid water equivalent)
* QRUNOFF, total liquid runoff
* RAIN, atmospheric rain
* SNOW, atmospheric snow
* SOILWATER_10CM, soil liquid water + ice in top 10cm of soil

***

## Copyright and Licensing Info

Copyright (c) 2023-24 California Institute of Technology (“Caltech”). U.S. Government sponsorship acknowledged. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Open Source License Approved by Caltech/JPL APACHE LICENSE, VERSION 2.0 • Text version: https://www.apache.org/licenses/LICENSE-2.0.txt • SPDX short identifier: Apache-2.0 • OSI Approved License: https://opensource.org/licenses/Apache-2.0
