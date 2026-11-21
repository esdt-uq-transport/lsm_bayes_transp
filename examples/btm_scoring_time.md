## Comparative Assessment for Held-Out Data at Multiple Time Points

This example describes evaluation of multiple generative models on held-out members of SWE ensembles. 
Before executing the evaluation scripts below, the ensemble data must be downloaded and location-specific quantiles computed 
for the target dates of interest as described in the [preprocessing example](swe_ens_process.md). Then, multiple passes of train/test 
partitions are selected and saved. For each train/test sample, the models are trained, and log scores (negative log likelihoods) are 
computed for the test fields. This example is designed to be run for any/all of a range of target dates as defined in a [metadata file](../config/LENS_BTM_Cases.csv).

* Optional: Plot selected city empirical CDFs for multiple time points: `lens_city_ecdf_decades.py`  
* Generate and save train/test collections: `gen_score_seq_ensfields.py`
* Train and score nonlinear and linear BTM models: `lens_h2osno_score_ens_btm_slct.py`
    - Command line arguments: `python lens_h2osno_score_ens_btm_slct.py LENS_BTM_Cases.csv ##` with `##` an integer for the desired row number
* Train and score nonstationary covariance model: `lens_h2osno_score_ens_gpns_slct.py`
    - Command line arguments: `python lens_h2osno_score_ens_gpns_slct.py LENS_BTM_Cases.csv ##` with `##` an integer for the desired row number
* Summarize performance for all cases: `lens_scoring_decades.py`

