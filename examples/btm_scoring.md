## Comparative Assessment for Held-Out Data 

This example describes evaluation of multiple generative models on held-out members of SWE ensembles. 
Before executing the evaluation scripts below, the ensemble data must be downloaded and location-specific quantiles computed 
for the target date of interest as described in the [preprocessing example](swe_ens_process.md). Then, multiple passes of train/test 
partitions are selected and saved. For each train/test sample, the models are trained, and log scores (negative log likelihoods) are 
computed for the test fields.

* Generate and save train/test collections: `gen_score_seq.py`
* Train and score nonlinear and linear BTM models: `lens_h2osno_score_ens_btm_epoch.py`
* Train and score nonstationary covariance model: `lens_h2osno_score_ens_gp_epoch.py`

