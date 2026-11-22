## Hyperparameter Sensitivity  

This example evaluates the sensitivity of BTM hyperparameters for the number of nearest neighbors for conditioning. 
Before executing the evaluation scripts below, the ensemble data must be downloaded and location-specific quantiles computed 
for the target date of interest as described in the [preprocessing example](swe_ens_process.md). In addition, train/test sample sequences 
previously generated with `gen_score_seq_ensfields.py` are used. In this example, the standard BTM with 
all hyperparameters optimized is fit initially. Then the number of nearest neighbors is set to multiple values and BTMs are fit with 
that hyperparameter fixed. For each train/test sample, the models are trained, and log scores (negative log likelihoods) are 
computed for the test fields.

* If needed, generate and save train/test collections: `gen_score_seq_ensfields.py`
* Train and score sequences of fixed neighborhood BTMs: `lens_h2osno_trsnf_fix_param_fields.py`
* Summary figure with scores: `lens_scoring_nbrseq.py`

