## Spatial Prediction

This example demonstrates spatial prediction for locations with missing data using Bayesian transport maps. 
After the standard pre-processing steps from [other examples](swe_ens_process.md), 
the prediction sites are defined and prediction performance is documented.

* Define held out spatial locations: `lens_h2osno_add_predict_blocks.py`
* Produce maps and figures of example predictions: `lens_map_predblk.py`
* BTM prediction/scoring script: `lens_h2osno_score_predblk_btm_locslct.py`
* Nonstationary GP script: `lens_h2osno_score_predblk_gpns_locslct.py`
* Combine and summarize scoring: 
    - PIT histograms: `blkavg_scoring_smry.py`
    - CRPS, prediction interval summary: `lens_h2osno_score_predblk_smry.ipynb`
