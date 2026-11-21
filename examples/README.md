### Examples

Additional examples illustrating Bayesian transport maps for land surface model ensembles. 
The documentation in this directory provides an overview of the processing steps and relevant python scripts for their execution. 
The scripts can be found in the repository `script` directory. Supporting routines are available in `lib` and supporting metadata are in `config`.

* [Ensemble preprocessing](swe_ens_process.md): Notes and scripts for downloading and preprocessing the ensemble data
* [BTM fitting and simulation](fit_sim_btm.md): Fitting BTM for ensembles under contemporary and future climate scenarios
* [Comparative assessment](btm_scoring.md): Evaluating multiple generative models for held-out ensemble members
* [Comparative assessment, multiple time points](btm_scoring_time.md): Evaluating multiple generative models for held-out ensemble members across multiple time points
* [Hyperparameter sensitivity](neighbor_sensitivity.md): Comparison of goodness of fit for range of selected neighborhood size
* [CESM-LE2](lens2_access.md): Datasets and BTM fitting for second CESM large ensemble experiment
* [Spatial prediction](predict_blocks.md): BTM fitting and prediction for blocks of locations with missing observations
