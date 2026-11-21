## CESM-LE2

This example demonstrates preprocessing and transport map fitting for the CESM2 large ensemble (CESM-LE2).

* [LENS2 datasets](https://doi.org/10.26024/KGMP-C556) are available from the NCAR Geoscience Data Exchange (GDEX)
    - SWE outputs available from the land daily files collection
    - SWE for future climate scenarios under the SSP 370 scenario can be subset with the wildcard, e.g. `H2OSNO.2075`. This example uses the `CMIP6` sub-ensemble only, not the `SMBB` collection
* Construction of quantiles and parametric fits: `lens2_h2osno_quantile_gamfit.py`
    - Quantile summary for selected locations: `lens2_city_ecdf_months.py`
* K-means cluster analysis: `lens2_h2osno_kmeans_ensfields.py`
* Cross validation sampling sequences: `gen_score_seq_ensfields.py`
* BTM estimation and scoring: `lens2_h2osno_score_ens_btm_slct.py`
* GP estimation and scoring: `lens2_h2osno_score_ensfields_gp_slct.py`
* Scoring summary for multiple dates: `lens2_scoring_months.py`
