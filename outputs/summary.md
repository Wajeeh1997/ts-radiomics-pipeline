# Soft-Tissue-Sarcoma Lung Metastasis Prediction — Block Comparison

| feature_block   | model        |   mean_auc |   std_auc |   permutation_p |   n_features |   n_patients |
|:----------------|:-------------|-----------:|----------:|----------------:|-------------:|-------------:|
| clinical        | RandomForest |   0.916667 | 0.0389415 |      0.00990099 |          105 |           51 |
| combined        | RandomForest |   0.84881  | 0.124665  |      0.00990099 |          319 |           51 |
| radiomics       | RandomForest |   0.789286 | 0.154193  |      0.00990099 |          214 |           51 |

Cohort: n=51 (TCIA/IDC Soft-Tissue-Sarcoma, Vallieres et al. 2015). Small-cohort caveat applies — treat AUCs as directional.
