Store all versions of models

1) `xgboost_trial.ipynb` : predicted variable = 1 as long as recommendation and activation matches (i.e, 0/0 and 1/1 = 1, 1/0 and 0/1 = 0)
2) `xgboost_v2.ipynb` : predicted variable = 1 only when recommendation and activation are both 1
     - submission_v1 output from this
     - xgboost with and without class weights compared --> using class weight increases predictions marginally
