import joblib
import pandas as pd

# REPLACE W IMPUTED DATA
eval_data = pd.read_parquet('data/evaluation_data.parquet')

# CHANGE THIS IF FILE NAME DIFF
selected = pd.read_csv('selected_features.csv')

eval_customer = eval_data['customer']
eval_merchant = eval_data['merchant']
eval_data = eval_data[selected['Selected Features']] # X values for the evaluation data, final predictions made on this

final_model = joblib.load('submission_v3_model.joblib')

# Predict the probability of activation on the evaluation set
print('Making predictions...')
eval_pred_proba = final_model.predict_proba(eval_data)[:, 1]

# create dataframe with predictions according to submission guidelines (customer, merchant, predicted_score)
submission = pd.DataFrame({'customer': eval_customer, 'merchant': eval_merchant, 'predicted_score': eval_pred_proba})

# Save the submission to a CSV file
print('Saving the submission...')
submission.to_csv('data/submission_v2.csv', index=False)