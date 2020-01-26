"""
Provide lead score predictions for new leads

Inputs:
    - a CSV formatted file named `input_predict.csv` located in this folder, containing the last 12333 rows of the original dataset but without "is booking" column.
    - This dataset needs to be cleaned using the function `clean_predict_data` in utils, before passed into the pipeline to generate lead score prediction.
Outputs:
    - a file named `output_predict.csv` that contains the same input file plus a new column "lead_score" with the predicted lead score
To run:
    $python predict.py
"""
import pandas as pd
from sklearn.base import BaseEstimator
from utils import clean_predict_data, load_pickle

import warnings
warnings.filterwarnings("ignore")

input_predictset_fname = "input_predict.csv"
output_predictset_fname = "output_predict.csv"

class ReshapeTransformer(BaseEstimator):
    def __init__(self):
        self.is_fitted = False

    def transform(self, X, y=None):
        return X.reshape(X.shape[0],)

    def fit(self, X, y=None):
        self.is_fitted = True
        return self
    
    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)


def main():
    """
    Main functiaon.
    1 .Loads the input predict data
    2. loads a trained machine learning pipeline
    3. adds lead score predictions to the file and saves it
    """
    print("LOAD INPUT DATA")
    input_data = pd.read_csv(input_predictset_fname)

    print("CLEAN INPUT DATA")
    cleaned_input_data = clean_predict_data(input_data)

    print("PREDICTING LEAD SCORE")
    pipeline = load_pickle("pipeline.pkl")
    input_data["lead_score"] = pipeline.predict_proba(cleaned_input_data)[:,1]
    
    print("SAVING OUTPUT")
    input_data.to_csv(output_predictset_fname, index=False)


if __name__ == "__main__":
    main()
