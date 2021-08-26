import os
from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None
    nfold  = 0
    for FOLD in range(2):
        print(f"#####  FOLD NUMBER {FOLD} ####")
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoders.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        for col in cols:
            lab = encoders[col]
            df.loc[:, col] = lab.transform(df[col].values.tolist())
    
        # Load model and cols
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
        nfold += 1
        print(f"##### FOLD NUMBER {FOLD}, PREDICTION {preds} ####")

    predictions /= nfold
    print(f"##### MEAN PREDICTION {predictions} ####")

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub

        
if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)

