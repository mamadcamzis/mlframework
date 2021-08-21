import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = []
    for col in train_df.columns:
        lab = preprocessing.LabelEncoder()
        lab.fit((train_df[col].values.tolist() + valid_df[col].values.tolist()))
        train_df.loc[:, col] = lab.transform(train_df[col].values.tolist())
        valid_df.loc[:, col] = lab.transform(valid_df[col].values.tolist())
        label_encoders.append((col, lab))


    # Data is ready to be trained
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(preds)
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoders, f"models/{MODEL}_label_encoders.pkl")
    joblib.dump(clf, f"models/{MODEL}.pkl")
