

import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
import math

#load train and test

def prepare_data():
    train  = pd.read_csv('../input/train.csv', index_col=0)
    test  = pd.read_csv('../input/test.csv', index_col=0)
    labels = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)
    columns = train.columns
    test_ind = test.index
    train = np.array(train)
    test = np.array(test)

    # label encode the categorical variables
    for i in range(train.shape[1]):
        if type(train[1,i]) is str:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[:,i]) + list(test[:,i]))
            train[:,i] = lbl.transform(train[:,i])
            test[:,i] = lbl.transform(test[:,i])
    return train.astype(float), test.astype(float), labels, test_ind



def gini(solution, submission):
    df = zip(solution, submission)
    df = sorted(df, key=lambda x: (x[1],x[0]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini



def create_single_model(train, test, y, plst):
    num_rounds = 2000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices
    xgtrain = xgb.DMatrix(train, label=y)
    xgval = xgb.DMatrix(train, label=y)

    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
    preds = model.predict(xgtest)
    return preds


def create_model(train, test, y, plst):
    #Using 5000 rows for early stopping.
    offset = 5000
    preds1 = create_single_model(train=train[offset:,:], test=test, y=y[offset:], plst=plst)
    #reverse train and labels and use different 5k for early stopping.
    # this adds very little to the score but it is an option if you are concerned about using all the data.
    preds2 = create_single_model(train=train[::-1,:], test=test, y=y[::-1], plst=plst)
    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    return preds1+preds2


def create_submission(preds, test_ind, filename='xgboost_benchmark.csv'):
    preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds = preds.set_index('Id')
    preds.to_csv(filename)


def encode_decode(y,  enc_type, type_funcs={'encode':lambda x: x, 'decode':lambda x: x}):
    return map(lambda yi: type_funcs[enc_type](yi), y)


def load_params():
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["min_child_weight"] = 5
    params["subsample"] = 0.8
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 7
    params["learning_rate"] = 0.05
    params["n_estimator"] = 200

    plst = list(params.items())
    return plst

def main():
    train, test, label, test_ind  = prepare_data()
    dec_enc = {'encode':lambda x: math.sqrt(x), 'decode':lambda x: x*x}
    label = encode_decode(label, 'encode', dec_enc)
    params = load_params()
    preds = create_model(train=train, test=test, y=label, plst=params)
    preds = encode_decode(preds, 'decode', dec_enc)
    create_submission(preds, test_ind, filename="xgb_sqrt.csv")

main()
