

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from astar import AStar, Anchor


def prepare_dataset():
    df = pd.read_csv('adult_data_fixed.csv')
    df = df.apply(lambda x: x.fillna(x.median()), axis=0)
    df = df.apply(lambda x: x.astype(int))
    labels = df['income']
    features = df.drop(['income'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    # X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.2)
    return X_train, X_test, y_train, y_test


def is_relevant(example, anchor):
    relevant = True
    for k in anchor.keys():
        if example[k] != anchor[k]:
            relevant = False
            break
    return relevant



def get_data():
    df = pd.read_csv('data/train.csv')
    y_train = df['target']
    X_train = df.drop(['target'], axis = 1)
    df = pd.read_csv('data/test.csv')
    y_test = df['target']
    X_test = df.drop(['target'], axis = 1)
    df = pd.read_csv('data/train_predictions.csv')
    pred_train = df['target']
    df = pd.read_csv('data/test_predictions.csv')
    pred_test = df['target']
    return X_train, X_test,  y_train, y_test ,pred_train, pred_test

def train_model(X_train, y_train):
    clf = RandomForestClassifier(max_depth = 10)
    clf.fit(X_train, y_train)
    return clf

def run_experiment(pool_size, threshold, distance_method, X_train, pred_train, clf, X_test, y_test):
    anchors = []
    for i in range(pool_size):
        astar = AStar(X_train.iloc[i], pred_train.iloc[i], X_train.iloc[:200], pred_train[:200], clf, threshold,
                      list(X_train.columns), distance_method=distance_method)
        path = list(astar.astar(Anchor({}, list(X_train.columns))))
        anchors.append((path[-1].anchor, pred_train.iloc[i]))
    covered = 0
    precise = 0
    collisions = 0
    for i in range(len(X_test)):
        anchors_found = 0
        for j in range(len(anchors)):
            if is_relevant(X_test.iloc[i], anchors[j][0]) and anchors_found == 0:
                covered += 1
                anchors_found += 1
                if anchors[j][1] == y_test.iloc[i]:
                    precise += 1
            if is_relevant(X_test.iloc[i], anchors[j][0]) and anchors_found == 1:
                collisions += 1
                break
    precision = precise / covered
    coverage = covered / len(X_test)
    return precision, coverage


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, pred_train, pred_test = get_data()
    clf = train_model(X_train, y_train)
    pool_size = 50
    threshold = 0.92
    direct_precision, direct_coverage = run_experiment(pool_size, threshold, "direct", X_train, pred_train, clf, X_test, y_test)
    uniform_precision, uniform_coverage = run_experiment(pool_size, threshold, "uniform", X_train, pred_train, clf, X_test, y_test)
    print ("direct precision: ", direct_precision, "direct coverage: ", direct_coverage)
    print ("uniform precision: ", uniform_precision, "uniform coverage: ", uniform_coverage)







