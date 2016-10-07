import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

n_estimator = 10
X, y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)
# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = cross_validation.train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
    random_state=0)

rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)


le = preprocessing.LabelEncoder()
le.fit(['a', 'b', 'c', 'test'])
le.transform(['b','test','c','b'])

enc = preprocessing.OneHotEncoder()
enc.fit([[1,2,3,4]])
enc.transform([1,2,1,3]).toarray()

enc.fit([le.transform(['a', 'b', 'c', 'test'])])
enc = preprocessing.OneHotEncoder()
enc.fit([[1,2,3,4]])
enc.transform([1, 2, 2, 1]).toarray()
enc.n_values_


class ColumnApplier(object):
    def __init__(self, column_stages):
        self._column_stages = column_stages

    def fit(self, X, y):
        for i, k in self._column_stages.items():
            k.fit(X[:, i])

        return self

    def transform(self, X):
        X = X.copy()
        for i, k in self._column_stages.items():
            X[:, i] = k.transform(X[:, i])
        return X

X = np.array([['a', 'dog', 'red'], ['b', 'cat', 'green']])
y = np.array([1, 2])
multi_encoder = \
    ColumnApplier(dict([(i, preprocessing.LabelEncoder()) for i in range(3)]))
multi_encoder.fit(X, None).transform(X)


