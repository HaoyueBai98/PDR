import copy

import pandas as pd
import weka.core.converters as converters
import weka.core.jvm as jvm
from sklearn.metrics import make_scorer, r2_score
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.filters import Filter

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics.scorer import make_scorer
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

import numpy as np


class WekaEvaluator():
    def __init__(self, dataset_path, cv=5, parallel=True, flag='train'):
        self.data = pd.read_hdf(dataset_path, key=flag)
        self.data.reset_index(drop=True, inplace=True)
        self.cv = cv
        self.parallel = parallel

    def _weka_evaluate_r(self, data):
        data.class_is_last()
        model = Classifier(classname='weka.classifiers.trees.RandomForest')
        evl = Evaluation(data)
        evl.crossvalidate_model(model, data, self.cv, Random(0))
        s = 1 - evl.relative_absolute_error / 100
        return s

    def _weka_evaluate_c(self, data):
        weka_filter = Filter(
            classname="weka.filters.unsupervised.attribute.NumericToNominal",
            options=["-R", "last"]
        )
        weka_filter.inputformat(data)
        data = weka_filter.filter(data)
        data.class_is_last()
        model = Classifier(classname='weka.classifiers.trees.RandomForest')
        evl = Evaluation(data)
        evl.crossvalidate_model(model, data, self.cv, Random(0))
        fscore = evl.weighted_f_measure
        s = fscore
        return s

    def weka_evaluate(self, df, task_type):
        if not jvm.started:
            jvm.start()
        d = df.shape[0]
        x = str(df.values.tolist())
        data = np.reshape(eval(x), [d, -1], order='C')
        data = data.astype(np.float64)
        data = converters.ndarray_to_instances(
            data, relation='tmp'
        )
        if task_type == 'reg':
            score = self._weka_evaluate_r(data)
        elif task_type == 'cls':
            score = self._weka_evaluate_c(data)
        else:
            score = -1
        return score


class SklearnEvaluate():
    def __init__(self, dataset_path, n_estimators=10, scoring="f1_micro", model="RF", flag='train', cv=5):
        self.data = pd.read_hdf(dataset_path, key=flag)
        self.data.reset_index(drop=True, inplace=True)
        self.n_estimators = n_estimators
        self.scoring = scoring
        self.model = model
        self.cv = 5

    def sklearn_evaluate(self, df, task_type):
        df = copy.deepcopy(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(-1, inplace=True)

        if task_type == 'regression':
            if self.model == 'LR':
                model = Lasso()
            elif self.model == 'SVM':
                model = LinearSVR()
            elif self.model == "LGB":
                model = LGBMRegressor()
            elif self.model == "XGB":
                model = XGBRegressor()
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
            score = cross_val_score(model, df.iloc[:, :-1],  df.iloc[:, -1], scoring=make_scorer(r2_score),
                                    cv=int(self.cv)).mean()
        elif task_type == 'classification':
            if self.model == 'LR':
                model = LogisticRegression()
            elif self.model == 'SVM':
                model = LinearSVC()
            elif self.model == 'LGB':
                model = LGBMClassifier()
            elif self.model == "XGB":
                model = XGBClassifier()
            else:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
            score = cross_val_score(model, df.iloc[:, :-1],  df.iloc[:, -1],
                                    scoring=self.scoring, cv=int(self.cv)).mean()
        else:
            score = -1
        return score

    def _evaluate(self, df, task_type):
        return self.sklearn_evaluate(df, task_type)