import numpy as np
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tpot import TPOTClassifier, TPOTRegressor
from datacleaner import autoclean
from collections import defaultdict, OrderedDict
from joblib import dump, load
import os
import io
import base64
import operator
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt

class AutoMLEstimator(object):

    def __init__(self, **kwargs):

        self.task = kwargs['task']
        self.speed = kwargs['speed']
        self.max_eval_time = 3*self.speed/2
        self.test_size = kwargs['test_size']
        if self.task == 'Classification':
            self.tpot_model = TPOTClassifier(generations=self.speed, population_size=self.speed*5, 
                verbosity=2, n_jobs=-1, max_eval_time_mins=self.max_eval_time)
        else:
            self.tpot_model = TPOTRegressor(generations=self.speed, population_size=self.speed*5, 
                verbosity=2, n_jobs=-1, max_eval_time_mins=self.max_eval_time)

    def remove_unnamed_columns(self, data):

        columns = list(data.columns)
        unnamed_cols = [col for col in columns if 'Unnamed' in col]
        return data.drop(unnamed_cols, axis=1)
        
    def preprocess_data(self, data, target_column):

        clean_data = autoclean(data)
        clean_data = self.remove_unnamed_columns(clean_data)
        X = clean_data.drop(target_column, axis=1)
        y = clean_data[target_column]

        return X, y

    def split_data(self, X, y):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size)

    def fit_model(self):

        self.tpot_model.fit(self.X_train, self.y_train)

    def run_automl(self, data, target_column):

        self.X, self.y = self.preprocess_data(data, target_column)
        self.split_data(self.X, self.y)
        self.fit_model()

    def get_feature_importances(self):

	    img = io.BytesIO()

	    if self.task == 'Classification':
	    	model = XGBClassifier(n_jobs=-1)
	    else:
	    	model = XGBRegressor(n_jobs=-1)

	    model.fit(self.X, self.y)
	    importances = model.feature_importances_

	    plt.figure(figsize=(12, 6))
	    plt.title('Most Important Features')
	    plt.xlabel('Feature')
	    plt.ylabel('Importance')
	    feature_names = list(self.X.columns)
	    feat_importances = dict(zip(feature_names, importances))
	    feat_importances = OrderedDict(sorted(feat_importances.items(), key=operator.itemgetter(1), reverse=True)[:5])
	    plt.xticks(range(len(feat_importances)), list(feat_importances.keys()))
	    plt.bar(range(len(feat_importances)), list(feat_importances.values()), align='center')
	    plt.savefig(img, format='png')
	    img.seek(0)
	    graph_url = base64.b64encode(img.getvalue()).decode()
	    plt.close()
	    return 'data:image/png;base64,{}'.format(graph_url)


    def evaluate_model(self):

        metrics = defaultdict()
        pred = self.tpot_model.fitted_pipeline_.predict(self.X_test)

        if self.task == 'Classification':

            metrics['task'] = 'Classification'
            metrics['accuracy'] = int(100*accuracy_score(pred, self.y_test))
            metrics['precision'] = int(100*precision_score(pred, self.y_test, average="weighted"))
            metrics['recall'] = int(100*recall_score(pred, self.y_test, average="weighted"))

        else:

            metrics['task'] = 'Regression'
            metrics['r2_score'] = r2_score(pred, self.y_test)
            metrics['mean_absolute_error'] = mean_absolute_error(pred, self.y_test)
            metrics['mean_squared_error'] = mean_squared_error(pred, self.y_test)

        metrics['model_name'] = type(self.tpot_model.fitted_pipeline_[-1]).__name__
        metrics['feat_importances'] = self.get_feature_importances()
        return metrics

    def save_model(self, directory):

    	model_pipeline = self.tpot_model.fitted_pipeline_
    	self.model_name = type(model_pipeline[-1]).__name__
    	dump(model_pipeline, os.path.join(directory, './{}.joblib'.format(self.model_name)))


