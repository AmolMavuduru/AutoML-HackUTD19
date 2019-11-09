import numpy as np
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tpot import TPOTClassifier, TPOTRegressor
from datacleaner import autoclean
from collections import defaultdict

class AutoMLEstimator(object):
    
    def __init__(self, **kwargs):
        
        self.task = kwargs['task']
        self.speed = kwargs['speed']
        self.test_size = kwargs['test_size']
        if self.task == 'Classification':
            self.tpot_model = TPOTClassifier(generations=self.speed, population_size=self.speed*10, verbosity=2, n_jobs=-1)
        else:
            self.tpot_model = TPOTRegressor(generations=self.speed, population_size=self.speed*10, verbosity=2, n_jobs=-1)
        
    def preprocess_data(self, data, target_column):
        
        clean_data = autoclean(data)
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
    
    def evaluate_model(self):
        
        metrics = defaultdict()
        pred = self.tpot_model.fitted_pipeline_.predict(self.X_test)
        
        if self.task == 'Classification':
            
            metrics['task'] = 'Classification'
            metrics['accuracy'] = int(100*accuracy_score(pred, self.y_test))
            metrics['precision'] = int(100*precision_score(pred, self.y_test))
            metrics['recall'] = int(100*recall_score(pred, self.y_test))
            
        else:
            
            metrics['task'] = 'Regression'
            metrics['r2_score'] = r2_score(pred, self.y_test)
            metrics['mean_absolute_error'] = mean_absolute_error(pred, self.y_test)
            metrics['mean_squared_error'] = mean_squared_error(pred, self.y_test)
        
        metrics['model_name'] = type(self.tpot_model.fitted_pipeline_[-1]).__name__
        return metrics
