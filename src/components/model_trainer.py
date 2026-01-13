import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import statsmodels.api as sm

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                # "XGBRegressor": XGBRegressor()
            }
            params = {
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                "AdaBoost Regressor":{
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'max_features': [0.1, 0.2, 0.3, 0.4],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "Ridge Regression":{
                    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    'fit_intercept': [True, False]
                },
                "Lasso Regression":{
                    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    'selection': ['random', 'cyclic']
                },
                "K-Neighbors Regressor":{
                    'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                # "XGBRegressor":{
                #     'learning_rate': [0.1, 0.01, 0.05],
                #     'n_estimators': [8,16,32,64,128,256]
                # }
            }
            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Model Report : {model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Exception("No best model found")
            logging.info(f"Best Model Found , Model Name: {best_model_name} , R2 Score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)



