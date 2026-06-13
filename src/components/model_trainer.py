import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import savefile

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression" : LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor(),
                "KNearest Neighbour" : KNeighborsRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "Gradient Boost" : GradientBoostingRegressor(),
                "XGBoost" : XGBRegressor(),
                "CatBoost" : CatBoostRegressor(verbose=False)
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boost":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNearest Neighbour":{
                    'n_neighbors' : [5,7,9,11]
                }
                
            }

            model_lst = []
            r2_score_lst = []

            logging.info("Model Training Started")

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]

                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model = model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)

                r2_score_lst.append(r2)
                model_lst.append(model)

            logging.info("Model Training completed Successfully")

            highest_r2_score_index = np.argmax(np.array(r2_score))
            best_model = model_lst[highest_r2_score_index]
            best_model_score = max(r2_score_lst)

            if best_model_score<0.6:
                raise CustomException("Best model not found")
            logging.info("Best model found on training and test data")

            logging.info(f"Saving the best model. Path : {self.model_trainer_config.trained_model_file_path}")
            savefile(best_model, self.model_trainer_config.trained_model_file_path)

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            best_model_score = r2_score(y_test, y_pred)

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)


    