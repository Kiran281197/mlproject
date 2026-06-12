import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import savefile

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function will create preprocessor object and save it in predefined locaiton 
        '''
        try:
            num_features = ["reading_score","writing_score"]
            cat_features = [
                "gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("standard_scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot_encoder", OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            return (
                preprocessor,
                num_features,
                cat_features
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation Initiated")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data is completed")

            logging.info("Obtaining the preprocessing object")

            preprocessor, num_features, cat_features = self.get_data_transformer_object()
            target_feature = "math_score"

            X_train = train_df.drop(target_feature, axis=1)
            X_test = test_df.drop(target_feature, axis=1)
            y_train = train_df[target_feature]
            y_test = test_df[target_feature]

            logging.info("train test split is done successfully")

            logging.info("Applying preprocessor object on training and testing dataframes.")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test) 

            train_arr = np.c_[
                X_train_arr, np.array(y_train)
            ]

            test_arr = np.c_[
                X_test_arr, np.array(y_test)
            ]

            logging.info("train and test data has been transformed successfully using preprocessor object.")

            savefile(
                obj = preprocessor, file_path=self.data_transformation_config.preprocessor_obj_file_path
            )

            logging.info(f"preprocessor object saved successfully. Path -> {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            ) 
        except Exception as e:
            raise CustomException(e, sys)


