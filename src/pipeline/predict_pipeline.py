import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            result = model.predict(data_scaled)

            return result
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: int,
                 parental_level_of_education : str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score : int,
                 writing_score : int):
        
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
    
    def get_data_as_dataframe(self):
        try:
            input_list = [
                self.gender,
                self.race_ethnicity,
                self.parental_level_of_education,
                self.lunch,
                self.test_preparation_course,
                self.reading_score,
                self.writing_score
            ]

            column_names = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                "reading_score",
                "writing_score"
            ]

            input_df = pd.DataFrame([input_list], columns=column_names)

            return input_df
        except Exception as e:
            raise CustomException(e, sys)
        
    

    


