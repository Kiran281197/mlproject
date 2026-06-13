import os
import sys

import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from src.logger import logging

def savefile(obj, file_path):
    '''
        This function will save the given object in the given file location.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            obj = dill.load(file)
            return obj 
    except Exception as e:
        raise CustomException(e, sys)
    