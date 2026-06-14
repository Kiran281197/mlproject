from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import sys

from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)

app = application

## Route for a home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    try:
        if request.method=="GET":
            return render_template("home.html")
        else:
            custom_data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch = request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=float(request.form.get("reading_score")),
                writing_score=float(request.form.get("writing_score"))
            ) 

            df_input = custom_data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            output = predict_pipeline.predict(df_input)

            return render_template("home.html", results=round(output[0],2))
    except Exception as e:
        raise CustomException(e, sys)


if __name__=="__main__":
    app.run(host="0.0.0.0")
