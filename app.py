import flask
from flask import Flask , request , render_template
import numpy as np
import pandas as pd

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)
from sklearn.preprocessing import StandardScaler
from src.Pipeline.prediction_pipeline import CustomData , PredictPipeline
application = Flask(__name__)
app=application

# Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            DRYBULBTEMPF=request.form.get("DRYBULBTEMPF"),
            RelativeHumidity=request.form.get("RelativeHumidity"),
            WindSpeed=request.form.get("WindSpeed"),
            WindDirection=request.form.get("WindDirection"),
            StationPressure=request.form.get("StationPressure"),
            Precip=request.form.get("Precip")
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=result[0])
    
if __name__ == '__main__':
    application.run(host="0.0.0.0",debug=True)
    

