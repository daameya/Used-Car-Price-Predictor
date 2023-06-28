
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            make=request.form.get('make'),
            condition=request.form.get('condition'),
            fuel_type=request.form.get('fuel_type'),
            color=request.form.get('color'),
            transmission=request.form.get('transmission'),
            drive_unit=request.form.get('drive_unit'),
            segment=request.form.get('segment'),
            year=request.form.get('year'),
            mileage_kilometers=request.form.get('mileage_kilometers'),
            volume_cm3=request.form.get('volume_cm3')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")