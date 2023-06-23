import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        make: str,
        condition: str,
        fuel_type: str,
        color: str,
        transmission: str,
        drive_unit: str,
        segment: str,
        year: int,
        mileage_kilometers: int,
        volume_cm3: int
    ):
        self.make = make
        self.condition = condition
        self.fuel_type = fuel_type
        self.color = color
        self.transmission = transmission
        self.drive_unit = drive_unit
        self.segment = segment
        self.year = year
        self.mileage_kilometers = mileage_kilometers
        self.volume_cm3 = volume_cm3

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "make": [self.make],
                "condition": [self.condition],
                "fuel_type": [self.fuel_type],
                "color": [self.color],
                "transmission": [self.transmission],
                "drive_unit": [self.drive_unit],
                "segment": [self.segment],
                "year": [self.year],
                "mileage(kilometers)": [self.mileage_kilometers],
                "volume(cm3)": [self.volume_cm3]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
