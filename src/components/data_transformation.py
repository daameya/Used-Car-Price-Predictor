import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
#from sklearn.feature_extraction.text import h

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        """ 
        This function is responsible for data transformation
        
         """
        try:
            numerical_columns = [ "year", "mileage(kilometers)", "volume(cm3)"]

            categorical_columns = [
                'make',
                'condition',
                'fuel_type',
                'color',
                'transmission',
                'drive_unit',
                'segment',
            ]

            #Define the custom ranking for each ordinal variable
            make_cat = ['mazda', 'mg', 'renault', 'gaz', 'aro', 'rover', 'uaz', 'alfa-romeo', 'audi',
            'oldsmobile', 'saab', 'peugeot', 'chrysler', 'wartburg', 'moskvich', 'volvo',
            'fiat', 'roewe', 'porsche', 'zaz', 'luaz', 'dacia', 'lada-vaz', 'izh', 'raf',
            'bogdan', 'bmw', 'nissan', 'mercedes-benz', 'mitsubishi', 'toyota', 'chery',
            'gmc', 'hyundai', 'honda', 'ssangyong', 'suzuki', 'opel', 'seat', 'volkswagen',
            'daihatsu', 'chevrolet', 'geely', 'saturn', 'kia', 'lincoln', 'eksklyuziv',
            'citroen', 'dong-feng', 'pontiac', 'ford', 'subaru', 'bentley', 'faw',
            'cadillac', 'lifan', 'plymouth', 'hafei', 'shanghai-maple', 'mini', 'jeep',
            'skoda', 'mercury', 'changan', 'lexus', 'isuzu', 'aston-martin', 'lancia',
            'great-wall', 'land-rover', 'jaguar', 'buick', 'daewoo', 'vortex', 'infiniti',
            'byd', 'smart', 'maserati', 'haval', 'acura', 'scion', 'tata', 'datsun', 'tesla',
            'mclaren', 'ravon', 'trabant', 'proton', 'fso', 'jac', 'asia', 'iran-khodro',
            'zotye', 'tagaz', 'saipa', 'brilliance']

            condition_cat = ['with mileage', 'with damage', 'for parts']

            fuel_type_cat = ['petrol', 'diesel', 'electrocar']

            color_cat = ['burgundy', 'black', 'silver', 'white', 'gray', 'blue', 'other', 'purple', 'red',
            'green', 'brown', 'yellow', 'orange']

            transmission_cat = ['mechanics', 'auto']

            drive_unit_cat = ['front-wheel drive', 'rear drive', 'all-wheel drive',
            'part-time four-wheel drive']

            segment_cat = ['B', 'C', 'D', 'M', 'E', 'A', 'J', 'S', 'F']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))
                
                ]
            )

            cat_pipeline= Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder",OrdinalEncoder(categories=[make_cat,condition_cat,fuel_type_cat,color_cat,transmission_cat,drive_unit_cat,segment_cat])),
                ("scaler",StandardScaler(with_mean=False))
                ]
                
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            #logging.info("Numerical columns standard scaling completed")

            #logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
              
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="priceUSD"
            numerical_columns = ["year", "mileage(kilometers)", "volume(cm3)"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)