import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')
  
  
class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()
    
  def get_data_transformation_object(self):
    try:
      numerical_columns = ['writing_score','reading_score']
      categorical_colums = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
      
      numerical_pipeline = Pipeline(
        steps=[('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]
      )
      categorical_pipeline = Pipeline(
        steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot_encoder',OneHotEncoder()),('scaler',StandardScaler(with_mean=False))]
      )
      
      logging.info(f"numerical columns: {numerical_columns}")
      logging.info(f"categorical columns: {categorical_colums}")
      
      preprocessor = ColumnTransformer([('num_pipeline',numerical_pipeline,numerical_columns),('cat_pipeline',categorical_pipeline,categorical_colums)])
      
      return preprocessor
      
    except Exception as e:
      raise CustomException(e,sys)
    
  def initiate_data_transformation(self,train_path,test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)
      
      logging.info("Reading train and test data completed.")
      logging.info("Obtaining preprocessor object.")
      
      target_column_name = "math_score"
      numerical_columns = ['writing_score','reading_score']
      
      # for train data
      input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
      target_features_train_df = train_df[target_column_name]
      
      # for test data
      input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
      target_features_test_df = test_df[target_column_name]
      
      logging.info("Applying data preprocessing in training and test dataframe.")
      
      preprocessor_obj = self.get_data_transformation_object()
      
      input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
      input_features_test_arr = preprocessor_obj.transform(input_features_test_df)
      
      train_arr = np.c_[input_features_train_arr,np.array(target_features_train_df)]
      test_arr = np.c_[input_features_test_arr,np.array(target_features_test_df)]
      
      logging.info("Saved preprocessing object.")
      
      
      save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj)
      
      return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
      
      
      
    except Exception as e:
      raise CustomException(e,sys)
    
    
