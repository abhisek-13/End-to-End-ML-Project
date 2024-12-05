import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from src.utils import evaluate_models

# for testing
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifact','model.pkl')
  
class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    
  def initiate_model_trainer(self,train_arr,test_arr):
    try:
      logging.info('Spliting training and test input data.')
      
      x_train,y_train,x_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
      
      models = {
        "Linear Regression":LinearRegression(),
        "AdaBoost":AdaBoostRegressor(),
        "Random Forest":RandomForestRegressor(),
        "GradientBoost":GradientBoostingRegressor(),
        "Decision tree":DecisionTreeRegressor()
        }
      params = {"Linear Regression":{},
          "AdaBoost":{
          'learning_rate':[.1,.01,.05,.001],
          #'loss':['linear','square','exponential'],
          'n_estimators':[8,16,32,64,128,256]
        },
        "Random Forest":{
          #'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
          #'max_features':['sqrt','log2',None],
          'n_estimators':[8,16,32,64,128,256]
        },
        "GradientBoost":{
          #'loss':['squared_error','huber','absolute_error','quantile'],
          'learning_rate':[.1,.01,.05,.001],
          'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
          #'criterion':['squared_error','friedman_mse'],
          #'max_features':['auto','sqrt','log2'],
          'n_estimators':[8,16,32,64,128,256]
          
        },
        "Decision tree":{
          'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
          #'splitter':['best','random'],
          #'max_features':['sqrt','log2']
        }        
      }
      
      model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)
      
      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      
      best_model = models[best_model_name]
      
      if best_model_score < 0.6:
        raise CustomException("No best model found.")
      
      logging.info(f"we found the best model:- {best_model}")
      
      save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
      logging.info("Saved the model pkl file successfully.")
      
      predicted = best_model.predict(x_test)
      r2 = r2_score(y_test,predicted)
      return r2
    
    except Exception as e:
      raise CustomException(e,sys)
    
if __name__ == "__main__":
  obj = DataIngestion()
  train_path,test_path = obj.initiate_data_ingestion()
  
  dtobj = DataTransformation()
  train_arr,test_arr,preprocess_obj = dtobj.initiate_data_transformation(train_path,test_path)
  
  modelobj = ModelTrainer()
  r2_scoree = modelobj.initiate_model_trainer(train_arr,test_arr)
  print(r2_scoree)
 
