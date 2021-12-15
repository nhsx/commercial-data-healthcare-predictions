#!/usr/bin/env python
# coding: utf-8

# Value of Commercial Product Sales Data in Healthcare Prediction
# 
# Author: Elizabeth Dolan Date: 22 November 2021
# 
# This code can be used to run MCR and grouped MCR on the optimal models of a Random Forest Regressor

# Code below does the following:
# -Imports key packages for MCR & analysis -Changes date to python date category type -Assigns y (the target variable the model needs to predict) -Assigns X (the features- variables inputted in order to predict y)-Splits data into training and test data -Check dataset sizes -Time series data split for cross validation - Cross validation grid search to find optimum hyperparameters for random forest regressor (remembering bootstrap parameter has to be set to False in order for MCR package to work) -Scores for R2, RSME and MAE on test data using model from grid search - Conduct MCR measuring importance of variables across Rashomon set - Save to file MCR scores and visual - Create grouped variables(features) for grouped MCR - Conduct grouped MCR
# Important note: In order for MCR to work you must have already created a 'Rashomon set' see report, a model who's instances produce a set of optimal ('best performing') models which can consistently make successful predictions. The code provided works effectively on a dataset that we cannot make public here due to health and commercial sensitivity/privacy - the example data given to run the code will not give you a model with high predictive accuracy i.e. the results on the MCR will not output valid/useful results - however, you can adapt this code to run on your own 'Rashomon sets' created in your own work and using other datasets.

#python 3.6.9

#Import some key packages for MCR
from mcrforest.forest import RandomForestRegressor # version 3.0.1
import pickle # version 4.0
import mcrforest
import numpy as np # version 1.19.5

from pkg_resources import get_distribution
print(get_distribution('mcrforest').version)

#import packages for analysis
import pandas as pd # version 1.1.5
import matplotlib.pyplot as plt # version 3.3.4
import seaborn as sns # version 0.11.2

from sklearn.model_selection import TimeSeriesSplit # version 0.24.2
from sklearn.ensemble import RandomForestRegressor as RF_sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#create dataframe from survey data
df = pd.read_csv('data/fake_feature_metadata.csv')

#change column to data type 'category', change date column data to 'datetime' data type
df['date'] = pd.to_datetime(df['date'])

#assign data to y target
y = df['cnt']
print('y','\n',y)

#assign data to X features
X = df[['weeknum', 'ltla_week_sales_17', 'decongestant_17', 'throat_17', 
        'cough_dry_17','cough_all_17', 'ltla_week_sales_24', 'decongestant_24',
        'throat_24', 'cough_dry_24', 'cough_all_24','decongestant_lr',
        'throat_lr', 'cough_dry_lr', 'cough_all_lr','decongestant_m',
        'throat_m', 'cough_dry_m', 'cough_all_m', 'liv_env_score', 'crime_score', 
        'housing_score', 'pop_16to24', 'pop_25to49', 'pop_50to64', 'pop_over65', 'pop_density', 'pct_male', 
        'pct_female', 'imd_rank', 'imd_score', 'imd_extent', 'imd_concentration', 'pct_pre1919', 'pct_pre1940', 
        'pct_pre1973', 'pct_pre1983', 'pct_community', 'pct_industrial', 'pct_residential', 'pct_transport', 
        'pct_agriculture', 'pct_natural', 'pct_recreation', 'pct_non_white', 'pct_lone_parent', 
        'pct_other_children', 'pct_detached', 'pct_semi', 'pct_terraced', 'pct_flat', 'average_rainfall',
        'total_rainfall', 'min_temp', 'average_temp', 'max_temp']] 


print('X','\n',X)

# data split into train and test data

X_train = X.iloc[:45844,:]
X_test = X.iloc[45844:66254,:]
y_train = y.iloc[:45844,]
y_test = y.iloc[45844:66254,]

# data sizes
print(len(X_train),len(y_train), 'train examples')
print(len(X_test),len(y_test), 'test examples')

#split training data in order to optimise RF model on it
#test_size set to ensure no data leakage at 13,188
#45,844 of training data rows, 314 instances of 146 weeks, around a fifth of 146 is 29, 29x314 = 9106

tscv = TimeSeriesSplit(n_splits=4, test_size=9106)
print(tscv)

#Cross validation grid search to find optimum hyperparameters for random forest regressor
rfc = RF_sklearn(random_state = 42)
param_grid = { 
            "n_estimators"      : [200, 300, 400], #300 otpimum tested 100,600
            "max_features"      : ["log2"], #"log2" is optimum tested 1,2, "auto"
            "min_samples_split" : [10,11,12], #11 is optimum tested 9,10,11,12
            "max_depth": [10,11,12], #11 is optimum tested 8,9,10,11,12
            "bootstrap": [False],
            }
grid = GridSearchCV(rfc, param_grid, cv=tscv, refit= True, n_jobs = -1, verbose = 3, return_train_score = True)
grid.fit(X_train, y_train.values.ravel())
predictions = grid.predict(X_test)
print(grid.best_params_)

#Scores for R2, RSME and MAE on test data using model from grid search
rf_mse = mean_squared_error(y_test, predictions)
rf_rmse = np.sqrt(rf_mse)
print('RMSE', rf_rmse)
print('r2', r2_score(y_test, predictions))
print('MAE', mean_absolute_error(y_test, predictions))

# conduct MCR
best_params = grid.best_params_
best_params['random_state'] = 42
best_params['n_jobs'] = -1

modelg = RandomForestRegressor(**best_params)
#modelg.debug()
modelg.fit(X_train.values[:45844,:], y_train.values.flatten()[:45844])
plt.rcParams.update({'font.size': 7})
r1 = modelg.plot_mcr(X_train.values[:45844,:], y_train.values.flatten()[:45844], num_times = 10, show_fig = False, feature_names = X_train.columns.tolist() )

#tabled MCR scores
print(r1)

#save to file tabled MCR scores
r1.to_csv("fake_outputs/mcr_padrus_scores.csv")

# Make a list of columns to create grouped variables for grouped MCR

week_col = ['weeknum']

sales_cols = ['ltla_week_sales_17', 'decongestant_17', 'throat_17', 
        'cough_dry_17','cough_all_17', 'ltla_week_sales_24', 'decongestant_24',
        'throat_24', 'cough_dry_24', 'cough_all_24','decongestant_lr',
        'throat_lr', 'cough_dry_lr', 'cough_all_lr','decongestant_m',
        'throat_m', 'cough_dry_m', 'cough_all_m']

imd_cols = ['liv_env_score', 'crime_score','housing_score','imd_rank', 'imd_score',
            'imd_extent', 'imd_concentration']

age_cols = ['pop_density','pop_16to24', 'pop_25to49', 'pop_50to64', 'pop_over65']

demo_cols =['pct_male','pct_female','pct_non_white', 'pct_lone_parent','pct_other_children']

housing_cols = ['pct_pre1919', 'pct_pre1940','pct_pre1973', 'pct_pre1983','pct_detached', 'pct_semi',
                'pct_terraced', 'pct_flat']

land_use_cols = ['pct_community', 'pct_industrial', 'pct_residential', 'pct_transport','pct_agriculture',
                 'pct_natural', 'pct_recreation']

weather_cols = ['average_rainfall','total_rainfall', 'min_temp', 'average_temp', 'max_temp']


# Set up the grouped variabled for grouped MCR

mcr_groupings =  [week_col, sales_cols, imd_cols, age_cols, demo_cols, housing_cols, land_use_cols, weather_cols]

grouping_names = [ 'week', 'sales', 'imd', 'age', 'demo', 'housing', 'land_use', 'weather']

grouping_names2indexes = {}
for i, g in enumerate(grouping_names):
    grouping_names2indexes[g] = np.asarray([ X_train.columns.tolist().index(v) for v in mcr_groupings[i] ])

## Grouped MCR 

r2 = modelg.plot_mcr(X_train, y_train.values.flatten(), feature_groups_of_interest = mcr_groupings, feature_names = grouping_names, num_times = 10, show_fig = False )

#Table scores for grouped MCR
print(r2)
