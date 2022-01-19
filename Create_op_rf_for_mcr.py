#!/usr/bin/env python
# coding: utf-8

# Value of Commercial Product Sales Data in Healthcare Prediction
# 
# Author: Elizabeth Dolan
# Date: 19 November 2021
# 
# This code can be used to create an optimised random forest regressor (using a time series cross validation grid search) that can be evaluated using Model Class Reliance (code for running MCR in separate file).  This code can be used and adapted to create baseline model and final models.
# 
# Code below does the following:
# 
# -Imports some key packages 
# -Creates dataframe from csv
# -Checks and describes the dataframe
# -Changes date to correct date format within python
# -Assigns y (the target - varaible the model needs to predict)
# -Assigns X (the features- variables inputted in order to predict y)
# -Splits data into training and test data (here we also split data to test separately on data from the pandemic timeframe). Data is manually split to prevent dataleakage as dataframe contains timeseries data with multiple entries from each date for different geographic areas.
# -Time series split (data) for cross validation (Time series cross validation). Manually checked, again to prevent data leakage.
# -Cross validation grid search to find optimum hyperparameters for random forest regressor (remembering bootstrap parameter has to be set to False in order for MCR package to work later)
# -Scores for R2, RSME and MAE on test data using model from grid search
# -Create optimised random forest regressor model using parameters given by grid search "op_rf"
# -Scores for R2, RSME and MAE on training data, test data, covid test data, and full dataset
# -Scatter Plot full dataset, showing predictions and targets
# -Line plots showing (1)predictions (2) predictions and actual targets
# -Plots feature importance using the following variable importance tools: random forest, permutation, SHAP

# python 3.6.9

#Imports some key packages
import pandas as pd # version 1.1.5
import numpy as np # version 1.19.5
import matplotlib.pyplot as plt # version 3.3.4
import seaborn as sns # version 0.11.2
from sklearn.model_selection import TimeSeriesSplit # version 0.24.2
from sklearn.ensemble import RandomForestRegressor as RF_sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import shap

#Creates dataframe from csv
#fyi in this analysis imported csv is sorted by date - this data order is needed in order for data splitting without leakage later on
df = pd.read_csv('data/fake_feature_metadata.csv')
print('\n','----Check read in data----')
print('Data shape','\n', df.shape) #how many rows and columns in data
print('Data rows for each local authorities','\n', df['ltla_code.1'].value_counts()) # how many ltla's (lower tier local authorities)
print('Data rows for each week','\n', df['date'].value_counts()) #how many dates (and for each ltla)
print('Description of data','\n', df.describe()) #describes dataframe
print('Total sales (17 day lag)','\n', df['ltla_week_sales_17'].sum()) #sums total sales
print('Total cough sales (17 day lag)','\n', df['cough_all_17'].sum()) #sums total cough sales (17 days in advance)
print('Total deaths','\n', df['cnt'].sum()) #sums deaths from respiratory disease
print('Data head','\n', df.head()) #check start of data incl. start date
print('Data tail','\n', df.tail()) #check end of data incl. end date

#change date column data to 'datetime' data type
df['date'] = pd.to_datetime(df['date'])
print('Data head with new date format','\n', df.head()) # check date change
print('Data tail with new date format','\n', df.tail()) # check date change

# check date change
dc = df['date'].unique()
print('Number of unique dates','\n', len(dc))
print('Data types','\n', df.dtypes) # check date change

#assign data to y target
y = df['cnt']
print('\n','----assign data to y and x----')
print('y','\n', y) #check y

#assign data to X features
X = df[['weeknum', 'ltla_week_sales_17', 'decongestant_17', 'throat_17', 
        'cough_dry_17','cough_all_17','ltla_week_sales_24', 'decongestant_24',
        'throat_24', 'cough_dry_24', 'cough_all_24', 'decongestant_lr',
        'throat_lr', 'cough_dry_lr', 'cough_all_lr', 'decongestant_m',
        'throat_m', 'cough_dry_m', 'cough_all_m', 'liv_env_score', 'crime_score', 
        'housing_score', 'pop_16to24', 'pop_25to49', 'pop_50to64', 'pop_over65', 'pop_density', 'pct_male', 
        'pct_female', 'imd_rank', 'imd_score', 'imd_extent', 'imd_concentration', 'pct_pre1919', 'pct_pre1940', 
        'pct_pre1973', 'pct_pre1983', 'pct_community', 'pct_industrial', 'pct_residential', 'pct_transport', 
        'pct_agriculture', 'pct_natural', 'pct_recreation', 'pct_non_white', 'pct_lone_parent', 
        'pct_other_children', 'pct_detached', 'pct_semi', 'pct_terraced', 'pct_flat', 'average_rainfall',
        'total_rainfall', 'min_temp', 'average_temp', 'max_temp']] 


print('X', '\n', X) # check X
print('\n','----Check training and test data splits----')
print('Last row of training data - where data will be split', '\n', df.loc[45843]) #used to check manually selected data rows for split are correct
print('First row of testing data - where data will be split', '\n', df.loc[45844])

# data split into train and test data, extra covid test set
X_train = X.iloc[:45844,:]
X_test = X.iloc[45844:66254,:]
X_covid_test = X.iloc[66254:,:] #covid period
y_train = y.iloc[:45844,]
y_test = y.iloc[45844:66254,]
y_covid_test = y.iloc[66254:,] #covid period

print('X_train', '\n', X_train) #check training data
print('X_test', '\n', X_test) #check testing data
print('X_covid_test', '\n', X_covid_test) #check covid test data
print(len(X_train),len(y_train), 'train examples') # get data sizes
print(len(X_test),len(y_test), 'test examples')
print(len(X_covid_test),len(y_covid_test), 'covid test examples')
print('y_train shape', y_train.shape) # check correct data shapes for model
print('X_train shape', X_train.shape)

#split training data in order to optimise RF model on it
#test_size set to ensure no data leakage at 9106 [depends on size of training data so needs updating if training data size changes]
#45,844 of training data rows, 314 instances of 146 weeks, around a fifth of 146 is 29, 29x314 = 9106

tscv = TimeSeriesSplit(n_splits=4, test_size=9106)
print('\n','----Check no data leakage in splits----')
print(tscv)

#see data splits
for train_index, test_index in tscv.split(X_train):
     print("TRAIN:", train_index, "TEST:", test_index)

#used to check no data leakage in splits - manually input from above output
print("train 1",(df.loc[0,'date']),"to",(df.loc[9419,'date']),"test 1",(df.loc[9420,'date']),"to",(df.loc[18525,'date']))
print("train 2",(df.loc[0,'date']),"to",(df.loc[18525,'date']),"test 2",(df.loc[18526,'date']),"to",(df.loc[27631,'date']))
print("train 3",(df.loc[0,'date']),"to",(df.loc[27631,'date']),"test 3",(df.loc[27632,'date']),"to",(df.loc[36737,'date']))
print("train 4",(df.loc[0,'date']),"to",(df.loc[36737,'date']),"test 4",(df.loc[36738,'date']),"to",(df.loc[45843,'date']))

#used to check no data leakage in splits - manually input from above printed data splits output
print("train 1",(df.loc[0,'ltla_name']),"to",(df.loc[9419,'ltla_name']),"test 1",(df.loc[9420,'ltla_name']),"to",(df.loc[18525,'ltla_name']))
print("train 2",(df.loc[0,'ltla_name']),"to",(df.loc[18525,'ltla_name']),"test 2",(df.loc[18526,'ltla_name']),"to",(df.loc[27631,'ltla_name']))
print("train 3",(df.loc[0,'ltla_name']),"to",(df.loc[27631,'ltla_name']),"test 3",(df.loc[27632,'ltla_name']),"to",(df.loc[36737,'ltla_name']))
print("train 4",(df.loc[0,'ltla_name']),"to",(df.loc[36737,'ltla_name']),"test 4",(df.loc[36738,'ltla_name']),"to",(df.loc[45843,'ltla_name']))

print('\n','----Cross Validation Grid Search----')
#set parameters to test, and run cross validation grid search.
#NB manually alter parameters to limit the computational expense of running gridsearch to iteratively optimise see examples below
rfc = RF_sklearn(random_state = 42)
param_grid = { 
            "n_estimators"      : [200, 300, 400], # 300 optimum tested 100,600
            "max_features"      : ["log2"], #"log2" optimum tested 1,2, "auto"
            "min_samples_split" : [10,11,12], #11 optimum tested 9,10,11,12
            "max_depth": [10,11,12], #11 optimum tested 8,9,10,11,12
            "bootstrap": [False],
            }
grid = GridSearchCV(rfc, param_grid, cv=tscv, refit= True, n_jobs = -1, verbose = 3, return_train_score = True)
grid.fit(X_train, y_train.values.ravel())
predictions = grid.predict(X_test)
print(grid.best_params_)

#import packages to produce prediction scores
rf_mse = mean_squared_error(y_test, predictions)
rf_rmse = np.sqrt(rf_mse)
print('\n','----Prediction Scores----')
print('Scores from grid search model run on test data')
print('RMSE', rf_rmse)
print('r2', r2_score(y_test, predictions))
print('MAE', mean_absolute_error(y_test, predictions))

#create model with optimised parameters, manually inputted from above gridsearch outputs
op_rf = RF_sklearn(random_state=42, bootstrap = False, n_jobs = -1, n_estimators = 300, max_features = 'log2', max_depth = 11, min_samples_split = 11)

#fit model with training data
op_rf.fit(X_train, y_train)

# r squared (r2 score) for model op_rf predicting on training data
r_sq = op_rf.score(X_train, y_train)
print('coefficient of determination (r2) training data:', r_sq)

# r squared (r2 score) for model op_rf predicting on testing data
r_sq = op_rf.score(X_test,y_test)
print('coefficient of determination (r2) test data:', r_sq)

# r squared (r2 score) for model op_rf predicting on testing covid data
r_sq = op_rf.score(X_covid_test,y_covid_test)
print('coefficient of determination (r2) covid test data:', r_sq)

# r squared (r2 score) for model op_rf predicting on all data
r_sq = op_rf.score(X, y)
print('coefficient of determination (r2) full dataset:', r_sq)

# look at predicted responses
y_pred = op_rf.predict(X)
print('predicted response:', y_pred, sep='\n')

# mean absolute error score for data sets
RDdeath_predictions = op_rf.predict(X_train)
lin_mae = mean_absolute_error(y_train,RDdeath_predictions)
print('MAE, training data', lin_mae)

RDdeath_predictions = op_rf.predict(X_test)
lin_mae = mean_absolute_error(y_test,RDdeath_predictions)
print('MAE, test data', lin_mae)

RDdeath_predictions = op_rf.predict(X_covid_test)
lin_mae = mean_absolute_error(y_covid_test,RDdeath_predictions)
print('MAE, covid test data', lin_mae)

RDdeath_predictions = op_rf.predict(X)
lin_mae = mean_absolute_error(y,RDdeath_predictions)
print('MAE, full dataset',lin_mae)

#Root mean squared error score on datasets
RDdeath_predictions = op_rf.predict(X_train)
lin_mse = mean_squared_error(y_train,RDdeath_predictions)
lin_rmse = np.sqrt(lin_mse)
print('RMSE, training data',lin_rmse)

RDdeath_predictions = op_rf.predict(X_test)
lin_mse = mean_squared_error(y_test,RDdeath_predictions)
lin_rmse = np.sqrt(lin_mse)
print('RMSE, test data',lin_rmse)

RDdeath_predictions = op_rf.predict(X_covid_test)
lin_mse = mean_squared_error(y_covid_test,RDdeath_predictions)
lin_rmse = np.sqrt(lin_mse)
print('RMSE, covid test data',lin_rmse)

RDdeath_predictions = op_rf.predict(X)
lin_mse = mean_squared_error(y,RDdeath_predictions)
lin_rmse = np.sqrt(lin_mse)
print('RMSE, full dataset',lin_rmse)

#just doublechecking r2 using alt method
r2_score(y,RDdeath_predictions)

# create visual, plotting predicted and actual deaths from respiratory disease

plt.rcParams["figure.figsize"] = (30,20)
plt.rcParams.update({'font.size': 20})
plt.plot(y, 'ro', RDdeath_predictions, 'bo')
plt.ylabel('RD Weekly Deaths')
plt.xlabel('Weeks from March 2016 to April 2020')
plt.legend(['Target', 'Prediction'])
plt.savefig('fake_outputs/actual_predicted.png')

#commented out as currently below visuals only work in Notebook not in running standard python script

#create new dataframe with predictions to create visual using dates 
#d = pd.DataFrame(RDdeath_predictions)
#d["RD_deaths"] = y
#d["date_deaths"] = df["date"]
#d = d.rename(columns={0:"Predicted_RD_deaths"})
#d = d.sort_values(by=['date_deaths'])
#d

#create visual showing lineplot of predictions
#plt.rcParams["figure.figsize"] = (30,20)
#sns.lineplot(data=d, x="date_deaths", y="Predicted_RD_deaths")

#create visual showing lineplot both predictions and target

#import matplotlib.patches as mpatches
#red_patch = mpatches.Patch(color='orange', label='RD Deaths')
#blue_patch = mpatches.Patch(color='blue', label='Predicted RD Deaths')

#plt.rcParams["figure.figsize"] = (50,20)
#sns.lineplot(data=d, x="date_deaths", y="Predicted_RD_deaths")
#sns.lineplot(data=d, x="date_deaths", y="RD_deaths")
#plt.ylabel("Respiratory Deaths", labelpad=14)
#plt.xlabel("Date", labelpad=14)

#plt.legend(handles=[red_patch, blue_patch]);

#feature importance on model "op_rf" inbuilt to scikit-learn python library for random forest - run on training data
print('feature importance scores', op_rf.feature_importances_)

#visual feature importance
importances = op_rf.feature_importances_
indices = np.argsort(importances)
features = X_train.columns
plt.rcParams.update({'font.size': 14})
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig('fake_outputs/feature_importance.png')

#import permutation importance variable importance tool - run on training data

perm_importance = permutation_importance(op_rf, X_train, y_train)

#visual for permutation importance
sorted_idx = perm_importance.importances_mean.argsort()
print(sorted_idx)
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.savefig('fake_outputs/permutation_importance_1.png')

#run again to see if differences between different instance of op_rf
perm_importance = permutation_importance(op_rf, X_train, y_train)

sorted_idx = perm_importance.importances_mean.argsort()
print(sorted_idx)
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.savefig('fake_outputs/permutation_importance_2.png')

#SHAP very computationally expensive - run on sample of 10 (check its working), 100, 1000 (compare to see if major changes between two)
X_train_shap = shap.sample(X_train, 10)
explainer = shap.KernelExplainer(op_rf.predict, X_train_shap)
shap_values = explainer.shap_values(X_train_shap)
shap.summary_plot(shap_values, X_train_shap)
plt.savefig('fake_outputs/shap_10.png')

# X_train_shap = shap.sample(X_train, 100)
# explainer = shap.KernelExplainer(op_rf.predict, X_train_shap)
# shap_values = explainer.shap_values(X_train_shap)
# shap.summary_plot(shap_values, X_train_shap)
# plt.savefig('fake_outputs/shap_100.png')

# X_train_shap = shap.sample(X_train, 1000)
# explainer = shap.KernelExplainer(op_rf.predict, X_train_shap)
# shap_values = explainer.shap_values(X_train_shap)
# shap.summary_plot(shap_values, X_train_shap)
# plt.savefig('fake_outputs/shap_1000.png')

