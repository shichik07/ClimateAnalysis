# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:32:40 2024

@author: juliu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, make_scorer
import pickle
import joblib

import os


#change directory
path = r"D:\Data\Dropbox\LifeAfter\Datascientest\Climate"
os.chdir(path)

# load data
df_default = pd.read_csv("Data/data_clean/merge_loc.csv")

# drop unnamed index 
df_default = df_default.drop("Unnamed: 0", axis = 1)

# okay so we have a couple of years for a couple of countries missing. For now we ignore that
a = df_default.groupby(["country"]).year.count()


## Implement preprocessing pipeline
# creat a class that can be used to calculate cumulative CO2 count (should be applied in order)
class CO2_cum(TransformerMixin, BaseEstimator):
    def __init__(self, CO2_col_name, country_col_name, year_col_name):
        self.CO2_col_name = CO2_col_name
        self.country_col_name = country_col_name
        self.year_col_name = year_col_name
        
    def fit(self, X, y=None):
        return(self)
    
    def transform(self, X):
        # groupby country and sum up by year
        X['co2_cum'] = X.sort_values(by = self.year_col_name).groupby([self.country_col_name])[self.CO2_col_name].cumsum()
        # now calculate the cumulative total CO2 by year
        yearly_total = X.groupby(self.year_col_name)['co2_cum'].sum().to_frame().reset_index() 
        X = pd.merge(X, yearly_total, left_on = self.year_col_name, right_on= self.year_col_name)
        # now calculate laging CO2 total,a s the effects are likely delayed, use the last co2 value for the five missing values
        #yearly_total_lag5 = yearly_total.shift(5)
        #X = pd.merge(X, yearly_total_lag5, left_on = self.year_col_name, right_on= self.year_col_name)
        return(X)

# get a transformer to scale degrees to radians    
class radian_scaler(TransformerMixin, BaseEstimator):
    def __init__(self, name_longitude, name_latitude):
        self.name_longitude = name_longitude
        self.name_latitude = name_latitude
        
    def fit(self, X, y=None):
        return(self)
    
    def transform(self, X):
        # groupby country and sum up by year
        X[self.name_longitude] =  (X[self.name_longitude]*np.pi)/180
        X[self.name_latitude] =  (X[self.name_latitude]*np.pi)/180
        return(X)  

# CO_2 transformer
co2_transform = CO2_cum(CO2_col_name = "co2", country_col_name = "country", year_col_name = "year")
# degree transformer
rad_scale = radian_scaler(name_longitude = "longitude", name_latitude = "latitude")
# standardize CO_2 cumulative measures
co2cum_scale = MinMaxScaler()
# standardize CO_2 cumulative totl measures
co2cum_t_scale = MinMaxScaler()
# standardize CO_2 measures
co2_scale = MinMaxScaler()
# standardize population measures
pop_scale = MinMaxScaler()
# standardize gdp measures
gdp_scale = MinMaxScaler()
# encode continents using one hot encoder
cont_ohe = OrdinalEncoder()
#encode countries as labels # for now, would like tow ork with coordinates
count_le = OrdinalEncoder()
# center year measures (so we have a true zero)
year_scale = MinMaxScaler()

# put it all together in preprocessing pipelin: 1st co2cum pip

# things to be done before splitting - creating new variables
col_transform1 = make_column_transformer((co2_transform, ["co2", "country", "year"]),
                                            (rad_scale, ["longitude", "latitude"]))

# things to be done after split
col_transform2 = make_column_transformer(
                                            (co2cum_t_scale, ['co2_cum_total']),
                                             (co2cum_scale, ['co2_cum']),
                                            (co2_scale, ['co2']),
                                            (pop_scale, ['population']),
                                            (gdp_scale, ['gdp']),
                                            (cont_ohe, ['continent']),
                                             (count_le, ['country']),
                                            (year_scale, ['year']))

# create new variables
df_default[["co2", "country", "year", 'co2_cum', 'co2_cum_total',"longitude", "latitude"]] = col_transform1.fit_transform(df_default)

# save results with new variables
df_default.to_csv(path_or_buf="Data/data_clean/merge_plus_new_vars.csv")


# make the relevant variables numeric
for col in ["co2", "year", "longitude", "latitude", 'co2_cum', 'co2_cum_total']:
    df_default[col] = pd.to_numeric(df_default[col])

# split into test- and training set
target = df_default.temp_anomaly
features = df_default.drop(['temp_anomaly'], axis = 1)#, 'co2_cum',"longitude", "latitude"], axis = 1) # we exclude country because we have latitude and longitude instead
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state = 45)

# save variables for plotting
Test_year = X_test.year
Test_country = X_test.country

# scale variables
X_train[['co2_cum_total','co2_cum', 'co2', 'population', 'gdp','continent', 'country', 'year']] = col_transform2.fit_transform(X_train)
X_test[['co2_cum_total','co2_cum', 'co2', 'population', 'gdp','continent', 'country', 'year']] = col_transform2.transform(X_test)


# load winning model
rfor_grid= joblib.load("rfor_grid")
#best params 
rfp = rfor_grid.best_params_

# get predictions winning model - we refit it here, because the data have been transformed 
# the grid model was trained on untransformed cumulative CO2 which is transformed now - has to be retrained to be fair

win_tree = RandomForestRegressor(n_estimators = rfp['n_estimators'], 
                                 min_samples_split = rfp['min_samples_split'], 
                                 max_depth = rfp['max_depth'], random_state= 42)
# define scores - does not work at all
scores = {'r2':make_scorer(r2_score), 
          'RMSE': make_scorer(root_mean_squared_error),
          'MAE': make_scorer(mean_absolute_error)}
cv_rep_r2 = cross_val_score(win_tree, X_train, y_train, scoring='r2', n_jobs= -1, cv = 5)
cv_rep_RMSE = cross_val_score(win_tree, X_train, y_train, scoring='neg_root_mean_squared_error', n_jobs= -1, cv = 5)
cv_rep_MAE = cross_val_score(win_tree, X_train, y_train, scoring='neg_mean_absolute_error', n_jobs= -1, cv = 5)
cv_rep_r2

win_tree.fit(X_train, y_train)
win_tree.score(X_test, y_test)
preds_rf = win_tree.predict(X_test)


# calculate performance metrics test set

r2_rf = r2_score(y_test, preds_rf)
rmse_rf = root_mean_squared_error(y_test, preds_rf)
mae_rf = mean_absolute_error(y_test, preds_rf)

r2_rf_train = np.mean(cv_rep_r2)
rmse_rf_train = -np.mean(cv_rep_RMSE)
mae_rf_train = -np.mean(cv_rep_MAE)


# put them in a dataframe
perf_df = pd.DataFrame({'Value': [r2_rf, rmse_rf, mae_rf, r2_rf_train, rmse_rf_train, mae_rf_train], 
                        'Metric': ["R2", "RMSE", "MAE", "R2", "RMSE", "MAE"], 
                        'Data': ["Test", "Test", "Test", "Train", "Train", "Train"]})



# Plot Model performance
plt.figure(figsize = (10,6))

plt.subplot(131)
pm1 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "R2"], x = "Data", y = "Value")
pm1.bar_label(pm1.containers[0],fontsize=10);
plt.title("R2 Score",)
plt.subplot(132)
pm2 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "RMSE"], x = "Data", y = "Value")
pm2.bar_label(pm2.containers[0], fontsize=10);
plt.title("RMSE Score")
plt.subplot(133)
pm3 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "MAE"], x = "Data", y = "Value")
pm3.bar_label(pm3.containers[0], fontsize=10);
plt.title("MAE Score")


# Plot Residuals
plt.figure(figsize = (10,8))
plt.subplot(211)
plt.title("Prediction Overall")
ax1 = sns.lineplot(x = Test_year, y = y_test, label = "Data")
sns.lineplot(x = Test_year, y = preds_rf, ax = ax1, label = "Prediction", color = "green")
ax1.legend(fontsize = 12)
plt.subplot(223)
b = sns.scatterplot(x = y_test, y = preds_rf)
plt.title("Residuals - RandomForrest")
plt.subplot(224)
sns.scatterplot(x = Test_year, y = y_test-preds_rf)
plt.title("Residuals by year - RandomForrest")




# Plot feature importance for random forrest
f_dict = {"Importance": win_tree.feature_importances_, 'Feature': win_tree.feature_names_in_, 'Model': "RandomForest"}
#f_dict2 = {"Importance": win_boost.feature_importances_, 'Feature': win_boost.feature_names_in_, 'Model': "GradientBoost"}

plt.figure(figsize = (6,6))
feat_importances =  pd.DataFrame(data=f_dict)
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
sns.barplot(data = feat_importances, y = 'Feature', x = 'Importance');
