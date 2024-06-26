# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:53:00 2024

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
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import joblib
import pickle

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


# create models
rid_reg = Ridge(random_state = 42)
rfor = RandomForestRegressor(n_estimators= 100, max_features= 1.0,random_state=45, min_samples_leaf= 2) # otherwise we get bagged trees
boost = GradientBoostingRegressor(random_state = 45)

# create dictionary for parameters forhyperparameter tuning linear regression
param_ridge ={ "alpha": np.logspace(0, 4, num=100)}
param_rfor = {'max_depth':  [int(x) for x in np.linspace(30, 90, num = 7)],
              'n_estimators': [300, 400],
    'min_samples_split': [int(x) for x in np.linspace(4, 10, num = 7)]}
param_boost = {'learning_rate': [0.02],
    'n_estimators': [int(x) for x in np.linspace(40, 60, num = 5)],
    'max_depth':  [int(x) for x in np.linspace(40, 60, num = 4)],
    'min_samples_split': [int(x) for x in np.linspace(25, 50, num = 6)]}

# Pipeline with preprocessing
pip_ridge = Pipeline(steps=[
    ("Preprocessing", col_transform2),
    ("Ridge", rid_reg)
    ])

pip_rfor = Pipeline(steps=[
    ("Preprocessing", col_transform2),
    ("Ridge", rfor)
    ])

# Random Search Boost
boost_random = RandomizedSearchCV(estimator = boost, param_distributions = param_boost, 
                               n_iter = 40, cv = 3, verbose=2, random_state=42, n_jobs = -1)
boost_random.fit(X_train, y_train)
boost_random.best_score_
boost_random.best_params_


# Random search for RF
rf_random = RandomizedSearchCV(estimator = rfor, param_distributions = param_rfor, 
                               n_iter = 40, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
rf_random.best_score_
rf_random.best_params_


# Perform grid search
grid_ridge = GridSearchCV(estimator = rid_reg, param_grid =param_ridge, cv=5, n_jobs= -1, verbose = True)
grid_rfor = GridSearchCV(estimator = rfor, param_grid =param_rfor, cv=5, n_jobs= -1, verbose = True)
grid_boost = GridSearchCV(estimator = boost, param_grid =param_boost, cv=5, n_jobs= -1, verbose = True)

# fit Gridsearches
grid_ridge.fit(X_train, y_train)
grid_rfor.fit(X_train, y_train)
 
# save models
joblib.dump(grid_ridge, "ridge_grid")
joblib.dump(grid_rfor, "rfor_grid")
joblib.dump(grid_boost, "boost_grid")

pickle.dump(grid_ridge, open("ridge_grid1", 'wb'))

# load models
ridge_grid = joblib.load("ridge_grid")
rfor_grid= joblib.load("rfor_grid")
boost_grid=joblib.load("boost_grid")

# best scores
ridge_grid.best_score_
rfor_grid.best_score_
boost_grid.best_score_

rcp = ridge_grid .best_params_
rfp = rfor_grid.best_params_
bp = boost_grid.best_params_

ridge_grid1 = Ridge(alpha = 1, random_state = 42)
rfor_grid1 = RandomForestRegressor(max_depth =  50, min_samples_split =  5, n_estimators = 400, random_state = 42) 
boost_grid1 = GradientBoostingRegressor(learning_rate= 0.02, max_depth = 40, min_samples_split = 30, n_estimators = 60,random_state = 42)
ridge_grid1.fit(X_train, y_train)
rfor_grid1.fit(X_train, y_train)
boost_grid1.fit(X_train, y_train)

# get the winning forest to get feature importance
win_tree = RandomForestRegressor(n_estimators = rfp['n_estimators'], 
                                 min_samples_split = rfp['min_samples_split'], 
                                 max_depth = rfp['max_depth'], random_state= 42)
win_tree.fit(X_train, y_train)
win_tree.score(X_test, y_test)


rfor_grid.best_params_
# get the winning boosted trees to get feature importance
win_boost = GradientBoostingRegressor(learning_rate=bp['learning_rate'], 
                                      n_estimators=bp['n_estimators'],
                                      max_depth=bp['max_depth'],
                                      min_samples_split=bp['min_samples_split'],
                                      random_state = 42)
win_boost.fit(X_train, y_train)
win_boost.score(X_test, y_test)



# and now use nested cross-validation 
#outer_scores ={}
#for name, gs in gridcvs.items():
#    nested_score = cross_val_score(gs, X_train_S, y_train_S, cv=5)
#    outer_scores[name] = nested_score
    #print(f'{name}: outer accuracy {100*nested_score.mean():.2f} +/- {100*nested_score.std():.2f}')

# Plotting
# Calculate Predictions on test set
preds_ridge = ridge_grid.predict(X_test)
preds_rf = rfor_grid.predict(X_test)
preds_boost = boost_grid.predict(X_test)

# calculate performance metrics test set
r2_rid = r2_score(y_test, preds_ridge)
rmse_rid = root_mean_squared_error(y_test, preds_ridge)
mae_rid = mean_absolute_error(y_test, preds_ridge)

r2_rf = r2_score(y_test, preds_rf)
rmse_rf = root_mean_squared_error(y_test, preds_rf)
mae_rf = mean_absolute_error(y_test, preds_rf)

r2_boost = r2_score(y_test, preds_boost)
rmse_boost = root_mean_squared_error(y_test, preds_boost)
mae_boost = mean_absolute_error(y_test, preds_boost)

# put them in a dataframe
perf_df = pd.DataFrame({'Value': [r2_rid, r2_rf, r2_boost, rmse_rid, rmse_rf, rmse_boost, mae_rid, mae_rf, mae_boost], 
                        'Metric': ["R2", "R2", "R2", "RMSE", "RMSE", "RMSE", "MAE", "MAE", "MAE"], 
                        'Model': ["Ridge", "Forest", "Boost", "Ridge", "Forest", "Boost", "Ridge", "Forest", "Boost"],
                        'Data': "Test"})

# Plot Model performance
plt.figure(figsize = (12,8))
plt.subplot(131)
pm1 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "R2"], x = "Model", y = "Value")
pm1.bar_label(pm1.containers[0],fontsize=10);
plt.title("R2 Score",)
plt.subplot(132)
pm2 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "RMSE"], x = "Model", y = "Value")
pm2.bar_label(pm2.containers[0], fontsize=10);
plt.title("RMSE Score")
plt.subplot(133)
pm3 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "MAE"], x = "Model", y = "Value")
pm3.bar_label(pm3.containers[0], fontsize=10);
plt.title("MAE Score")

# Plot Residuals
plt.figure(figsize = (14,12))
plt.subplot(231)
a = sns.scatterplot(x = y_test, y = preds_ridge)
plt.title("Residuals - RidgeRegression",)
plt.subplot(232)
b = sns.scatterplot(x = y_test, y = preds_rf)
plt.title("Residuals - RandomForrest")
plt.subplot(233)
sns.scatterplot(x = y_test, y = preds_boost)
plt.title("Residuals - GradientBoost")
plt.subplot(234)
sns.scatterplot(x = Test_year, y = y_test-preds_ridge)
plt.title("Residuals by year - RidgeRegression")
plt.subplot(235)
sns.scatterplot(x = Test_year, y = y_test-preds_rf)
plt.title("Residuals by year - RandomForrest")
plt.subplot(236)
sns.scatterplot(x = Test_year, y = y_test-preds_boost)
plt.title("Residuals by year - GradientBoost")

# Plot Predictions versus data
plt.figure(figsize = (14,12))
plt.title("Prediction Overall")
plt.subplot(311)
ax = sns.lineplot(x = Test_year, y = y_test)
sns.lineplot(x = Test_year, y = preds_ridge, ax = ax, color ='red')
plt.subplot(312)
ax1 = sns.lineplot(x = Test_year, y = y_test)
sns.lineplot(x = Test_year, y = preds_rf, ax = ax1, color = "green")
plt.subplot(313)
ax2 = sns.lineplot(x = Test_year, y = y_test)
sns.lineplot(x = Test_year, y = preds_boost, ax = ax2)

# plot predictions versus data on the example Germany
ind_G = Test_country == "Germany"
ind_A = Test_country == "Bolivia"

plt.figure(figsize = (14,12))
plt.title("Prediction Example Germany")
plt.subplot(311)
ax = sns.lineplot(x = Test_year[ind_G], y = y_test[ind_G], marker="o")
sns.lineplot(x = Test_year[ind_G], y = preds_ridge[ind_G], ax = ax, marker="o", color =  "red")
plt.subplot(312)
ax1 = sns.lineplot(x = Test_year[ind_G], y = y_test[ind_G], marker="o")
sns.lineplot(x = Test_year[ind_G], y = preds_rf[ind_G], ax = ax1,  marker="o", color = "green")
plt.subplot(313)
ax = sns.lineplot(x = Test_year[ind_G], y = y_test[ind_G], marker="o")
sns.lineplot(x = Test_year[ind_G], y = preds_boost[ind_G], ax = ax, marker="o")

plt.figure(figsize = (14,12))
plt.title("Prediction Example Bolivia")
plt.subplot(311)
ax = sns.lineplot(x = Test_year[ind_A], y = y_test[ind_A], marker="o")
sns.lineplot(x = Test_year[ind_A], y = preds_ridge[ind_A], ax = ax, marker="o", color =  "red")
plt.subplot(312)
ax1 = sns.lineplot(x = Test_year[ind_A], y = y_test[ind_A], marker="o")
sns.lineplot(x = Test_year[ind_A], y = preds_rf[ind_A], ax = ax1,  marker="o", color = "green")
plt.subplot(313)
ax = sns.lineplot(x = Test_year[ind_A], y = y_test[ind_A], marker="o")
sns.lineplot(x = Test_year[ind_A], y = preds_boost[ind_A], ax = ax, marker="o")

# Plot feature importance for random forrest
f_dict = {"Importance": win_tree.feature_importances_, 'Feature': win_tree.feature_names_in_, 'Model': "RandomForest"}
f_dict2 = {"Importance": win_boost.feature_importances_, 'Feature': win_boost.feature_names_in_, 'Model': "GradientBoost"}


feat_importances =  pd.concat([pd.DataFrame(data=f_dict), pd.DataFrame(data=f_dict2)])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
sns.barplot(data = feat_importances, y = 'Feature', x = 'Importance', hue = "Model");


 