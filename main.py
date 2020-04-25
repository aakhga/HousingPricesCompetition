import warnings

# Importing all necessary libraries here
import pandas as pd
import numpy as np
import eli5
from scipy.stats import skew
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce

 # Charts
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# Import model libraries
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Calculate mean
from sklearn.metrics import mean_absolute_error

# Disabling warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Import the data
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Get an overview of the data
print("Train size:", train_data.shape)
print("Test size:", test_data.shape)

print('\n-----------START data processing----------')
# Delete outliers
train_data = train_data[train_data.GrLivArea < 4500]
train_data.reset_index(drop=True, inplace=True)

y = train_data.SalePrice.reset_index(drop=True)
train_data.drop(['SalePrice', 'Utilities', 'Street', 'PoolQC', 'Fence',
             'Alley'], axis=1, inplace=True)

all_data = pd.concat([train_data, test_data]).reset_index(drop=True)
print("Train size + Test size =", all_data.shape)

# Some of the non-numeric predictors are stored as numbers; we convert them into strings 
# all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
# all_data['YrSold'] = all_data['YrSold'].astype(str)
# all_data['MoSold'] = all_data['MoSold'].astype(str)

# Fill NaN in some features
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")
all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")
all_data['Exterior1st'] = all_data['Exterior1st'].fillna("VinylSd")
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna("VinylSd")
all_data['SaleType'] = all_data['SaleType'].fillna("WD")

# Simplify features
# all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
# all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
# all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
# all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
# all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

train_data = all_data.iloc[:len(y), :]

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_data, y,
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Select categorical columns
categorical_cols = [cname for cname in train_data.columns if
                    train_data[cname].dtype == "object"]

numerical_cols = ['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF', '2ndFlrSF',
                  'BsmtFinSF1', 'OverallCond', 'LotArea', 'GarageCars', 'YearRemodAdd',
                  '1stFlrSF', 'GarageArea', 'Fireplaces', 'BsmtFullBath', 'TotRmsAbvGrd',
                  'PoolArea', 'BsmtFinSF2', 'MiscVal', 'ScreenPorch']

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = test_data[my_cols].copy()
num_attributes = train_data.select_dtypes(exclude='object').copy()

fig = plt.figure(figsize=(13,18))
for i in range(len(num_attributes.columns)):
    fig.add_subplot(10,5,i+1)
    sns.distplot(num_attributes.iloc[:,i].dropna())
    plt.xlabel(num_attributes.columns[i])

plt.tight_layout()
plt.show()

# SETUP MODELS
print('-----------SETUP models----------')
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

def get_score(leafsize):
    # Define model   
    xgboost = XGBRegressor(learning_rate=0.01, n_estimators=leafsize,
                         max_depth=3, min_child_weight=0,
                         gamma=0, subsample=0.7,
                         colsample_bytree=0.7,
                         objective='reg:linear', nthread=-1,
                         scale_pos_weight=1, seed=27,
                         reg_alpha=0.00006)
    
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', xgboost)])
    
    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)
    
    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)
    
    mae = mean_absolute_error(y_valid, preds)
    print("For max leaf nodes: ", leafsize, " MeanAbsoluteError is: ", mae)
    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                                  cv=8, scoring='neg_mean_absolute_error')
    print("Average MAE cross-val-score (across experiments):\n", scores.mean())
    mae = mean_absolute_error(y_valid, preds)
    print("Average MAE score:\n", mae)

candidate_max_leaf_nodes = [5500]
results = {i: get_score(i) for i in candidate_max_leaf_nodes}

# PERMUTATION IMPORTANCE
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
y = train_data.SalePrice
train_data = train_data.drop(['SalePrice'], axis=1)
base_features = [i for i in train_data.columns if train_data[i].dtype in [np.int64]]
X = train_data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = XGBRegressor(learning_rate=0.01, n_estimators=5500,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006).fit(train_X, train_y)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = base_features)

# USE TEST DATA TO PREDICT MODEL
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

def get_score(leafsize):
    # Define model   
    xgboost = XGBRegressor(learning_rate=0.01, n_estimators=leafsize,
                         max_depth=3, min_child_weight=0,
                         gamma=0, subsample=0.7,
                         colsample_bytree=0.7,
                         objective='reg:linear', nthread=-1,
                         scale_pos_weight=1, seed=27,
                         reg_alpha=0.00006)
    
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', xgboost)])
    
    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)
    
    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)
    
    mae = mean_absolute_error(y_valid, preds)
    print("For max leaf nodes: ", leafsize, " MeanAbsoluteError is: ", mae)
    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                                  cv=8, scoring='neg_mean_absolute_error')
    print("Average MAE cross-val-score (across experiments):\n", scores.mean())
    mae = mean_absolute_error(y_valid, preds)
    print("Average MAE score:\n", mae)

candidate_max_leaf_nodes = [5500]
results = {i: get_score(i) for i in candidate_max_leaf_nodes}

train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
y = train_data.SalePrice
train_data = train_data.drop(['SalePrice'], axis=1)
base_features = [i for i in train_data.columns if train_data[i].dtype in [np.int64]]
X = train_data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = XGBRegressor(learning_rate=0.01, n_estimators=5500,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006).fit(train_X, train_y)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = base_features)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = XGBRegressor(learning_rate=0.01, n_estimators=5500,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# GENERATE TEST PREDICTIONS
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
