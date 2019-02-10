import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = "~/KaggleCompetitions/HousingPrices/train.csv"

home_data = pd.read_csv(iowa_file_path)


new_home_data = home_data.copy()
new_home_data.drop("SalePrice", axis=1, inplace=True)
missingValueCountByColumn = (new_home_data.isnull().sum())
missingColumn = missingValueCountByColumn > 0

# path to file you will use for predictions
test_data_path = "~/KaggleCompetitions/HousingPrices/test.csv"

# read test data file using pandas
test_data = pd.read_csv(test_data_path)


new_test_data = test_data.copy()
test_missingValueCountByColumn = (new_test_data.isnull().sum())
test_missingColumn = test_missingValueCountByColumn > 0

new_data_string = new_home_data.select_dtypes('object')
new_data_not_string = new_home_data.select_dtypes(
    exclude='object')


median_imputer = SimpleImputer(strategy='median')
missing_imputer = SimpleImputer(strategy='constant', fill_value='missing')

new_data_string_imputed = pd.DataFrame(missing_imputer.fit_transform(new_data_string))
new_data_string_imputed.columns = new_data_string.columns
new_data_string_imputed_encoded = pd.get_dummies(new_data_string_imputed)
new_data_not_string_imputed = pd.DataFrame(median_imputer.fit_transform(new_data_not_string))
new_data_not_string_imputed.columns = new_data_not_string.columns

new_home_data = pd.concat([new_data_string_imputed_encoded, new_data_not_string_imputed], axis=1)


test_data_string = new_test_data.select_dtypes('object')
test_data_not_string = new_test_data.select_dtypes(exclude='object')

for col in test_data_not_string.columns:
    if missingColumn[col] or test_missingColumn[col]:
        new_home_data[col+"_was_missing"] = new_home_data[col].isnull()


test_data_string_imputed = pd.DataFrame(missing_imputer.transform(test_data_string))
test_data_string_imputed.columns = test_data_string.columns
test_data_string_imputed_encoded = pd.get_dummies(test_data_string_imputed)
test_data_not_string_imputed = pd.DataFrame(median_imputer.transform(test_data_not_string))
test_data_not_string_imputed.columns = test_data_not_string.columns
new_test_data = pd.concat([test_data_string_imputed_encoded, test_data_not_string_imputed], axis=1)

for col in test_data_not_string.columns:
    if test_missingColumn[col] or missingColumn[col]:
        new_test_data[col+"_was_missing"] = new_test_data[col].isnull()

final_train, final_test = new_home_data.align(new_test_data,
                                              join='inner',
                                              axis=1)


y = home_data.SalePrice
# Create X
features = final_test.columns
X = final_train[features]

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
nouse_X, valid_X, nouse_y, valid_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

best_n = 0
best_rate = 0
best_mse = 1000000
for a, b in np.ndindex((900, 10)):

    my_model = XGBRegressor(n_estimators=100+a, learning_rate=0.01*b, n_jobs=4)
    my_model.fit(train_X, train_y, early_stopping_rounds=5,
                 eval_set=[(test_X, test_y)], verbose=False)
    mse = mean_squared_error(my_model.predict(valid_X), valid_y)
    if mse < best_mse:
        best_n = 100 + a
        best_rate = 0.01*b
        best_mse = mse


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = XGBRegressor(n_estimators=best_n, learning_rate=best_rate, n_jobs=4)

# fit rf_model_on_full_data on all data from the
rf_model_on_full_data.fit(X, y, verbose=False)


# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_XX = final_test[features]

# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(test_XX)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
