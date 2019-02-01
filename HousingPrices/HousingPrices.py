import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = "~/KaggleCompetitions/HousingPrices/train.csv"

home_data = pd.read_csv(iowa_file_path)


new_home_data = home_data.copy()
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
new_data_not_string = new_home_data.select_dtypes(exclude='object')

median_imputer = SimpleImputer(strategy='median')
missing_imputer = SimpleImputer(strategy='constant', fill_value='missing')

new_data_string_imputed = pd.DataFrame(missing_imputer.fit_transform(new_data_string))
new_data_string_imputed.columns = new_data_string.columns
new_data_not_string_imputed = pd.DataFrame(median_imputer.fit_transform(new_data_not_string))
new_data_not_string_imputed.columns = new_data_not_string.columns

new_home_data = pd.concat([new_data_string_imputed, new_data_not_string_imputed], axis=1)


for col in new_test_data.columns:
    if missingColumn[col] or test_missingColumn[col]:
        new_home_data[col+"_was_missing"] = new_home_data[col].isnull()


y = new_home_data.SalePrice
# Create X
features = list(new_home_data.select_dtypes(exclude='object'))
features.remove('SalePrice')

X = new_home_data[features]


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the
rf_model_on_full_data.fit(X, y)


test_data_string = new_test_data.select_dtypes('object')
test_data_not_string = new_test_data.select_dtypes(exclude='object')


test_data_string_imputed = pd.DataFrame(missing_imputer.fit_transform(test_data_string))
test_data_string_imputed.columns = test_data_string.columns
test_data_not_string_imputed = pd.DataFrame(median_imputer.fit_transform(test_data_not_string))
test_data_not_string_imputed.columns = test_data_not_string.columns
new_test_data = pd.concat([test_data_string_imputed, test_data_not_string_imputed], axis=1)

for col in new_test_data.columns:
    if test_missingColumn[col] or missingColumn[col]:
        new_test_data[col+"_was_missing"] = new_test_data[col].isnull()
# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = new_test_data[features]

# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
