##Adding in categorical data

import pandas as pd

file_path = 'train.csv'
data = pd.read_csv(file_path)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import Imputer

y = data.SalePrice

predictors = data.drop(['SalePrice'], axis =1)

##FILL IN EMPTY NUMERIC CELLS WITH AVERAGE VALUES
numeric_predictors = predictors.select_dtypes(exclude=['object'])
my_imputer=Imputer()
imputed_numeric_predictors = my_imputer.fit_transform(numeric_predictors)

##ENCODING CATERGORICAL DATA
object_of_interest=['HouseStyle','LotConfig']
object_predictors = predictors[object_of_interest]
one_hot_encoded_predictors = pd.get_dummies(object_predictors)

##COMBING NUMERICAL AND "CATEGORICAL" DATA
combined_data = pd.concat([numeric_predictors, one_hot_encoded_predictors], axis=1)
combined_data = combined_data.fillna(value=0)

train_x, val_x, train_y, val_y = tts(combined_data, y, train_size =0.9, test_size=0.1, random_state=0)

model = RandomForestRegressor()
model.fit(train_x, train_y)

 #Measuring mean_absolute_error
model_predictions = model.predict(val_x) #predicting the outcomes for validation data (data not used in model fitting)
model_mae = mae(val_y, model_predictions)
print("The mae for the model is %d " %model_mae)


response = input("Would you like to make submission file? y/n ")

if response == "y":
    #Predicting outcomes of test values
    test_data = pd.read_csv('test.csv')

    test_predictors = test_data
    test_numeric_predictors = test_predictors.select_dtypes(exclude=['object'])
    my_imputer=Imputer()
    test_imputed_numeric_predictors = my_imputer.fit_transform(test_numeric_predictors)

    test_object_of_interest=['HouseStyle','LotConfig']
    test_object_predictors = predictors[test_object_of_interest]
    test_one_hot_encoded_predictors = pd.get_dummies(test_object_predictors)

    test_combined_data = pd.concat([test_numeric_predictors, test_one_hot_encoded_predictors], axis=1)
    test_combined_data = combined_data.fillna(value=0)

    #true values for outcome are unkown, will go into submission file
    predicted_prices = model.predict(test_combined_data)
    #Making submission file
    my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice':predicted_prices})
    my_submission.to_csv('submission.csv', index=False)
else:
    print("Terminate")
