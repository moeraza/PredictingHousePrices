from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read Data
data = pd.read_csv('train.csv')
X = np.array(data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object']))
y = data.SalePrice
#split up data as a means to use some for validation
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.25)

#Make a pipline for cleaner code and easier deployability
my_pipeline = make_pipeline(Imputer(), XGBRegressor())
my_pipeline.fit(train_X, train_y, xgbregressor__early_stopping_rounds = 20,
        xgbregressor__eval_set= [(val_X, val_y )],
        xgbregressor__verbose= False)

#Use cross validation to assess model since we have a "small" amount of data
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error', cv=4)
print(cv_score)
print('Mean Absolute Error %2f' %(-1 * cv_score.mean()))

#Before creating submission file, verify that MAE is good enough
response = input("Would you like to make submission file? y/n ")
if response == "y":
    test_data = pd.read_csv('test.csv')
    test_X =np.array(test_data.select_dtypes(exclude=['object']))
    predicted_prices = my_pipeline.predict(test_X)

    my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice':predicted_prices})
    my_submission.to_csv('submission.csv', index=False)
else:
    print("Terminate")
