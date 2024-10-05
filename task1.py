import pandas as pd
from sklearn import linear_model
import math

# Reading Dataset From CSV File
df_train = pd.read_csv('train_cleaned.csv')

# Defining Independent and Dependent Variables
X = df_train[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr']]
y = df_train['SalePrice']

# Creating Linear Regression Object
reg = linear_model.LinearRegression()
reg.fit(X, y)

# Predicts sales price and stores in a file called PredictedPrice.csv
df_testing = pd.read_csv('test_cleaned.csv')
input_X = df_testing[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr']]
sales_price_predicted = []

for i in range(1459):
    np = reg.predict(input_X)[i]
    new_price = math.ceil(np)
    sales_price_predicted.append(new_price)

df_testing["SalesPricePredicated"] = sales_price_predicted
df_testing.to_csv("PredictedPrice.csv")
