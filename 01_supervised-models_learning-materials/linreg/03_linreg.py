# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Import Dataframe
bmi_life_data = pd.read_csv("lib/03_linreg_data.csv") # Rename in browser
X = bmi_life_data[['BMI']]
y = bmi_life_data[['Life expectancy']]

# Make and fit the linear regression model
bmi_life_model = LinearRegression()
bmi_life_model.fit(X, y)

# Make a prediction using the model
laos_life_exp = bmi_life_model.predict(21.07931)
