# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

### Loading the Dataset
dataset = pd.read_csv("msleep.csv")

# Handling missing values using SimpleImputer
subset = dataset.iloc[:, 6:11].values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(subset)
subset = imputer.transform(subset)
dataset.iloc[:, 6:11] = subset

### Training the prediction model and making sleep duration predictions

# Selecting independent and dependent variables
X = dataset[["bodywt", "brainwt", "awake", "sleep_rem"]]  # Independent Variables
y = dataset["sleep_total"]  # Dependent Variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Adding predictions to a DataFrame for comparison
test_results = X_test.copy()
test_results['actual_sleep_total'] = y_test
test_results['predicted_sleep_total'] = y_pred

# Calculating the prediction error
test_results['error'] = abs(test_results['actual_sleep_total'] - test_results['predicted_sleep_total'])

### Plotting the Histogram
plt.figure(figsize=(15, 8))
sns.histplot(test_results["error"], bins=15, kde=True, log_scale=(True, False))
plt.title('Prediction Error Distribution - Logarithmic Histogram and KDE')
plt.xlabel('Error Values (Log Scale)')
plt.ylabel('Frequency / Density')
plt.show()
