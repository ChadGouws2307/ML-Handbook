# Regression

# @ Author: Chad Gouws
# Date: 03/05/2019

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read Data
data = pd.read_csv('Data/Linear Regression/weight-height.csv', sep=',')

# Create needed lists
gender = list(data['Gender'])
height = list(data['Height'])
weight = list(data['Weight'])

# BMI Calculation
np_height = 2.54*np.array(data['Height'])
np_weight = 0.453582*np.array(data['Weight'])
bmi = np_weight / (np_height/100)**2

data['BMI'] = bmi

# Set Male=1 and Female=0
sex = []

for person in gender:
    if person == 'Male':
        sex.append(1)

    else:
        sex.append(0)

data['Sex'] = sex


# Split male and female data
male_height = []
male_weight = []
female_height = []
female_weight = []

for i in range(len(height)):
    if sex[i] == 1:
        male_height.append(np_height[i])
        male_weight.append(np_weight[i])

    else:
        female_height.append(np_height[i])
        female_weight.append(np_weight[i])


# Descriptive Statistics
# Median
median_height = np.median(np_height)
median_weight = np.median(np_weight)
f_median_height = np.median(female_height)
f_median_weight = np.median(female_weight)
m_median_height = np.median(male_height)
m_median_weight = np.median(male_weight)

# Mean
mean_height = np.mean(np_height)
mean_weight = np.mean(np_weight)
f_mean_height = np.mean(female_height)
f_mean_weight = np.mean(female_weight)
m_mean_height = np.mean(male_height)
m_mean_weight = np.mean(male_weight)

# Standard Deviation
std_dev_height = np.std(np_height)
std_dev_weight = np.std(np_weight)
f_std_dev_height = np.std(female_height)
f_std_dev_weight = np.std(female_weight)
m_std_dev_height = np.std(male_height)
m_std_dev_weight = np.std(male_weight)

# Skewness
height_skew = skew(np_height)
weight_skew = skew(np_weight)

male_height_skew = skew(male_height)
female_height_skew = skew(female_height)

male_weight_skew = skew(male_weight)
female_weight_skew = skew(female_weight)

# Kurtosis
height_kurtosis = kurtosis(np_height)
weight_kurtosis = kurtosis(np_weight)

male_height_kurtosis = kurtosis(male_height)
male_weight_kurtosis = kurtosis(male_weight)

female_height_kurtosis = kurtosis(female_height)
female_weight_kurtosis = kurtosis(female_weight)

# List all descriptive stats
overall_stats_height = [mean_height, median_height, std_dev_height, height_skew, height_kurtosis]
overall_stats_weight = [mean_weight, median_weight, std_dev_weight, weight_skew, weight_kurtosis]

male_height_description = [m_mean_height, m_median_height, m_std_dev_height, male_height_skew, male_height_kurtosis]
male_weight_description = [m_mean_weight, m_median_weight, m_std_dev_weight, male_weight_skew, male_weight_kurtosis]

female_height_description = [f_mean_height, f_median_height, f_std_dev_height, female_height_skew, female_height_kurtosis]
female_weight_description = [f_mean_weight, f_median_weight, f_std_dev_weight, female_weight_skew, female_weight_kurtosis]

height_df = pd.DataFrame([female_height_description, overall_stats_height, male_height_description],
                         columns=['MEAN', 'MEDIAN', 'STANDARD DEV', 'SKEW', 'KURTOSIS'])
weight_df = pd.DataFrame([female_weight_description, overall_stats_weight, male_weight_description],
                         columns=['MEAN', 'MEDIAN', 'STANDARD DEV', 'SKEW', 'KURTOSIS'])

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
print(height_df.round(3))
print(weight_df.round(3))

height_df.to_excel('Data/Linear Regression/height_description.xlsx')
weight_df.to_excel('Data/Linear Regression/weight_description.xlsx')


# Plots
# Histograms
plt.hist(np_weight, bins=50)
plt.xlabel('Weight [kg]')
plt.ylabel('Samples')
plt.title('Histogram of Weight')
plt.show()

plt.hist(np_height, bins=50)
plt.xlabel('Height [cm]')
plt.ylabel('Samples')
plt.title('Histogram of Height')
plt.show()

plt.hist(male_weight, bins=40)
plt.xlabel('Weight [kg]')
plt.ylabel('Samples')
plt.title('Male Histogram of Weight')
plt.show()

plt.hist(male_height, bins=40)
plt.xlabel('Height [cm]')
plt.ylabel('Samples')
plt.title('Male Histogram of Height')
plt.show()

plt.hist(female_weight, bins=40)
plt.xlabel('Weight [kg]')
plt.ylabel('Samples')
plt.title('Female Histogram of Weight')
plt.show()

plt.hist(female_height, bins=40)
plt.xlabel('Height [cm]')
plt.ylabel('Samples')
plt.title('Female Histogram of Height')
plt.show()

# Scatter
plt.scatter(np_height, np_weight, marker='x', c='r')
plt.title('Scatter Plot of Height and Weight')
plt.xlabel('Height [cm]')
plt.ylabel('Weight [kg]')
plt.show()


# Linear Regression

# Convert data to SI Units
np_height = 2.54*np.array(data['Height']).reshape(-1, 1)
np_weight = 0.453582*np.array(data['Weight']).reshape(-1, 1)

# Split Data in Training and Testing sets
x_train, x_test, y_train, y_test = train_test_split(np_height, np_weight, test_size=0.2, random_state=0)
data = pd.DataFrame(x_train, y_train)

regressor = LinearRegression()
regressor.fit(x_train, y_train, )                 # Train Linear Model

a = regressor.coef_
b = regressor.intercept_

print('\nLinear Equation: y = ' + str(a[0]) + '*x ' + str(b[0]) + '\n')

y_pred = regressor.predict(x_test)
train_pred = regressor.predict(x_train)
rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
rmse_train = np.sqrt(metrics.mean_squared_error(y_train, train_pred))

print('Test Root Mean Squared Error: ', rmse_test)
print('Train Root Mean Squared Error:', rmse_train)
print('\nTrain and Test Performance Difference: ' + str(round(100*(rmse_train-rmse_test)/rmse_train, 4)) + ' %\n')

# View Predicted vs Actual Data
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df.head(30))

# Plot Model and Data
x_linear = np.array([137, 202]).reshape(2)
y_linear = np.array(a*x_linear + b).reshape(2)

train = plt.scatter(x_train, y_train, marker='x', c='g')
test = plt.scatter(x_test, y_test, marker='x', c='r')
plt.plot(x_linear, y_linear, c='k')

plt.xlabel('Height [cm]')
plt.ylabel('Weight [kg]')
plt.title('Linear Regression of Height and Weight')
plt.legend((train, test), ('Train', 'Test'))

plt.show()
