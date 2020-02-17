# Feature Selection

# @ Author: Chad Gouws
# Date: 22/05/2019

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Collect data
age = np.random.randint(18, 80, 5000)
type_of_car = np.random.randint(0, 30, 5000)
lattitude = np.random.randint(0, 5000, 5000)
longitude = np.random.randint(0, 5000, 5000)
sex = np.random.randint(0, 2, 5000)
place_of_work = np.random.randint(0, 150, 5000)
profession = np.random.randint(0, 50, 5000)

month_born = np.random.randint(0, 12, 5000)
place_of_birth = np.random.randint(0, 40, 5000)

# Create risk
age_risk = (0.016*(age-30)**2 - 0.7*age + 31)/100
car_risk = (0.1*type_of_car + 2)/100
work_risk = (1 + 0.04*place_of_work)/100
profession_risk = (5 - 0.001*profession)/100
lat_risk = (-0.00000015*lattitude**2+5.5)/100
long_risk = 4.5/(100+1*abs(longitude-2500))+0.02

sex_risk = []
for i in list(sex):
    if i == 0:
        sex_risk.append(0.085)

    else:
        sex_risk.append(0.07)

sex_risk = np.array(sex_risk)

person_risk = (age_risk+car_risk+lat_risk+long_risk+work_risk+profession_risk+sex_risk)/7

risk = []

for i in list(person_risk):
    if i < 0.04:
        risk.append(0)

    elif 0.04 <= i < 0.05:
        risk.append(1)

    elif 0.05 <= i < 0.06:
        risk.append(2)

    elif 0.06 <= i < 0.07:
        risk.append(3)

    else:
        risk.append(4)

df = pd.DataFrame(age, columns=['Age'])
df['Car'] = type_of_car
df['Lat.'] = lattitude
df['Long.'] = longitude
df['Work'] = place_of_work
df['Prof.'] = profession
df['Sex'] = sex
df['Month'] = month_born
df['Birth'] = place_of_birth
df['Risk'] = risk

Y = df.iloc[:, -1]
X = df.iloc[:, 0:8]

# Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(X, Y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
feature_scores.columns = ['Feature', 'Score']
print(feature_scores.nlargest(9, 'Score'))

# Feature Importance
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X, Y)

print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(9).plot(kind='barh')
plt.show()

# Heatmap Correlation Matrix
import seaborn as sb

corrmat = df.corr()
top_corr = corrmat.index
plt.figure(figsize=(20, 20))
g = sb.heatmap(df[top_corr].corr(), annot=True, cmap='autumn')
plt.show()
