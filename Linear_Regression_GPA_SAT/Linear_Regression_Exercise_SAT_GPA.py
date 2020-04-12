
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv("1.01. Simple linear regression.csv")
print(data)

print(data.describe())

# y = b0 + b1x1

y = data['GPA']
x1 = data['SAT']


x = sm.add_constant(x1)
# Contain the output of the Ordinary Least Squares(OLS) regression
results = sm.OLS(y, x).fit()
print(results.summary())

plt.scatter(x1, y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
#plt.show()

