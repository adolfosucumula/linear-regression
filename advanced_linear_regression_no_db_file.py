import numpy as np
import statsmodels.api as sm


X= np.array([[0,1],[2,3],[3,5],[4,8],[5,10],[6,12],[7,14],[8,16],[9,18],[10,20]])
y = np.array([2,4,5,6,8,10,11,12,14,16])

X = sm.add_constant(X)

# Create model and fit it
model = sm.OLS(y, X)

results = model.fit()

print(f"Coefficient of determination: {results.rsquared}")
print(f"Adjusted coefficient of determination: {results.rsquared_adj}")
print(f"Regression coefficient: {results.params}")

#Predict response
print(f"Predicted response: \n{results.fittedvalues}")
print(f"predicted response: \n{results.predict(X)}")

