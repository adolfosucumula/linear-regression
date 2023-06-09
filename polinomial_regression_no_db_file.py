
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X= np.array([1,2,3,4,5,6,7,8,9,10]).reshape((-1, 1))
y = np.array([2,4,5,6,8,10,11,12,14,16])

# Transform input

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(X)
#Or 
X_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

#Create model

model = LinearRegression().fit(X_, y)

#get Results
r_sq = model.score(X_, y)
print(f"Coeffiecient of determination: {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
