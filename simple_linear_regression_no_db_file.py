
import numpy as np
from sklearn.linear_model import LinearRegression
 

X= np.array([1,2,3,4,5,6,7,8,9,10]).reshape((-1, 1))
y = np.array([2,4,5,6,8,10,11,12,14,16])

#model = LinearRegression().fit(X, y.reshape((-1, 1)))
model = LinearRegression().fit(X, y)

r_sq = model.score(X, y)
print(f"Coefficient of determination: {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_}")

y_pred = model.predict(X)
print(f"Predicted response: \n{y_pred}")
print("")
print("==========================")
print("")
# That can be doing like:
y_pred = model.intercept_ + model.coef_ * X
print(f"Predicted response: \n{y_pred}")

