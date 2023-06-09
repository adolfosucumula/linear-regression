
import numpy as np
from sklearn.linear_model import LinearRegression

X= [[0,1],[2,3],[3,5],[4,8],[5,10],[6,12],[7,14],[8,16],[9,18],[10,20]]
y = [2,4,5,6,8,10,11,12,14,16]

X, y = np.array(X), np.array(y)

# create the model and fit it
model = LinearRegression().fit(X, y)

#Get results
r_sq = model.score(X, y)
print(f"Coefficient of determination: {r_sq}")
print(f"Intercept: \n{model.intercept_}")
print(f"coefficients: \n{model.coef_}")


#Predict response
y_pred = model.predict(X)
print(f"Predicted response: \n{y_pred}")
print("")
print("=============== SECOND WAY ===============")
print("")
y_pred2 = model.intercept_ + np.sum(model.coef_ * X, axis=1)
print(f"Predicted response: \n{y_pred2}")


