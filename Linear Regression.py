# import pandas as pd
# import math
# import numpy as np
# from sklearn import preprocessing
# from sklearn.model_selection  import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
import numpy
from sklearn.metrics import r2_score

x = [20,25,30,35,40,45,50]
y = [0.0404,0.0487,0.0576,0.0605,0.0661,0.0782,0.0871]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

speed = mymodel(23)
print(speed)

# X = np.array([[20],[25],[30],[35],[40],[45],[50])
# Y = np.transpose(np.array([[0.0404],[0.0487],[0.0576],[0.0605],[0.0661],[0.0782],[0.0871]]))
#
# poly_reg = PolynomialFeatures(degree=4)
# x_poly = poly_reg.fit_transform(X)
# lin_reg = LinearRegression()
# lin_reg.fit(x_poly,Y)
# print(lin_reg.predict([[43]]))

# X = np.arange(6).reshape(3, 2)
# poly = PolynomialFeatures(2)
# poly.fit_transform(X)
# poly = PolynomialFeatures(interaction_only=True)
# poly.fit_transform(X)


