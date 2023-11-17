import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#матрица счетов
X_reduced = pd.read_csv('X_reduced_acc_492.csv', header=None, delimiter=';', decimal='.').values
#матрица весов
X_loadings = pd.read_csv('X_loadings_weight_492.csv', header=None, delimiter=';', decimal='.').values

#восстанавливаем ихображение 
#.dot(a, b) - скалярное произведение массивов a и b 
#.transpose() - транспонирование матрицы
img = np.dot(X_loadings, X_reduced.transpose())

plt.imshow(img.reshape(100, 100), cmap='gray')
plt.show()