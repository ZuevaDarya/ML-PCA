import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('task_01_14_25.csv',header=None,  delimiter=',', decimal='.')
X_features = data.values

pca = PCA(svd_solver='full')

#проецирование объектов на ГК
projected_coordinates = pca.fit_transform(X_features)

coordinate_on_first_gc = projected_coordinates[0][0]
coordinate_on_second_gc = projected_coordinates[0][1]
print("Координата первого объекта - 1я ГК: ", round(coordinate_on_first_gc, 3))
print("Координата первого объекта - 2я ГК: ", round(coordinate_on_second_gc, 3))

#explained_variance_ratio_ - доля объясненной дисперсии
variance = pca.explained_variance_ratio_
dispersion = [1 - x for x in variance]

percent_dispersion_on_two_gc = variance[0] + variance[1]

print("Доля объясненной дисперсии при первых 2х ГК: ", round(percent_dispersion_on_two_gc, 3))
print("Дисперсия: ", dispersion)

plt.scatter(projected_coordinates[:, 0:1], projected_coordinates[:, 1:2])
plt.show()

