import datasets
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor(n_neighbors=1)
reg.fit(datasets.X_TRAIN, datasets.Y_TRAIN)
reg.score(datasets.X_TRAIN, datasets.Y_TRAIN)

pred = reg.predict(datasets.X_TEST)

print("Prediction:\n", pred, end="\n\n")
print("Test:\n", datasets.Y_TEST, end="\n\n")
