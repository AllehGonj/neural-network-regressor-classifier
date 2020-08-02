import datasets
from sklearn.neighbors import KNeighborsClassifier

clsf = KNeighborsClassifier(n_neighbors=1)
clsf.fit(datasets.X_TRAIN, datasets.Y_TRAIN)
clsf.score(datasets.X_TRAIN, datasets.Y_TRAIN)

pred = clsf.predict(datasets.X_TEST)

print("Prediction:\n", pred, end="\n\n")
print("Test:\n", datasets.Y_TEST, end="\n\n")
