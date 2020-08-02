import datasets
from sklearn.naive_bayes import GaussianNB

cb = GaussianNB()
cb.fit(datasets.X_TRAIN, datasets.Y_TRAIN)

pred = cb.predict(datasets.X_TEST)

print("Prediction:\n", pred, end="\n\n")
print("Test:\n", datasets.Y_TEST, end="\n\n")