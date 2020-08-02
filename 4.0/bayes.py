import datasets
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(datasets.X_TRAIN, datasets.Y_TRAIN)

pred = clf.predict(datasets.X_TEST)

print("Prediction:\n", pred, end="\n\n")
print("Test:\n", datasets.Y_TEST, end="\n\n")
