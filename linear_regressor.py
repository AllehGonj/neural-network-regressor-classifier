import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

reg = linear_model.LogisticRegression(C=1e5, max_iter=1000)
reg.fit(datasets.X_TRAIN, datasets.Y_TRAIN)

pred = reg.predict(datasets.X_TEST)

print("Prediction:\n", pred, end="\n\n")
print("Test:\n", datasets.Y_TEST, end="\n\n")