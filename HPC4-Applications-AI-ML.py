from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
# Create a synthetic dataset
X, y = make_classification(n_samples=100000, n_features=20, n_informative=15,
random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Measure training time
start = time.time()
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1) # -1 means use all available CPU cores
clf.fit(X_train, y_train)
print(f"Training Time: {time.time() - start:.2f} seconds")
# Accuracy on test data
print("Test Accuracy:", clf.score(X_test, y_test))
