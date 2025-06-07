import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv('breast-cancer.csv')

# drop useless columns
df = df.drop(['id'], axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# convert diagnosis to binary
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# features and labels
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with linear kernel
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train_scaled, y_train)
print("Linear SVM")
print(classification_report(y_test, svm_linear.predict(X_test_scaled)))

# SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
print("RBF SVM")
print(classification_report(y_test, svm_rbf.predict(X_test_scaled)))

# visualize decision boundary (only using 2 features)
features = ['radius_mean', 'texture_mean']
X_vis = df[features]
y_vis = df['diagnosis']

# scale selected 2D features
X_vis_scaled = scaler.fit_transform(X_vis)

# fit RBF SVM for plotting
svm_vis = SVC(kernel='rbf', C=1, gamma='scale')
svm_vis.fit(X_vis_scaled, y_vis)

# mesh for plotting decision boundary
h = .02
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# plot decision boundary
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=y_vis, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('SVM RBF Decision Boundary')
plt.show()

# hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=5, verbose=0)
grid.fit(X_train_scaled, y_train)

# best paramtrs
print("Best Params:", grid.best_params_)

# use best model on test set
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
print("Best RBF SVM after tuning")
print(classification_report(y_test, y_pred))

# cross-validation on best model
scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
print("CV Scores:", scores)
print("Mean CV Score:", scores.mean())
