import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


df = pd.read_csv("Blockchain103.csv")

df["Node Uptime"] = pd.to_timedelta(df["Node Uptime"]).dt.days

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline_rf = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

param_grid_rf = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}

grid_search_rf = GridSearchCV(
    pipeline_rf, param_grid_rf, cv=5, scoring="accuracy", n_jobs=-1
)
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_

print("Best Hyperparameters for Random Forest:")
print(grid_search_rf.best_params_)

y_pred_rf = best_rf_model.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

pipeline_svm = Pipeline(
    [("scaler", StandardScaler()), ("classifier", SVC(random_state=42))]
)

param_grid_svm = {
    "classifier__C": [0.1, 1, 10],
    "classifier__kernel": ["linear", "rbf"],
    "classifier__gamma": ["scale", "auto"],
}

grid_search_svm = GridSearchCV(
    pipeline_svm, param_grid_svm, cv=5, scoring="accuracy", n_jobs=-1
)
grid_search_svm.fit(X_train, y_train)

best_svm_model = grid_search_svm.best_estimator_

print("\nBest Hyperparameters for Support Vector Machine:")
print(grid_search_svm.best_params_)

y_pred_svm = best_svm_model.predict(X_test)

print("\nSupport Vector Machine Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

pipeline_mlp = Pipeline(
    [("scaler", StandardScaler()), ("classifier", MLPClassifier(random_state=42))]
)

param_grid_mlp = {
    "classifier__hidden_layer_sizes": [(50, 50), (100,)],
    "classifier__activation": ["relu", "tanh"],
    "classifier__alpha": [0.0001, 0.001, 0.01],
}

grid_search_mlp = GridSearchCV(
    pipeline_mlp, param_grid_mlp, cv=5, scoring="accuracy", n_jobs=-1
)
grid_search_mlp.fit(X_train, y_train)

best_mlp_model = grid_search_mlp.best_estimator_

print("\nBest Hyperparameters for Multi-Layer Perceptron:")
print(grid_search_mlp.best_params_)

y_pred_mlp = best_mlp_model.predict(X_test)

print("\nMulti-Layer Perceptron Classification Report:")
print(classification_report(y_test, y_pred_mlp))
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))

accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

best_model = max(
    [
        (accuracy_rf, "Random Forest"),
        (accuracy_svm, "Support Vector Machine"),
        (accuracy_mlp, "Multi-Layer Perceptron"),
    ]
)

print(f"\nBest Model: {best_model[1]} with Accuracy: {best_model[0]}")
