import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("notebooks\\salma-wahwah\\train\\features.csv")

# Feature and label separation
X = np.array(data.drop(columns=['label']))
y = np.array(data['label'])

# Encode the labels as integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best Parameters from GridSearchCV: {best_params}")

# Train the model with the best parameters
best_rf_model = grid_search.best_estimator_

# Make predictions
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# K-Fold Cross-Validation
print("\nPerforming K-Fold Cross-Validation...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and print accuracy for each fold
cv_scores = []
for train_index, val_index in kf.split(X, y_encoded):
    X_train_cv, X_val_cv = X[train_index], X[val_index]
    y_train_cv, y_val_cv = y_encoded[train_index], y_encoded[val_index]
    
    best_rf_model.fit(X_train_cv, y_train_cv)
    y_val_pred = best_rf_model.predict(X_val_cv)
    fold_accuracy = accuracy_score(y_val_cv, y_val_pred)
    cv_scores.append(fold_accuracy)
    print(f"Fold Accuracy: {fold_accuracy:.4f}")

# Calculate the average cross-validation accuracy
print(f"\nAverage K-Fold Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

# Save the model
joblib.dump(best_rf_model, 'best_rf_model.pkl')
print("Model saved successfully.")
