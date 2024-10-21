# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:38:36 2024

@author: Khadeeja Azizi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#STEP ONE

#Read the CSV file into a pandas dataframe
df = pd.read_csv("C:/Users/purpl/OneDrive/Documents/GitHub/Project1/Project1Data.csv")


# Print information about the dataframe, such as column names, data types, and non-null values
print(df.info())
# Print the first 5 rows of the dataframe to check if the data is loaded correctly
print(df.head())
# Print the names of all columns in the dataframe
print(df.columns)

# Extract the 'X' 'Y' and 'Z' columns from the dataframe
x = df.get("X")
print(x)
y = df.get("Y")
print(y)
z = df.get("Z")
print(z)

############################################################################################

#STEP TWO

# Statistical Analysis: Group data by 'Step' and compute basic statistics (mean, min, max, etc.)
grouped_data = df.groupby('Step').agg(['mean', 'min', 'max', 'std'])
print("Basic statistics (mean, min, max, std) for each class (Step):")
print(grouped_data)


# Plot histograms to show the distribution of X, Y, and Z for each Step

# Histogram for X coordinate
plt.figure(figsize=(11, 4))  # Set the figure size for the plots
plt.subplot(1, 3, 1)
for step in df['Step'].unique():
    plt.hist(df[df['Step'] == step]['X'], alpha=0.5, label=f'Step {step}')
plt.title('X Distribution by Step')
plt.xlabel('X')
plt.ylabel('Frequency')
plt.legend()

# Histogram for Y coordinate
plt.subplot(1, 3, 2)
for step in df['Step'].unique():
    plt.hist(df[df['Step'] == step]['Y'], alpha=0.5, label=f'Step {step}')
plt.title('Y Distribution by Step')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.legend()

# Histogram for Z coordinate
plt.subplot(1, 3, 3)
for step in df['Step'].unique():
    plt.hist(df[df['Step'] == step]['Z'], alpha=0.5, label=f'Step {step}')
plt.title('Z Distribution by Step')
plt.xlabel('Z')
plt.ylabel('Frequency')
plt.legend()

# Display the histograms
plt.tight_layout()
plt.show()


# Create a 3D scatter plot for X, Y, Z coordinates, and use color to represent the Step values
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Normalize the step values for coloring
step_values = df['Step']
norm = plt.Normalize(step_values.min(), step_values.max())  # Normalize the step values
cmap = plt.cm.viridis  # Color map for the scatter plot

# Plot the scatter points, coloring by the 'Step' column
sc = ax.scatter(df['X'], df['Y'], df['Z'], c=step_values, cmap=cmap, s=50, alpha=0.7)

# Set axis labels and title
ax.set_title('3D Scatter Plot of X, Y, Z by Step (with Color Bar)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a color bar to represent the 'Step' values
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=10)
cbar.set_label('Step')

# Show the plot
plt.show()

#########################################################################################

#STEP 3 (CORRELATION STUDY)

from sklearn.model_selection import StratifiedShuffleSplit

# StratifiedShuffleSplit based on the 'Step' column (classification label)
my_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state= 40)

# Split the data using StratifiedShuffleSplit
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)

# Now strat_df_train and strat_df_test contain the stratified split data


# Checking class distribution in the splits
print("Training Set Class Distribution:\n", strat_df_train['Step'].value_counts(normalize=True))
print("Test Set Class Distribution:\n", strat_df_test['Step'].value_counts(normalize=True))


# Select the independent and dependent variables
x_columns = ['X', 'Y', 'Z']  
y_column = 'Step'  #target variable

X_train = strat_df_train[x_columns]  # Training features
y_train = strat_df_train[y_column]  # Training target
X_test = strat_df_test[x_columns]  # Test features
y_test = strat_df_test[y_column]  # Test target

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
my_scaler = StandardScaler()

# Fit the scaler on the training set 
my_scaler.fit(X_train)

# Transform both the training and test sets
X_train_scaled = my_scaler.transform(X_train)
X_test_scaled = my_scaler.transform(X_test)

# convert the scaled arrays back to dataframes (with original column names)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)



# Calculate the Pearson correlation between features and target
correlation_matrix = df.corr()

# Print the correlation matrix to inspect the values
print("Correlation matrix:")
print(correlation_matrix)

# Plot the correlation heatmap using Seaborn
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap between Features and Target')
plt.show()


########################################################################################

#STEP 4


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_dt = GridSearchCV(dt_classifier, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_dt)

# SVC
from sklearn.svm import SVC

svc = SVC()
param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svc.fit(X_train, y_train)
best_svc = grid_search_svc.best_estimator_
print("Best SVC Model:", best_svc)


# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(rf_clf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
print("Best Random Forest Model:", best_rf)

# RandomizedSearchCV for Random Forest
from sklearn.model_selection import RandomizedSearchCV

param_dist_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search_rf = RandomizedSearchCV(rf_clf, param_dist_rf, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search_rf.fit(X_train, y_train)
best_random_rf = random_search_rf.best_estimator_
print("Best Random Forest Model (Randomized Search):", best_random_rf)

# Evaluate Training and Test Accuracy for Each Model

# Decision Tree
y_train_pred_dt = best_dt.predict(X_train)
y_test_pred_dt = best_dt.predict(X_test)
acc_train_dt = accuracy_score(y_train, y_train_pred_dt)
acc_test_dt = accuracy_score(y_test, y_test_pred_dt)
print(f"Decision Tree - Accuracy (Train): {acc_train_dt}, Accuracy (Test): {acc_test_dt}")


# SVC
y_train_pred_svc = best_svc.predict(X_train)
y_test_pred_svc = best_svc.predict(X_test)
acc_train_svc = accuracy_score(y_train, y_train_pred_svc)
acc_test_svc = accuracy_score(y_test, y_test_pred_svc)
print(f"SVC - Accuracy (Train): {acc_train_svc}, Accuracy (Test): {acc_test_svc}")

# Random Forest (Grid Search)
y_train_pred_rf = best_rf.predict(X_train)
y_test_pred_rf = best_rf.predict(X_test)
acc_train_rf = accuracy_score(y_train, y_train_pred_rf)
acc_test_rf = accuracy_score(y_test, y_test_pred_rf)
print(f"Random Forest (Grid Search) - Accuracy (Train): {acc_train_rf}, Accuracy (Test): {acc_test_rf}")

# Random Forest (Randomized Search)
y_train_pred_random_rf = best_random_rf.predict(X_train)
y_test_pred_random_rf = best_random_rf.predict(X_test)
acc_train_random_rf = accuracy_score(y_train, y_train_pred_random_rf)
acc_test_random_rf = accuracy_score(y_test, y_test_pred_random_rf)
print(f"Random Forest (Randomized Search) - Accuracy (Train): {acc_train_random_rf}, Accuracy (Test): {acc_test_random_rf}")



####################################################################################

# STEP 5: Model Performance Analysis

# Collect accuracy results from the best models
results = {
    'Decision Tree': acc_test_dt, 
    'Random Forest (Grid Search)': acc_test_rf, 
    'SVC': acc_test_svc,
    'Random Forest (Randomized Search)': acc_test_random_rf           
}

# Create a bar plot
model_names = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, color='purple')
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison after Hyperparameter Tuning')
plt.xlim(0, 1)  # Accuracy ranges from 0 to 1
plt.show()

from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix
import seaborn as sns

# Initialize dictionaries to hold the metrics for each model
results = {
    'Decision Tree': {},  
    'Random Forest (Grid Search)': {},
    'SVC': {},
    'Random Forest (Randomized Search)': {}
}

# Predictions for each model
models = {
    'Decision Tree': best_dt,  
    'Random Forest (Grid Search)': best_rf,
    'SVC': best_svc,
    'Random Forest (Randomized Search)': best_random_rf
}

# Calculate metrics for each model
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Store the metrics in the results dictionary
    results[model_name]['Accuracy'] = accuracy
    results[model_name]['F1 Score'] = f1
    results[model_name]['Precision'] = precision

# Display the results
print("Model Performance Metrics:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")

# confusion matrix for the best performing model (F1)
best_model_name = max(results, key=lambda x: results[x]['F1 Score'])  # Identify the best model based on F1.
best_model = models[best_model_name]

y_pred_best_model = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best_model)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title(f'Confusion Matrix for {best_model_name}')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification report for detailed metrics of the best model
from sklearn.metrics import classification_report
print("\nClassification Report for the Best Model:")
print(classification_report(y_test, y_pred_best_model))


##############################################################################################

#STEP 6 (stacked model performance analysis)

from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Base models
dt = DecisionTreeClassifier(random_state=42)  
svc = SVC(random_state=42)
rf = RandomForestClassifier(random_state=42)

# Define the stacked model
stacked_model = StackingClassifier(estimators=[
    ('dt', dt),  
    ('svc', svc), 
    ('rf', rf)], 
    final_estimator=RandomForestClassifier(random_state=42))

# Fit the model
stacked_model.fit(X_train, y_train)

# Predictions
y_train_pred_stacked = stacked_model.predict(X_train)
y_test_pred_stacked = stacked_model.predict(X_test)

# Evaluate accuracy
acc_train_stacked = accuracy_score(y_train, y_train_pred_stacked)
acc_test_stacked = accuracy_score(y_test, y_test_pred_stacked)

# Evaluate F1 score and precision
f1_train_stacked = f1_score(y_train, y_train_pred_stacked, average='weighted')
f1_test_stacked = f1_score(y_test, y_test_pred_stacked, average='weighted')
precision_train_stacked = precision_score(y_train, y_train_pred_stacked, average='weighted')
precision_test_stacked = precision_score(y_test, y_test_pred_stacked, average='weighted')

# Print results
print(f"Stacked Model Accuracy (Train): {acc_train_stacked:.2f}")
print(f"Stacked Model Accuracy (Test): {acc_test_stacked:.2f}")
print(f"Stacked Model F1 Score (Train): {f1_train_stacked:.2f}")
print(f"Stacked Model F1 Score (Test): {f1_test_stacked:.2f}")
print(f"Stacked Model Precision (Train): {precision_train_stacked:.2f}")
print(f"Stacked Model Precision (Test): {precision_test_stacked:.2f}")

# Update the results dictionary
results['Stacked Model'] = {
    'Accuracy': acc_test_stacked,
    'F1 Score': f1_test_stacked,
    'Precision': precision_test_stacked
}

# Confusion Matrix for the stacked model
cm_stacked = confusion_matrix(y_test, y_test_pred_stacked)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_stacked, annot=True, fmt='d', cmap='Greens', cbar=True, 
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix for Stacked Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#######################################################################################

#STEP 7


import joblib

# 'best_model' is the Decision Tree model selected from previous steps based on F1

# Save the model using joblib
joblib_file = "best_decision_tree_model.joblib"  # Define the filename for the model
joblib.dump(best_model, joblib_file)  # Save the model

print(f"Model saved as {joblib_file}")

# Prepare the coordinates for prediction
coordinates_to_predict = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

# Convert coordinates to DataFrame with correct column names
coordinates_to_predict_df = pd.DataFrame(coordinates_to_predict, columns=['X', 'Y', 'Z'])

# Predict the maintenance steps using the saved model
predictions = best_model.predict(coordinates_to_predict_df)

# Display the predictions
for coord, pred in zip(coordinates_to_predict, predictions):
    print(f"Coordinates: {coord} => Predicted Maintenance Step: {pred}")



























