from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import glob as glob
import os
from config import *
import pickle

print("Detector's Performance.")
os.chdir(OUTF)
csv_files = [f for f in glob.glob('*.csv')]
print(os.getcwd())
print(csv_files)
dfs = []
for csv in csv_files:
    df = pd.read_csv(csv)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

# Split the data into training and testing sets
X = final_df[['KP', 'mean', 'std']]
y = final_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=42)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
score = clf.score(X_test, y_test)
print(f'Test set accuracy: {score*100:.3f} %')

# Predict the probabilities of the positive class
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Print the AUC-ROC score
print(f'AUC-ROC score: {auc_roc*100:.4f} %')

y_pred = clf.predict(X_test) 
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)


new_data = pd.DataFrame([[3.5, 0.5, 0.2]], columns=X.columns)
prediction = clf.predict(new_data)
print(f'Prediction: {prediction[0]}')
probability = clf.predict_proba(new_data)[:, 1]
print(f'Probability of positive class: {probability[0]:.4f}')

# 1 means adv 
# -1 means org


import joblib
filename = 'finalized_model.sav'
joblib.dump(clf, filename)


'''
# Plot the confusion matrix
import matplotlib.pyplot as plt 
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
class_names = ['Adv', 'Non_adv']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
s = [['TN', 'FP'], ['FN', 'TP']]
fmt = 'd'
for i, j in enumerate(tick_marks):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
'''
