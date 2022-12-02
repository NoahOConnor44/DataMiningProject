from dataModification import adjust_main_df
from models import random_forest_accuracy, gradient_boosting_accuracy, decision_tree_accuracy, knn_accuracy, svm_accuracy
from utils import plot_all
import json


# clean, merge, and drop columns
df = adjust_main_df()

print('-------------------------------------------------------')
print('|  About to evaluate all models and report findings...|')
print('|  Estimated time to run: ~3-4 minutes                |')
print('-------------------------------------------------------\n\n')

# Optional tuning parameter for every model. Automatically set to true
random_forest = random_forest_accuracy(df)
gradient_boosting = gradient_boosting_accuracy(df)
decision_tree = decision_tree_accuracy(df)

# Evaluating KNN with and without SMOTE
knn_smote = knn_accuracy(df) 
knn = knn_accuracy(df,smote=False) 

# Evaluating SVM with and without SMOTE
svm_smote = svm_accuracy(df)
svm = svm_accuracy(df,smote=False)

# store 10 run average f1 score for each model

scores = {
    'Random Forest': random_forest,
    'Gradient Boosting': gradient_boosting,
    'Decision Tree': decision_tree,
    'KNN With Smote': knn_smote,
    'Knn Without Smote': knn,
    'SVM With Smote': svm_smote,
    'SVM Without Smote': svm
}

with open('results.txt', 'w') as f:
     f.write(json.dumps(scores))

# print graph with findings
plot_all(df,scores)


