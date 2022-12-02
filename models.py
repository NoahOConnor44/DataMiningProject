from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from utils import get_average_f1

'''
References: 

1) Hyperparameter tuning reference: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
2) 

'''

# This creates a random forest classifier model with or without hypertuning, 
# splits the training/test data and averages the f1 score over 10 runs
def random_forest_accuracy(df,tuning=True):
    # Hyper parameter tuning code, it takes a while to run so we commented it out.
    # The best parameters are specified and ran below when tuning is set to true
    '''
    param_grid = {
         'max_depth': [12, 15, 20, 25],
         'max_features': [12, 15, 25],
         'min_samples_leaf': [1, 3, 5],
         'n_estimators': [75, 100, 150]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    '''

    print('Evaluating random forest model')
    model = RandomForestClassifier(max_depth=12, max_features=25, min_samples_leaf=1, n_estimators=150) if tuning else RandomForestClassifier()
    return get_average_f1(model,df)


# This creates a gradient boosting classifier model with or without hypertuning, 
# splits the training/test data and averages the f1 score over 10 runs
def gradient_boosting_accuracy(df,tuning=True):
    print('Evaluating gradient boosting model')
    # Hyper parameter tuning code, it takes a while to run so we commented it out.
    # The best parameters are specified and ran below when tuning is set to true
    '''
    param_grid_gb = {
    'n_estimators': [300, 500],
    'max_depth': [3, 5],
    'min_samples_leaf': [2, 3]
    }
    gb_tuned = GradientBoostingClassifier()
    grid_search_gb = GridSearchCV(estimator=gb_tuned, param_grid=param_grid_gb, cv=3)
    grid_search_gb.fit(X_train, y_train)
    
    print(grid_search_gb.best_params_)
    '''
    # # The best parameters given were max_depth=3, min_samples_leaf=3, n_estimators=300
    model = GradientBoostingClassifier(max_depth=3, min_samples_leaf=3, n_estimators=300) if tuning else GradientBoostingClassifier()
    return get_average_f1(model,df)


# This creates a decision tree classifier model with or without hypertuning, 
# splits the training/test data and averages the f1 score over 10 runs
def decision_tree_accuracy(df,tuning=True):
    # Hyper parameter tuning code, it takes a while to run so we commented it out.
    # The best parameters are specified and ran below when tuning is set to true
    '''
    dtree = DecisionTreeClassifier()
    param_grid_dtree = {
         'max_depth': [3, 5, 7, 10],
         'min_samples_leaf': [1, 2, 3, 5],
         'criterion': ["gini", "entropy"]
    }
    grid_search_dt = GridSearchCV(estimator=dtree, param_grid=param_grid_dtree, cv=3, scoring='f1')
    grid_search_dt.fit(X_train, y_train)
    print(grid_search_dt.best_params_)
    '''

    print('Evaluating Decision Tree Classifier model')
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, criterion='entropy') if tuning else DecisionTreeClassifier()
    return get_average_f1(model,df)

# This creates a knn classifier model with or without hypertuning, 
# splits the training/test data and averages the f1 score over 10 runs
def knn_accuracy(df,tuning=True, smote=True):
    # Hyper parameter tuning code, it takes a while to run so we commented it out.
    # The best parameters are specified and ran below when tuning is set to true
    '''
    knn_tuned = KNeighborsClassifier()
    param_grid_knn = {
        'leaf_size': [3, 5, 10, 20],
        'n_neighbors': [2, 3, 5],
        'p': [1,2]}
    grid_search_knn = GridSearchCV(estimator=knn_tuned, param_grid=param_grid_knn)
    grid_search_knn.fit(X_train,y_train)
    print(grid_search_knn.best_params_)
    '''
    print('Evaluating KNeighborsClassifier (KNN) model with SMOTE=',smote)
    model = KNeighborsClassifier(metric='manhattan', n_neighbors=2, weights='distance') if tuning else KNeighborsClassifier()
    return get_average_f1(model, df,smote)

# This creates an SVM classifier model with or without hypertuning, 
# splits the training/test data and averages the f1 score over 10 runs
def svm_accuracy(df,tuning=True,smote=True):
    # Hyper parameter tuning code, it takes a while to run so we commented it out.
    # The best parameters are specified and ran below when tuning is set to true
    '''
    SVM with SMOTE and hyperparameter tuning
    param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly']}
    svm = SVC()
    grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm)
    grid_search_svm.fit(X_train_SMOTE, y_train_SMOTE)
    print(grid_search_svm.best_params_)
    '''
    
    print('Evaluating SVM model with SMOTE=',smote)
    model = SVC(C=10, gamma=0.001) if tuning else SVC()
    return get_average_f1(model,df, smote)