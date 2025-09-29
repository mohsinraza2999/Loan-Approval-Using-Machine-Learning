from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
from scripts import evaluating


def get_data():

    clean_data=pd.read_csv('data/training_data/preprocess_data.csv')

    return split_data(clean_data)

def split_data(df):

    y=df['Loan_Status']

    X=df.drop(columns='Loan_Status',axis=1)

    
    return train_test_split(X,y,stratify=y,random_state=13,test_size=0.2)

def get_hyper_parameters():
    
    cv_params={'max_depth': [None],
                'max_features': [1.0],
                'max_samples': [0.7],
                'min_samples_leaf': [2,4],
                'min_samples_split': [4,6],
                'n_estimators': [130,140]
                }


    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    return scoring, cv_params


def train():
    RFC=RandomForestClassifier(random_state=0)

    scoring, cv_params=get_hyper_parameters()

    T_RFC=GridSearchCV(RFC,cv_params,scoring=scoring,refit='f1')

    X_train,X_test,y_train,y_test=get_data()

    T_RFC.fit(X_train,y_train)

    write_pickle('model/', T_RFC, 'Random Forest Model')

    save_FI(T_RFC,X_test.columns)

    y_pred=T_RFC.best_estimator_.predict(X_test)

    evaluating.training_results(T_RFC, 'f1', y_test, y_pred)




def save_FI(model,columns):
    importances = model.best_estimator_.feature_importances_
    RF_importances = pd.DataFrame(importances, index=columns)
    evaluating.saving_results('Feature_Importance','results/',RF_importances)

def write_pickle(path, model_object, save_name:str):
    '''
    save_name is a string.
    '''
    with open(path + save_name + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)
    