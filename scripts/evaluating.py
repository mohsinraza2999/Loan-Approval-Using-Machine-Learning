import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def training_results(model_object, metric:str, y_test,y_pred):
 
    metric_dict = {'precision': 'mean_test_precision',
                 'recall': 'mean_test_recall',
                 'f1': 'mean_test_f1',
                 'accuracy': 'mean_test_accuracy',
                 }

    cv_results = pd.DataFrame(model_object.cv_results_)

    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    training = pd.DataFrame({'result type': ['Training results'],
                        'precision': [precision],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy],
                        },
                       )

    testing=test_result(y_test,y_pred)

    saving_results('training_testing_result','results/',pd.concat([training,testing],axis=0))

    save_cm_matrix(y_pred,y_test, model_object)

    saving_results('best_parameters','results/',pd.DataFrame(data=model_object.best_params_,index=range(0,1)))

def test_result(y_pred,y_test):
    

    acc=accuracy_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    prec=precision_score(y_test,y_pred)
    table = pd.DataFrame({'result type': ['Testing results'],
                        'precision': [prec],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [acc]
                        })
    
    
    return table

def save_cm_matrix(y_pred,y_test,model):
    cm=confusion_matrix(y_test,y_pred,labels=model.classes_)
    display_cm=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)


def saving_results(name,path,data:pd.DataFrame):

    data.to_csv(path+name+'.csv')
