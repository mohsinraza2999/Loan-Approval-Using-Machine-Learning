import numpy as np
import pickle

def load_model(data):
    path='E:/google data analyzing practice/LoanAproval Project/model/'
    prepared_sample = np.array(data).reshape(1, -1)
    with open(path+'Random Forest Model.pickle', 'rb') as to_read:
        model=pickle.load(to_read)
    # Replace with actual model prediction logic
    prediction = model.predict(prepared_sample)
    if prediction[0]==1:
        message='Loan is Approved'
    else:
        message='Loan is Not Approved'
    
    return message
    
    
    