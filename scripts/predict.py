import numpy as np
import pickle
import os




def load_model(data):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build path to the model file
    model_path = os.path.join(BASE_DIR, "model", "Random Forest Model.pickle")


    prepared_sample = np.array(data).reshape(1, -1)
    with open(model_path, 'rb') as to_read:
        model=pickle.load(to_read)
    # Replace with actual model prediction logic
    prediction = model.predict(prepared_sample)
    if prediction[0]==1:
        message='Loan is Approved'
    else:
        message='Loan is Not Approved'
    
    return message
    
    
    