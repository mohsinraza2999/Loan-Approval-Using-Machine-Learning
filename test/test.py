from scripts.predict import load_model
import numpy as np
sample = [0,0.0, 1, 0, 5849, 0.0, 1200, 360.0, 1.0, 1]
prepared_sample = np.array(sample).reshape(1, -1)
results=['Loan is Approved','Loan is Not Approved']
def test_predict():
    assert load_model(prepared_sample) in results