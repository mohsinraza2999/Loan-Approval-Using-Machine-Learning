from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app
import numpy as np
client = TestClient(app)

sample = [0,0.0, 1, 0, 5849, 0.0, 1200, 360.0, 1.0, 1]
prepared_sample = np.array(sample).reshape(1, -1)
results=['Loan is Approved','Loan is Not Approved']
def test_predict():
    
    response = client.post("/")
    assert response.status_code == 200
    #assert response.json()["prediction"] in results
