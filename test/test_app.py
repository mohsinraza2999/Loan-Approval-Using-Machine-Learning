from fastapi.testclient import TestClient
client = TestClient(app)
from main import app
import numpy as np
sample = [0,0.0, 1, 0, 5849, 0.0, 1200, 360.0, 1.0, 1]
prepared_sample = np.array(sample).reshape(1, -1)
results=['Loan is Approved','Loan is Not Approved']
def test_predict():
    payload = {
        "feature1": 0.0,
        "feature2": 1,
        "feature3": 0,
        "feature4": 5849,
        "feature5": 0.0,
        "feature6": 1200,
        "feature7": 360.0,
        "feature8": 1.0,
        "feature9": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in results
