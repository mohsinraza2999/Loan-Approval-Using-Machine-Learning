from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app
client = TestClient(app)

results=['Loan is Approved','Loan is Not Approved']
def test_app():
    
    response = client.get("/")
    print("RESPONSE : \n",response)
    assert response.status_code == 200

def test_prediction():
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
    for key in payload.keys():
        payload[key]=str(payload[key])
    response = client.post("/submit", data=payload)
    print("RESPONSE :\n",response)
    assert response.status_code == 200
    assert response.json()["prediction"] in results


