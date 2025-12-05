from fastapi.testclient import TestClient
from main import app

def test_predict_positive():
    with TestClient(app) as client:
        response = client.post("/process", json={"comments": ["I love you!"]})
        json_data = response.json()
        assert response.status_code == 200
        assert json_data["predictions"] == [0]


def test_predict_negative():
    with TestClient(app) as client:
        response = client.post("/process", json={"comments": ["Fuck you!"]})
        json_data = response.json()
        assert response.status_code == 200
        assert json_data["predictions"] == [1]
