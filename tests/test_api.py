from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_query_endpoint():
    response = client.post("/query", json={"question": "Test question"})
    assert response.status_code == 200
    assert "answer" in response.json()
