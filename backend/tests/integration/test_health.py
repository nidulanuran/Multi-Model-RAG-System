from fastapi.testclient import TestClient


from app.main import app
from app.config import settings

client=TestClient(app)

def test_health_check():

    response=client.get("/api/v1/health")

    assert response.status_code==200

    assert response.json()=={
        "status":"ok",
        "service":settings.app_name,
        "version":settings.app_version,
        "environment":settings.environment
    }