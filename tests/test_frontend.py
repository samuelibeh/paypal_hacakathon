from fastapi.testclient import TestClient

from app.main import app


def test_index_serves_the_frontend():
    resp = TestClient(app).get("/")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="chatForm"' in resp.text
