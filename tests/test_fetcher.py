from fastapi.testclient import TestClient

from apps.fetcher import app


def test_get_survival():
    with TestClient(app=app) as client:
        r = client.get('survival')

        r.raise_for_status()

        result: dict = r.json()
        assert 'items' in result.keys()

        assert len(result['items']) == 50

        assert set(result['items'][0]['data'].keys()) == {'patient', 'field', 'sample'}
