def test_prediction_module_exists():
    from app.main import app

    assert app is not None