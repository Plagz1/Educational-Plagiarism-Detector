from src.preprocessing import preprocess_text


def test_preprocess_basic():
    text = "Dogs are running!"
    processed = preprocess_text(text)
    assert "dog" in processed
    assert "running" not in processed


def test_preprocess_empty():
    assert preprocess_text("") == ""
