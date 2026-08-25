"""
Unit tests for SentimentAnalyzer
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from model.sentiment_model import SentimentAnalyzer


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


class TestSentimentLabels:
    def test_positive_text(self, analyzer):
        result = analyzer.predict("This product is absolutely amazing and wonderful!")
        assert result["label"] == "Positive"

    def test_negative_text(self, analyzer):
        result = analyzer.predict("This is terrible and awful, I hate it.")
        assert result["label"] == "Negative"

    def test_neutral_text(self, analyzer):
        result = analyzer.predict("The box arrived today.")
        assert result["label"] == "Neutral"

    def test_negation(self, analyzer):
        result = analyzer.predict("This is not bad at all, I really like it.")
        assert result["label"] in ("Positive", "Neutral")

    def test_intensifier(self, analyzer):
        pos = analyzer.predict("This is very good.")
        basic = analyzer.predict("This is good.")
        assert pos["score"] >= basic["score"]


class TestReturnSchema:
    def test_keys_present(self, analyzer):
        result = analyzer.predict("Hello world")
        for key in ["label", "score", "confidence", "word_count", "highlights", "text_preview"]:
            assert key in result

    def test_score_range(self, analyzer):
        for text in ["amazing", "terrible", "the cat sat"]:
            r = analyzer.predict(text)
            assert -1.0 <= r["score"] <= 1.0

    def test_confidence_range(self, analyzer):
        for text in ["great", "bad", "okay"]:
            r = analyzer.predict(text)
            assert 0.0 <= r["confidence"] <= 1.0

    def test_empty_string(self, analyzer):
        result = analyzer.predict("")
        assert result["label"] == "Neutral"
        assert result["score"] == 0.0

    def test_emoji_positive(self, analyzer):
        result = analyzer.predict("😊😊😊")
        assert result["score"] > 0

    def test_emoji_negative(self, analyzer):
        result = analyzer.predict("👎👎👎")
        assert result["score"] < 0

    def test_long_text_preview(self, analyzer):
        long_text = "good " * 100
        result = analyzer.predict(long_text)
        assert result["text_preview"].endswith("...")


class TestHighlights:
    def test_highlights_are_list(self, analyzer):
        result = analyzer.predict("The product is excellent and fantastic.")
        assert isinstance(result["highlights"], list)

    def test_highlight_structure(self, analyzer):
        result = analyzer.predict("Wonderful and beautiful experience.")
        for h in result["highlights"]:
            assert "word" in h
            assert "type" in h
            assert h["type"] in ("positive", "negative")

    def test_max_highlights(self, analyzer):
        text = " ".join(["good"] * 20)
        result = analyzer.predict(text)
        assert len(result["highlights"]) <= 8


class TestFlaskRoutes:
    @pytest.fixture
    def client(self):
        from app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_index_route(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"SentimentIQ" in response.data

    def test_analyze_route_success(self, client):
        response = client.post("/analyze", json={"text": "This product is absolutely amazing!"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["label"] == "Positive"
        assert data["score"] > 0

    def test_analyze_route_empty(self, client):
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_batch_route_success(self, client):
        response = client.post("/batch", json={"texts": ["Great service!", "Terrible experience."]})
        assert response.status_code == 200
        data = response.get_json()
        assert len(data["results"]) == 2
        assert data["results"][0]["label"] == "Positive"
        assert data["results"][1]["label"] == "Negative"

    def test_batch_route_empty(self, client):
        response = client.post("/batch", json={"texts": []})
        assert response.status_code == 400
        assert "error" in response.get_json()
