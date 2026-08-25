from flask import Flask, render_template, request, jsonify
from model.sentiment_model import SentimentAnalyzer

app = Flask(__name__)
analyzer = SentimentAnalyzer()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = analyzer.predict(text)
    return jsonify(result)

@app.route("/batch", methods=["POST"])
def batch_analyze():
    data = request.get_json()
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    results = [analyzer.predict(t) for t in texts if t.strip()]
    return jsonify({"results": results})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug)
