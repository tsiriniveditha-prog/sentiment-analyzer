from app import app

if __name__ == __main__:
    from waitress import serve
    import os
    port = int(os.environ.get(PORT, 5000))
    print(fServing SentimentIQ on http://0.0.0.0:{port})
    serve(app, host=0.0.0.0, port=port)

