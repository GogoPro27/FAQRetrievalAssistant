from flask import Flask, render_template, request
from app.retrieval import search

app = Flask(__name__)


def process_search_query(query_text):
    if not query_text:
        return [], 0.0

    response = search(query_text)
    return response["results"], response["confidence"], response["below_threshold"]


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    confidence = 0.0
    below_threshold = False

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        results, confidence, below_threshold = process_search_query(query)

    return render_template(
        "index.html",
        query=query,
        results=results,
        confidence=confidence,
        below_threshold=below_threshold
    )


if __name__ == "__main__":
    app.run(debug=True)
