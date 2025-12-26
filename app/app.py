from __future__ import annotations

from flask import Flask, render_template, request
from app.retrieval import search

app = Flask(__name__)


def process_search_query(query_text) -> tuple[list, float, bool, str | None]:
    """Process a search query and return results with confidence metrics.

    Returns: (results, confidence, below_threshold, error_message)
    """
    if not query_text:
        return [], 0.0, False, "Query cannot be empty"

    try:
        response = search(query_text)
        return (
            response["results"],
            response["confidence"],
            response["below_threshold"],
            None
        )

    except Exception as e:
        return [], 0.0, True, str(e)


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    """Handle main page rendering and search requests."""
    query = ""
    results = []
    confidence = 0.0
    below_threshold = False
    error_message = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        results, confidence, below_threshold, error_message = process_search_query(query)

    return render_template(
        "index.html",
        query=query,
        results=results,
        confidence=confidence,
        below_threshold=below_threshold,
        error_message=error_message
    )


if __name__ == "__main__":
    app.run(debug=True)
