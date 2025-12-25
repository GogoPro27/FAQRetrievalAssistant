from flask import Flask, render_template, request
from app.retrieval import search

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results = search(query)

    return render_template(
        "index.html",
        query=query,
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)
