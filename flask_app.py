from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from gawett import second_gawetts_rule

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    matrix_str = data["matrix"]

    # Convert input matrix to a numpy array
    matrix = np.array(json.loads(matrix_str)).astype("float")

    for i in range(matrix.shape[0]):
        matrix[i][i] = np.nan

    # Call second_gawetts_rule function
    results = second_gawetts_rule(matrix)

    return jsonify(results)


if __name__ == "__main__":
    app.run()
