from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ML model
with open("profit_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():

    profit = None

    if request.method == "POST":
        try:
            rd = float(request.form["rd"])
            admin = float(request.form["admin"])
            marketing = float(request.form["marketing"])

            input_data = np.array([[rd, admin, marketing]])
            prediction = model.predict(input_data)[0]

            profit = round(prediction, 2)

        except:
            profit = "Invalid Input"

    return render_template("index.html", profit=profit)


if __name__ == "__main__":
    app.run(debug=True)