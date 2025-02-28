from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("rf.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/result", methods=["POST"])
def output():
    # Retrieve input values from the form
    var1 = float(request.form["src_bytes"])
    var2 = float(request.form["dst_bytes"])
    var3 = int(request.form["count"])
    var4 = float(request.form["srv_count"])
    var5 = float(request.form["dst_host_count"])
    var6 = int(request.form["protocol_type"])  
    var7 = int(request.form["service"])
    var8 = int(request.form["flag"])
    var9 = float(request.form["same_srv_rate"])
    var10 = float(request.form["diff_srv_rate"])
    var11 = float(request.form["dst_host_same_srv_rate"])
    var12 = float(request.form["dst_host_diff_srv_rate"])

    # Format data correctly for the model
    predict_data = np.array([
        var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12
    ]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(predict_data)

    # Define target classes (handling string cases)
    target_mapping = {
        0: "Normal",
        1: "Anomaly",
        "normal": "Normal",  # Handling case where model returns string
        "anomaly": "Anomaly"
    }

    # Ensure the prediction is mapped correctly
    result = target_mapping.get(prediction[0], "Unknown")

    # Pass the result to result.html
    return render_template("result.html", result=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
