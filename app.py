from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS # for flask cors

from chat import get_response

app = Flask(__name__)
# CORS(app)  # for flask cors

# As we addded flask-cors so we don't need this
@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)