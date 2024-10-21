import pickle

from flask import Flask
from flask import request
from flask import jsonify


def readFile(fileName):
    with open(fileName, 'rb') as f_in:
        result = pickle.load(f_in)
        return result


dv = readFile('dv.bin')
model = readFile('model1.bin')

app = Flask('subscription')


@app.route('/predict_sub', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    subscription = y_pred >= 0.5

    result = {
        'subscription_probability': float(y_pred),
        'subscription': bool(subscription)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
