import pickle
import json
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load model and scalar
with open('reg_model.pkl', 'rb') as file:
    regmodel = pickle.load(file)
with open('scaling.pkl', 'rb') as file:
    scalar = pickle.load(file)


def convert_to_tuple(obj):
    if isinstance(obj, np.ndarray):
        return tuple(obj.tolist())
    elif isinstance(obj, list):
        return [convert_to_tuple(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_tuple(value) for key, value in obj.items()}
    return obj


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = json.loads(request.data, object_hook=convert_to_tuple)
        print(data)

        input_data = np.array(list(data.values())).reshape(1, -1)
        new_data = scalar.transform(input_data)
        output = regmodel.predict(new_data)
        print(output[0])
        return jsonify(output[0])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The House price prediction is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
