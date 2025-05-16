# app.py
from flask import Flask, request, jsonify
from model.train import train_model
from model.predict import load_model, predict_online, predict_offline
app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    filepath = data.get('filepath')
    model_type = data.get('model_type', 'rf')
    hyperparams = data.get('hyperparams', {})
    result = train_model(filepath, model_type, hyperparams)
    return jsonify(result)

@app.route('/predict-online', methods=['POST'])
def predict_online_route():
    data = request.json
    model_type = data.get('model_type', 'rf')
    features = data.get('features')
    model = load_model(f'saved_models/{model_type}_model.pkl')
    prediction = predict_online(model, features)
    return jsonify({"prediction": int(prediction)})

@app.route('/predict-offline', methods=['POST'])
def predict_offline_route():
    data = request.json
    model_type = data.get('model_type', 'rf')
    filepath = data.get('filepath')
    model = load_model(f'saved_models/{model_type}_model.pkl')
    predictions = predict_offline(model, filepath)
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
