
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load the pre-trained model
model = load_model('model (2).h5')

# Initialize Flask app
app = Flask(__name__)

# Define the classes
classes = ['American', 'Chinese', 'European', 'Indian', 'Japanese', 'Korean']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = classes[class_index]

        return jsonify({'cuisine': predicted_class}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
