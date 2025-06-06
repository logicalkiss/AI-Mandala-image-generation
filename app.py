import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Define class labels
CLASS_LABELS = {
    0: 'Zhabdrung_Ngawang_Namgyal',
    1: 'Sangay_Tempa',
    2: 'Guru_Rinpoche'
}

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model with proper input shape
MODEL_PATH = 'models/myModel.h5'
model = load_model(MODEL_PATH, compile=False)  # Load without compilation first
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile after loading

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load and preprocess the image
            img = image.load_img(filepath, target_size=(299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image

            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            
            # Process predictions
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_class = CLASS_LABELS[predicted_class_index]
            
            # Get all class probabilities
            class_probabilities = {
                CLASS_LABELS[i]: float(prob) 
                for i, prob in enumerate(predictions[0])
            }
            
            # Clean up the uploaded file
            os.remove(filepath)

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities
            })

        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    # Ensure the models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    app.run(debug=True) 