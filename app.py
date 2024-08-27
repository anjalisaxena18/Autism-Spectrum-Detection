from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle
import joblib  # Import joblib for loading models

# Load the pre-trained model
# Replace 'model.pkl' with the actual filename

# Load the pre-trained model
  # Replace 'model.pkl' with the actual filename


app = Flask(__name__)

# Load the pre-trained models
image_model = load_model('model1.h5')  # Replace 'image_model.h5' with the actual filename
binary_model = joblib.load('model.pkl')  # Replace 'binary_model.pkl' with the actual filename

def predict_image_class(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust the target size based on your image classification model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image data
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return predicted_class

def predict_binary(form_data, model):
    # Assuming form_data is a dictionary with binary values as 0 or 1
    try:
        input_data = [float(form_data[f'feature{i}']) for i in range(1, 17)]  # Convert to float
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return int(prediction[0])
    except (KeyError, ValueError) as e:
        print(f"Error in predict_binary: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # Image classification
        file = request.files['file']
        if file.filename != '':
            # Create 'uploads' directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            img_path = 'uploads/' + file.filename
            file.save(img_path)
            predicted_class = predict_image_class(img_path, image_model)
            return redirect(url_for('show_result', result_type='Image', predicted_class=int(predicted_class)))

    elif all(f'feature{i}' in request.form for i in range(1, 17)):
        # Binary prediction based on form submission
        form_data = request.form.to_dict()
        print(f"Form data: {form_data}")
        predicted_binary = predict_binary(form_data, binary_model)
        if predicted_binary is not None:
            return redirect(url_for('show_result', result_type='Binary', predicted_class=int(predicted_binary)))

    return jsonify({'error': 'Invalid request. Please check your input data.'})

@app.route('/result/<result_type>/<predicted_class>')
def show_result(result_type, predicted_class):
    return render_template('result.html', result_type=result_type, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
