# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import openai

app = Flask(__name__)

# Load your machine learning model
model = load_model('trained_model.h5')

# Configure OpenAI GPT API
openai.api_key = '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'

# Function to classify air quality
def classify_air_quality(value):
    if value <= 50:
        return "Good"
    elif 50 < value <= 100:
        return "Moderate"
    elif 100 < value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 150 < value <= 200:
        return "Unhealthy"
    elif 200 < value <= 300:
        return "Very Unhealthy"
    elif value > 300:
        return "Hazardous"
    else:
        return "Exception Occurred!"
    
# Function to get chatbot response
def get_chatbot_response(user_query):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Air quality: {user_query}",
        max_tokens=150
    )
    return response.choices[0].text.strip()    

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Read and preprocess the image
        img = Image.open(file)
        img = img.resize((224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg16.preprocess_input(img)


        # Make prediction using the model
        prediction = model.predict(img)

        # Classify air quality based on prediction
        result_class = classify_air_quality(prediction[0][0])
        # Get chatbot response
        chatbot_response = get_chatbot_response(result_class)

        return jsonify({'result_class': result_class, 'chatbot_response': chatbot_response})
    return jsonify({'error': 'Invalid file format'})

# Route to handle user queries and get chatbot response
@app.route('/chat', methods=['GET'])
def chat():
    user_query = request.args.get('query')
    chatbot_response = get_chatbot_response(user_query)
    return jsonify({'chatbot_response': chatbot_response})

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)
