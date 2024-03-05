from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO
from Detector import *
import os
import io
import zipfile
import shutil
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import jsonify
from sklearn.metrics import classification_report

app = Flask(__name__)
socketio = SocketIO(app)

# Configure your model and class file paths
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
classFile = "coco.names"
threshold = 0.5

# Initialize the detector
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()

# Function to download a file from a URL
def download_file(url, destination):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
    except Exception as e:
        raise ValueError(f"Error downloading file: {str(e)}")

# Function to extract a zip file
def extract_zip(zip_path, extract_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        raise ValueError(f"Error extracting zip file: {str(e)}")

# Function to train the model
def train_model(train_path, num_classes, batch_size, epochs):
    try:
        model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Dropout layer with a dropout rate of 0.5
        Dense(num_classes, activation='softmax')
    ])

        
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Adding data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        train_generator = train_datagen.flow_from_directory(train_path,
                                                            target_size=(224, 224),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')
        
        # Adding learning rate scheduling
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
        
        model.fit(train_generator, epochs=epochs, callbacks=[lr_scheduler])
        
        return model
    except Exception as e:
        raise ValueError(f"Error training model: {str(e)}")



@socketio.on('video_feed')
def video_feed():
    video_path = 'your_video_path.mp4'
    detector.predictVideo(video_path, threshold=0.5)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/for_image')
def for_image():
    return render_template('photo.html')

@app.route('/for_video')
def for_video():
    return render_template('video.html')

# Route to serve the index.html file
@app.route('/train')
def index():
    return render_template('index.html')

# Route to serve the upload.html file for testing
@app.route('/test')
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Handle file upload and perform object detection
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            # Save the uploaded file
            image_path = 'testing/' + uploaded_file.filename
            uploaded_file.save(image_path)

            # Perform object detection
            detector.predictImage(image_path, threshold)

            # Render the result or redirect to a new page as needed
            return render_template('result.html')

    # Redirect to the main page if no file is uploaded or an error occurs
    return redirect(url_for('index'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        # Handle video upload and perform object detection
        uploaded_video = request.files['video']

        if uploaded_video.filename != '':
            # Save the uploaded video
            videopath = 'testing/' + uploaded_video.filename
            uploaded_video.save(videopath)

            detector.predictVideo(videopath, threshold)

            # Redirect to the video.html page with the video path
            return render_template('video.html', video_path=videopath)

    # Redirect to the main page if no video is uploaded or an error occurs
    return redirect(url_for('for_video'))

# Route to test the trained model
@app.route('/test', methods=['POST'])
def test():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Load the trained model
        model_path = 'trained_model.h5'
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'})

        model = load_model(model_path)

        # Load and preprocess the image
        img_bytes = file.read()
        img_stream = io.BytesIO(img_bytes)
        img = tf.image.decode_image(img_stream.read(), channels=3)
        img = tf.image.resize(img, [224, 224])  # Resize image
        img = tf.expand_dims(img, axis=0)  # Create a batch
        img = img / 255.0  # Normalize pixel values

        # Predict the class probabilities
        predictions = model.predict(img)
        predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]

        # Get the class name from the index
        class_names = os.listdir('extracted_dataset')
        predicted_class_name = class_names[predicted_class_index]

        # Check if accuracy is below 0.84 and declare object not detected
        if float(predictions[0][predicted_class_index]) < 0.8:
            return jsonify({'class_name': 'Object not detected'})

        return jsonify({
            'class_name': predicted_class_name,
            'accuracy': float(predictions[0][predicted_class_index]),
        })
    except Exception as e:
        return jsonify({'error': f'Error during testing: {str(e)}'})


# Route to train the model
@app.route('/train', methods=['POST'])
def train():
    try:
        if 'driveLink' not in request.json:
            return 'No Google Drive link provided'
        
        drive_link = request.json['driveLink']
        zip_path = 'dataset.zip'
        extract_path = 'extracted_dataset'

        # Download the file from Google Drive
        download_file(drive_link, zip_path)

        # Extract the zip file
        extract_zip(zip_path, extract_path)

        train_path = 'extracted_dataset'  # Assuming images are already organized into class folders
        num_classes = len(os.listdir(train_path))  # Automatically determine number of classes
        batch_size = 32
        epochs = 10  # Adjust as needed

        # Training the model
        model = train_model(train_path, num_classes, batch_size, epochs)
        model.save('trained_model.h5')

        return 'Model trained and saved successfully'
    except Exception as e:
        return f'Error during training: {str(e)}'

if __name__ == '__main__':
    socketio.run(app, debug=True)