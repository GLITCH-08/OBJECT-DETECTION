from flask import Flask, render_template, request, redirect, url_for
from Detector import *

app = Flask(__name__)

# Configure your model and class file paths
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"
classFile = "coco.names"
threshold = 0.5

# Initialize the detector
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()

@app.route('/')
def index():
    return render_template('index.html')

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
            return render_template('result.html', image_path=image_path)

    # Redirect to the main page if no file is uploaded or an error occurs
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
