from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO
from Detector import *

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

@socketio.on('video_feed')
def video_feed():
    video_path = 'your_video_path.mp4'
    detector.predictVideo(video_path, threshold=0.5)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/for_image')
def for_image():
    return render_template('index.html')

@app.route('/for_video')
def for_video():
    return render_template('video.html')

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

if __name__ == '__main__':
    socketio.run(app, debug=True)