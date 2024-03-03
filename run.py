from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"
imagePath = "testing/2.jpg"
videoPath = "testing/walk.mp4"
threshold=0.5

detector= Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath,threshold)
# detector.predictVideo(videoPath,threshold)
