from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"
detector= Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
