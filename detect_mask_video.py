import logging
import os
import threading
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

def mask_detect(faceNet, args, maskNet):
	while(True):
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)

		labelList = []
		iterCount = 0

		while(len(labelList) < 30):
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
			iterCount += 1
			if(iterCount > 50):
				vs.stop()
				iterCount = False
				break
			(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred
				label = 1 if mask > withoutMask else 0
				labelList.append(label)

			vs.stop()
		print(labelList)
		if(iterCount == False):
			print("Could not analyze presence of mask.\nPlease bring your face closer to the camera")
		else:
			if(labelList.count(0) > (0.8 * 30)): return False
			else: return True
		

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

result = mask_detect(faceNet, args, maskNet)