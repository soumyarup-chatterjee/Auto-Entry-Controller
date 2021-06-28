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

import serial
import pyfirmata

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type = str, default = "face_detector", help = "path to face detector model directory")
ap.add_argument("-m", "--model", type = str, default = "mask_detector.model", help = "path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type = float, default=0.5, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

quitFlag = False

print("[INFO] Starting video stream...")
try:
	vs = VideoStream(src = 0).start()
except Exception:
	print("[ERROR] Could not start video stream. Exiting...")
	sys.quit()

time.sleep(1.0)

#----------------------------------------------------------------------------------------------------------------------------

def detect_and_predict_mask(frame, faceNet, maskNet, detectFaceOnly):
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

	if(detectFaceOnly and len(faces) > 0): return True
	if(detectFaceOnly and len(faces) == 0): return False

	if len(faces) > 0:
		faces = np.array(faces, dtype = "float32")
		preds = maskNet.predict(faces, batch_size = 32)
	else:
		return (False, False)
	return (locs, preds)

#----------------------------------------------------------------------------------------------------------------------------

def captureFrame():
	retries = frame = 0
	passInfo = False
	while retries < 2:
		try:
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
		except Exception:
			print("[ERROR] Could not capture the frame. Retrying...")
			retries += 1
			continue
		passInfo = True
		break
	return (frame, passInfo)

#----------------------------------------------------------------------------------------------------------------------------

def faceIsPresent():
	print("[INFO] Looking for a person")
	frame, ok = captureFrame()
	if(ok and detect_and_predict_mask(frame, faceNet, maskNet, True)):
		return True
	return False

#----------------------------------------------------------------------------------------------------------------------------

def mask_detect():
	labelList = []
	iterCount = 0

	while(len(labelList) < 30):
		iterCount += 1
		frame, success = captureFrame()
		if(not success or iterCount > 50):
			print("[ERROR] Could not analyze presence of mask.")
			return False
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, False)
		for (box, pred) in zip(locs, preds):
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred
			label = 1 if mask > withoutMask else 0
			labelList.append(label)

	if(labelList.count(0) > (0.8 * 30)):
		return False
	return True

#----------------------------------------------------------------------------------------------------------------------------

def temp_detect(arduino):
	time.sleep(1)
	data = arduino.readline()

	failure, temps = False, []
	try:
		decodedData = str(data[0:len(data)].decode("utf-8"))
		tempVal = decodedData.split('x')
		for item in tempVal:
			temps.append(float(item))
	except Exception:
		print("[ERROR] Erroneous sensor value")
		failure = True

	if(not failure):
		return temps
	return []
	data = 0

#----------------------------------------------------------------------------------------------------------------------------

def worker():
	retries = 0
	while (not quitFlag and retries < 3):
		passFlag = False

		try:
			arduino = serial.Serial('/dev/ttyACM0', 9600)
		except Exception:
			retries += 1
			print("[ERROR] Connection with Arduino failed. Retrying...")
			time.sleep(2)
			continue
		
		if retries > 0: print("Connection with Arduino succesfully established")
		retries = 0

		face_result = faceIsPresent()
		temps = temp_detect(arduino)
		
		if not temps:
			retries += 1
			continue
		ambTemp, objTemp = [temps[i] for i in (0, 1)]

		if((abs(objTemp - ambTemp) > 1) or face_result):
			print("\nPerson detected. Scanning...")
			print("Temperature Value : " + str(objTemp))
			if objTemp > 37.5:
				print("Temperature Check : FAIL")
			else:
				print("Temperature Check : PASS")
				result = mask_detect()
				if result == 1:
					print("Mask Check : PASS")
					passFlag = True
				else:
					print("Mask Check : FAIL")
			if passFlag: 
				print("Result : ALLOWED")
				arduino.write(bytes("1", "utf-8"))
			else:
				print("Result : NOT ALLOWED\n")
		else:
			print("No one detected. Waiting...")
		time.sleep(2)

	if retries >= 3:
		print("Could not connect with the Arduino. Press any key to exit program")

#----------------------------------------------------------------------------------------------------------------------------

worker_thread = threading.Thread(target = worker, daemon = False)
worker_thread.start()
ch = input()
quitFlag = True
worker_thread.join()
print("Stopping video stream")
vs.stop()
print("Exiting program...")