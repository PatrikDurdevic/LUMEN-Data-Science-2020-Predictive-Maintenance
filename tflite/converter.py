from tensorflow import keras
from tensorflow import lite
import os
from tqdm import tqdm

for root, dirs, files in os.walk("../models/"):
	for file in files:
		if file.endswith(".h5"):
			modelPath = os.path.join(root, file)

			rootParts = root.split('/')
			processedRoot = "/".join(rootParts[2:])
			fileParts = file.split('.')
			targetModelPath = os.path.join(processedRoot, fileParts[0] + ".tflite")

			if not os.path.exists(processedRoot):
				try:
					os.makedirs(processedRoot)
				except:
					pass

			model = keras.models.load_model(modelPath)
			converter = lite.TFLiteConverter.from_keras_model(model)
			flite_model = converter.convert()
			open(targetModelPath, "wb").write(tflite_model)