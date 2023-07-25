from tkinter import *
from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import np_utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model 
import plot_model
import cv2
import os
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

model=load_model('28april.h5')

while(1):
	path=filedialog.askopenfilename(filetypes=(("JPG", ".jpg"), ("All files", "*.*")))
	img=cv2.imread(path)
	plt.imshow(img)
	plt.show()
	test_image = cv2.imread(path)
	test_image= cv2.resize(test_image,(140,92))
	test_image = test_image.reshape(92,140,3)
	test_image = np.array(test_image)
	test_image = test_image.astype('float32')
	test_image /= 255
	test_image= np.expand_dims(test_image, axis=0)
	pa=model.predict(test_image)
	if(model.predict_classes(test_image)==[0]):
		s="BENIGN with Accuracy: "+ str(pa[0][0]*100) + "%\n"
	else:
		s="MALIGNANT with Accuracy: "+ str(pa[0][1]*100) + "%\n"

	print(s)