from tkinter import *
from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from PIL import Image, ImageTk
import h5py
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
import np_utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model 
import plot_model
import cv2
import os
from sklearn.model_selection import train_test_split
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

K.set_image_data_format('channels_last')


model=1
window = tk.Tk()
img=Image.open('E:\\A Elecsis\\AAAAAAAA\\aaaa\\1 Breast Cancer Detection\\1.webp')
bg=ImageTk.PhotoImage(img)
window.title("BREAST CANCER DETECTION")
wd=str(window.winfo_screenwidth()-500)+"x"+str(window.winfo_screenheight()-100)
window.geometry(wd)
label=Label(window,image=bg)
label.place(x=0,y=0)

# load data
numepochs=30
batchsize=128
folder_path = './data/'
images = []
labels = []
class_label = 0

def load_images_from_folder(folder,class_label):
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder, filename))
		if img is not None:
			img = cv2.resize(img,(140,92))
			img = img.reshape(92,140,3)
			images.append(img)
			labels.append(class_label)
	class_label=class_label+1
	return class_label


# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding="same",input_shape=(92,140,3), activation='relu'))
	#model.add(Conv2D(32, (3, 3), activation='relu',padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu',padding = 'same'))
	#model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
	#model.add(Conv2D(128, (3, 3), activation='relu',padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(Dense(50, activation='relu'))
	#model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
	return model


class_label = 0
class_label = load_images_from_folder(folder_path+'benign',class_label)
class_label = load_images_from_folder(folder_path+'malignant',class_label)

Data = np.asarray(images)
Labels = np.asarray(labels)

X_train,X_test,y_train,y_test=train_test_split(Data,Labels,test_size=0.2,random_state=2)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

tr="train data shape:"+"\n"
tr=tr+"test data shape:"+"\n"
tr=tr+str(X_test.shape)+"\n"
tr=tr+"train label shape:"+"\n"
tr=tr+str(y_train.shape)+"\n"
tr=tr+"test label shape:"+"\n"
tr=tr+str(y_test.shape)+"\n"

def training(X_train, y_train,X_test, y_test,tr):
	global hist
	# build the model
	model = larger_model()
	# Fit the model
	hist=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=128)
	model.summary()
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=1,batch_size=batchsize)
	model.save('28april.h5')
	print("Deep Net Accuracy: %.2f%%" % (scores[1]*100))
	
	#create text field
	greetings_disp =tk.Text(master=window,height=20,width=120, fg="midnight blue")
	greetings_disp.grid(column=0,row=3)
	ly= ""
	for layer in model.layers:
		ly=ly+str(layer.name)+"       "  + "				layer input: "+str(layer.input)+"\n"#str(layer.inbound_nodes)+str(layer.outbound_nodes)      "			<<--inputs-->> \n\n" + tr+
	greetings_disp.insert(tk.END ,"			<<--LAYER ARCHITECTURE-->> \n\n" +ly+"\n\n NETWORK is trained with Accuracy  of"+ str(scores[1]*100)+"%")
	return model
	
def graphh():

	# visualizing losses and accuracy
	train_loss=hist.history['loss']
	val_loss=hist.history['val_loss']
	train_acc=hist.history['accuracy']
	val_acc=hist.history['val_accuracy']
	xc=range(30)

	plt.figure(1,figsize=(10,5))
	plt.subplot(121)
	plt.plot(xc,train_loss)
	plt.plot(xc,val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train','val'])
	plt.style.use(['classic'])

	#plt.figure(2,figsize=(7,5))
	plt.subplot(122)
	plt.plot(xc,train_acc)
	plt.plot(xc,val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)

	plt.legend(['train','val'],loc=4)
	plt.style.use(['classic'])
	plt.show()
	
def test_test(model):
	test_image = X_test[0:1]
	pa=model.predict(test_image)
	if(model.predict_classes(test_image)==[0]):
		s="BENIGN with Accuracy: " + str(pa[0][0]*100) + "%\n"
	else:
		s="MALIGNANT with Accuracy: "+ str(pa[0][1]*100) + "%\n"

	return s

def test_random(model,path):
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
	return s

def b_test_test():
	greetings=test_test(model)
	#create text field
	greetings_disp =tk.Text(master=window,height=1,width=45 ,fg="midnight blue")
	greetings_disp.grid(column=0,row=6)
	greetings_disp.insert(tk.END , greetings)

def b_random_test_show():
	global path1
	path=filedialog.askopenfilename(filetypes=(("JPG", ".jpg"), ("All files", "*.*")))
	path1=path
	img=cv2.imread(path1)
	plt.imshow(img)
	plt.show()
	#greetings_disp =tk.Text(master=window,height=0,width=0 ,fg="midnight blue")
	#greetings_disp.grid(column=0,row=10)
	#greetings_disp.insert(tk.END , greetings)

def b_random_test():
	path=filedialog.askopenfilename(filetypes=(("JPG", ".jpg"), ("All files", "*.*")))
	greetings=test_random(model,path)
	#create text field
	greetings_disp =tk.Text(master=window,height=1,width=45 ,fg="midnight blue")
	greetings_disp.grid(column=0,row=12)
	greetings_disp.insert(tk.END , greetings)
	img=cv2.imread(path)
	plt.imshow(img)
	plt.show()

def b_training():
	global model
	model=training(X_train, y_train,X_test, y_test,tr)




labelfont=('Times New Roman', 50, 'bold')
label1 = tk.Label(text="   Breast Cancer Detection      ", anchor='n',  font=labelfont , fg="midnight blue" , bg="red")
label1.grid(column=0,row=0)




button1 = tk.Button(font=('Times New Roman', 10),text="Start Training" , command= b_training , bg="pink")
button1.grid(column=0,row=2, padx=10,pady=10)

button2 = tk.Button(font=('Times New Roman', 10),text="Test an Image from Dataset" , command=b_test_test , bg="pink")
button2.grid(column=0,row=5,padx=10,pady=10)


button4 = tk.Button(font=('Times New Roman', 10),text="Select an Image for Testing" , command= b_random_test, bg="pink")
button4.grid(column=0,row=11,padx=10,pady=10)

button5 = tk.Button(font=('Times New Roman', 10),text="See Loss and Accuracy plots" , command= graphh, bg="pink")
button5.grid(column=0,row=13,padx=10,pady=10)

window.mainloop()






