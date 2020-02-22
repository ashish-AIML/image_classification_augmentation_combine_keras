'''
PYTHON CODE TO TEST THE TRAINED KERAS MODEL
'''
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2

# load the image
image = cv2.imread('M:\\Tericsoft\\Teric_Research\\keras-data-augmentation\\dogs_vs_cats_small\\cats\\cats_00001.jpg')
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('M:\\Tericsoft\\Teric_Research\\image-classification-keras\\cats_and_dogs.model')

# classify the input image
(dog, cat) = model.predict(image)[0]

# build the label
label = "cat" if cat > dog else "dog"
proba = cat if cat > dog else dog
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)