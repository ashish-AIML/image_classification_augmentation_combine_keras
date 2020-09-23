# Combination of Image Augmentation and Image Classification

This is summary readme for combination and integration of image augmentation and image classification using Keras network


[train](train.py) consists of 3 parts:
1. Image Augmentation: 
```
aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
```
Give an image of each class and 100 augmented images are generated with respect to the above augmentation methods

2. Building Neural Network: A simple neural network is build with 2 Conv layers

3. Post processing is done on the data after augmentation and building neural network. The final training is performed. 

![training graph](https://github.com/cudanexus/ashish/blob/master/image_classification_augmentation_combine_keras/results/plot.png)


The model gets saved, and then run [test](test.py) to verify the predictions. 


---
## License & Copyright

@ Ashish & Team

***
