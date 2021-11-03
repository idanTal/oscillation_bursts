# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:04:16 2021


@author: Idan Tal
"""
# %%
import os
import numpy as np
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub

# Use hub.KerasLayer to load a model from TensorFlow Hub. 
IMAGE_SHAPE = (331, 331)
m = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_large/classification/4", output_shape=[1001])])
m.build([None, 331, 331, 3])  # Batch input shape.
classifier = tf.keras.Sequential([hub.KerasLayer(m, input_shape=IMAGE_SHAPE+(3,))])

# %% load an image
base_dir = r"K:\\slowFluctuationsNHP\\data\\experimentPrograms\\FVNatImages_13\\library\\images\\"
save_dir = r"K:\\slowFluctuationsNHP\\output\\figures\\image_classification_nasnetLarge\\"
for ii in np.arange(1,221):
    filename = str(ii) + ".jpg"
    img_fn = os.path.join(base_dir, filename)
    # check if the image exists
    if os.path.isfile(img_fn):
        img1 = Image.open(img_fn).resize(IMAGE_SHAPE)    
    else:
        continue
    
    
    img1 = np.array(img1)/255.0
    img1.shape
    
    # Add a batch dimension, and pass the image to the model.  
    result = classifier.predict(img1[np.newaxis, ...])
    result.shape
    
    
    # find the top class ID 
    predicted_class = np.argmax(result[0], axis=-1)
    predicted_class
    
    # Decode the prediction
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    
    # plot the image and the prediction
    plt.imshow(img1)
    plt.axis('off')
    predicted_class_name = imagenet_labels[predicted_class]
    _ = plt.title("Prediction: " + predicted_class_name.title())
    
    save_fn = os.path.join(save_dir, filename)
    
    plt.savefig(save_fn)
    plt.close()