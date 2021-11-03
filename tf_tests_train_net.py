# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:30:23 2021

@author: Idan Tal
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:04:16 2021


@author: Idan Tal
"""
# %%
import os
import numpy as np
import pandas as pd
# import time

import PIL.Image as Image
import matplotlib.pylab as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub

# Any compatible image classifier model from tfhub.dev will work here.
imgsz = 299
IMAGE_SHAPE = (imgsz, imgsz)
m = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/classification/4", output_shape=[1001])])
m.build([None, imgsz, imgsz, 3])  # Batch input shape.
classifier = tf.keras.Sequential([hub.KerasLayer(m, input_shape=IMAGE_SHAPE+(3,))])

# load an image
base_dir = r"K:\\slowFluctuationsNHP\\data\\FV_images\\"
save_dir = r"K:\\slowFluctuationsNHP\\data\\FV_images\\figures\\image_classification_inception\\"
for ii in np.arange(1,81):
    filename = str(ii) + ".jpg"
    img_fn = os.path.join(base_dir, filename)
    # check if the image exists
    if os.path.isfile(img_fn):
        img1 = Image.open(img_fn).resize(IMAGE_SHAPE)    
    else:
        continue
    
    
    img1 = np.array(img1)/255.0
    img1.shape
    
    # %% Add a batch dimension, and pass the image to the model.
    
    result = classifier.predict(img1[np.newaxis, ...])
    result.shape
    
    # The result is a 1001 element vector of logits, rating the probability of each class for the image.
    
    predicted_class = np.argsort(-result[0],axis=-1)
    
    # Decode the prediction
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    
    predicted_class_name = []
    for jj in np.arange(0,5):
        predicted_class_name.append(imagenet_labels[predicted_class[jj]])
    

    predicted_prob = result[0][predicted_class[0:5]]
    
    
    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 
    
    df = pd.DataFrame(list(zip(predicted_class_name, predicted_prob*10)), columns =['Label', 'prob']) 
    
    f = plt.figure()    
    f.add_subplot(131)
    plt.imshow(img1)
    plt.axis('off')
    # Plot the total crashes
    f.add_subplot(133, aspect=20)
    plt.xlim(0,100)
    sns.set_color_codes("pastel")
    sns.barplot(x="prob", y="Label", data=df)

    
    save_fn = os.path.join(save_dir, filename)
    
    plt.savefig(save_fn)
    plt.close()