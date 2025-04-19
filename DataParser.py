import numpy as np
from PIL import Image
import os
import random
import math

def getImagesFromFile(img_dir, percentage):
    """
    Returns a numpy array of images randomly selected from the given directory.
    The number of images is determined by the percentage given.
    """
    
    all_image_files = [img for img in os.listdir(img_dir) if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    # randomly select only a percentage of the files
    random.shuffle(all_image_files)
    selected_image_files = all_image_files[:math.floor(len(all_image_files) * percentage)]

    # get the images in an appropriate format
    images = []
    for filename in selected_image_files:
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert("RGB")

        img = transformImage(img)

        images.append(img)

    return images


def transformImage(img):
    """
    Transform the image into a 64x64 numpy array
    """
    img = img.resize((64, 64))
    img = np.array(img) / 255

    return img

def getTrainData(train_path="data/train", percentage = 0.2):
    """
    Get the train images as numpy arrays
    and their labels: 0 for cat, 1 for dog
    as numpy arrays
    """

    train_cat_images = getImagesFromFile(train_path + "/cats", percentage)
    train_dog_images = getImagesFromFile(train_path + "/dogs", percentage)
    train_set = np.array(train_cat_images + train_dog_images)

    labels = np.array([0] * len(train_cat_images) + [1] * len(train_dog_images))

    # shuffle the samples and their labels randomly
    permutation = np.random.permutation(len(train_set))
    train_set = train_set[permutation]
    labels = labels[permutation]

    return train_set, labels

def getTestData(test_path="data/test", percentage = 0.2):
    """
    Get the test images as numpy arrays
    and their labels: 0 for cat, 1 for dog
    as numpy arrays
    """

    test_cat_images = getImagesFromFile(test_path + "/cats", percentage)
    test_dog_images = getImagesFromFile(test_path + "/dogs", percentage)
    test_set = np.array(test_cat_images + test_dog_images)

    labels = np.array([0] * len(test_cat_images) + [1] * len(test_dog_images))

    # shuffle the samples and their labels randomly
    permutation = np.random.permutation(len(test_set))
    test_set = test_set[permutation]
    labels = labels[permutation]

    return test_set, labels



