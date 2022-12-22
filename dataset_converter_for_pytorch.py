"""Reformat the dataset (originally intended for TensorFlow) so that it works with PyTorch instead.
Borrows heavily from this tutorial: https://medium.com/analytics-vidhya/how-to-read-tfrecords-files-in-pytorch-72763786743f
"""


#
# IMPORTS
#
# Suppress TensorFlow saying that it doesn't take advantage of my CPU architecture's AVX and AVX2 operations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Other imports
import glob
import tensorflow as tf
from PIL import Image
import cv2
from torchvision import transforms
import torch
import numpy as np
import io
from torch.utils.data import Dataset
import pdb


#
# CLASSES
# 
class FlowerDataset(Dataset):
    """PyTorch dataset class with our features extracted (ie train_ids, train_class, train_images)
    """
    def __init__(self, id, classes, image, img_height, img_width, mean, std, is_valid):
        if img_height != img_width:
            raise ValueError("Wren hasn't built this to handle differing image heights and widths yet.  Just needs to be implemented into transforms.Resize() and transforms.CenterCrop() calls")
        self.id = id
        self.classes = classes
        if type(image) == np.ndarray:
            self.image = Image.fromarray(image.astype(uint8), "RGB")
        else:
            self.image = image
        self.is_valid = is_valid    # flag for whether this is a (validation or test) dataset
        if self.is_valid == 1:
            self.aug = transforms.Compose([
                # This series of transforms intentionally does not contain data augmentation.
                transforms.ToTensor(),
                transforms.Resize(img_height),
                transforms.CenterCrop(img_height),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.aug = transforms.Compose([
                # Data augmentation does not contain flips or color changes, because chirality and color can be important in recognizing flowers.
                transforms.ToTensor(),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.RandomRotation(degrees=(-180, 180), expand=False),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.5),
                transforms.Resize(img_height),
                transforms.RandomCrop(img_height),
                transforms.Normalize(mean=mean, std=std),
            ])
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index):
        """This is what's invoked when FlowerDataset[index] is used.
        Returns a (tensor, int) tuple.  The tensor is the image, and the int is the label.
        """
        id = self.id[index]
        img = np.array(Image.open(io.BytesIO(self.image[index])))    # converting images from bytes to numpy array
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)    # Resizing all the images to (128x128)    # TODO: Why not a higher resolution?
        img = self.aug(img)    # applying image augmentation
        img = img.float()    # Convert to float32
        return torch.tensor(img, dtype=torch.float), int(self.classes[index])
    

#
# HELPER FUNCTIONS
#
def PRODUCE_DATASET():
    """Written by Wren.  Return two FlowerDataset objects: one for training, one for validation.
    """
    def _parse_image_function(example_proto):
        """Parses the input tf.Example proto using the dictionary
        train_feature_description.
        """
        return tf.io.parse_single_example(example_proto, train_feature_description)
    
    # Use the glob library to grab the file with their path names in training and validation sub-folders
    train_files = glob.glob("kaggle/input/tpu-getting-started/*/train/*.tfrec")
    val_files = glob.glob("kaggle/input/tpu-getting-started/*/val/*.tfrec")

    # See examples of what is loaded in train_files
    print(train_files[:5])

    # Collect the ids, filenames, and images in bytes in three different list variables for training and validation files
    # Create a dictionary describing the faetures
    train_feature_description = {
        'class': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }

    # Store the features in 3 different lists
    # TRAINING FEATURES
    train_ids = []
    train_class = []
    train_images = []
    for i in train_files:
        train_image_dataset = tf.data.TFRecordDataset(i)
        train_image_dataset = train_image_dataset.map(_parse_image_function)        # TODO: Should this be indented?
        ids = [str(id_features['id'].numpy())[2:-1] for id_features in train_image_dataset] # [2:-1] is done to remove b' from 1st and 'from last in train id names        # TODO: Should this be indented?
        train_ids = train_ids + ids        # TODO: Should this be indented?
        classes = [int(class_features['class'].numpy()) for class_features in train_image_dataset]        # TODO: Should this be indented?
        train_class = train_class + classes        # TODO: Should this be indented?
        images = [image_features['image'].numpy() for image_features in train_image_dataset]        # TODO: Should this be indented?
        train_images = train_images + images        # TODO: Should this be indented?
    # VALIDATION FEATURES - Wren implemented this part
    val_ids = []
    val_class = []
    val_images = []
    for i in val_files:
        val_image_dataset = tf.data.TFRecordDataset(i)
        val_image_dataset = val_image_dataset.map(_parse_image_function)    # TODO: Should this be indented?
        ids = [str(id_features['id'].numpy())[2:-1] for id_features in val_image_dataset]        # TODO: Should this be indented?
        val_ids = val_ids + ids        # TODO: Should this be indented?
        classes = [int(class_features['class'].numpy()) for class_features in val_image_dataset]        # TODO: Should this be indented?
        val_class = val_class + classes        # TODO: Should this be indented?
        images = [image_features['image'].numpy() for image_features in val_image_dataset]        # TODO: Should this be indented?
        val_images = val_images + images        # TODO: Should this be indented?


    train_dataset = FlowerDataset(id=train_ids, classes=train_class, image=train_images, img_height=128, img_width=128, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_valid=0)
    val_dataset = FlowerDataset(id=val_ids, classes=val_class, image=val_images, img_height=128, img_width=128, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_valid=1)
    return train_dataset, val_dataset


def PRODUCE_TESTING_DATASET():
    """Written by Wren.  Produce the testing dataset (not validation).
    """
    def _parse_image_function(example_proto):
        """Parses the input tf.Example proto using the dictionary
        train_feature_description.
        """
        return tf.io.parse_single_example(example_proto, train_feature_description)
    
    # Use the glob library to grab the files with their path names in testing sub-folders
    test_files = glob.glob("kaggle/input/tpu-getting-started/*/test/*.tfrec")
    
    # Collect the ids, filenames, and images in bytes in three different list variables for training and validation files
    # Create a dictionary describing the features
    train_feature_description = {
        #'class': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    
    # Store the features in a list
    # TESTING FEATURES - Wren implemented this part
    test_ids = []
    test_class = []
    test_images = []
    for i in test_files:
        test_image_dataset = tf.data.TFRecordDataset(i)
        test_image_dataset = test_image_dataset.map(_parse_image_function)    # TODO: Should this be indented?
        ids = [str(id_features['id'].numpy())[2:-1] for id_features in test_image_dataset]        # TODO: Should this be indented?
        test_ids = test_ids + ids        # TODO: Should this be indented?
        classes = [0 for class_features in test_image_dataset]        # TODO: Should this be indented?
        test_class = test_class + classes        # TODO: Should this be indented?
        images = [image_features['image'].numpy() for image_features in test_image_dataset]        # TODO: Should this be indented?
        test_images = test_images + images        # TODO: Should this be indented?
        
    test_dataset = FlowerDataset(id=test_ids, classes=test_class, image=test_images, img_height=128, img_width=128, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_valid=1)
    return test_dataset
