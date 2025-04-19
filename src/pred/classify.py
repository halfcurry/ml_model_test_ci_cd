from typing import Any
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests

IMAGE_SHAPE = (128, 128)

def load_model() -> tf.keras.Sequential:
    """
    Loads and returns a pre-trained MobileNetV2 model for image classification.
    
    Returns:
        tf.keras.Sequential: A Keras sequential model with MobileNetV2 architecture
    """
    classifier = tf.keras.Sequential([
        hub.KerasLayer(
            "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2",
            input_shape=IMAGE_SHAPE + (3,)
        )
    ])
    return classifier


def load_labels() -> np.ndarray:
    """
    Downloads and loads ImageNet labels for classification.
    
    Returns:
        np.ndarray: Array of string labels corresponding to ImageNet classes
    """
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    return imagenet_labels


def preprocess_img(img: Image) -> np.ndarray:
    """
    Preprocesses an input image for model prediction.
    
    Args:
        img (PIL.Image): Input image to preprocess
        
    Returns:
        np.ndarray: Preprocessed image array normalized to [0,1] range
    """
    img = img.resize(IMAGE_SHAPE)
    img = np.array(img) / 255
    return img


def load_image(img_url: str) -> Image:
    """
    Downloads and loads an image from a given URL.
    
    Args:
        img_url (str): URL of the image to load
        
    Returns:
        PIL.Image: Loaded image object or None if loading fails
    """
    try:
        img = Image.open(requests.get(img_url, stream=True).raw)
        return img
    except Exception as e:
        print(e)
        print("Failed to load image from URL. Please check the URL and try again.")


def predict_image(img_original: Image) -> dict:
    """
    Makes predictions on a single image using the loaded model.
    
    Args:
        img_original (PIL.Image): Original image to classify
        
    Returns:
        dict: Dictionary containing prediction class name and probability
    """
    img = preprocess_img(img_original)
    model = load_model()
    result = model.predict(img[np.newaxis, ...])
    predicted_class = tf.math.argmax(result[0], axis=-1)
    scores = tf.nn.softmax(result[0])
    probability = np.max(scores)

    imagenet_labels = load_labels()
    predicted_class_name = imagenet_labels[predicted_class]

    return {
        "prediction": predicted_class_name,
        "probability": probability.item()
    }


def run_classifier(image: str) -> Any:
    """
    Main function to run the image classification pipeline.
    
    Args:
        image (str): URL of the image to classify
        
    Returns:
        dict: Classification results with prediction, probability and status code
              Returns None if image loading fails
    """
    img = load_image(image)
    if img is None:
        return None
    pred_results = predict_image(img)
    pred_results["status_code"] = 200
    return pred_results