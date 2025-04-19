from src.pred.classify import load_model, load_labels, preprocess_img, load_image, predict_image
from PIL import Image
import numpy as np

# Test constants
VALID_IMAGE_URL = "https://images.dog.ceo/breeds/spitz-japanese/tofu.jpg"
INVALID_IMAGE_URL = "https://invalid-url-that-does-not-exist.jpg"

def test_load_model():
    model = load_model()
    assert model is not None
    assert isinstance(model, object)

def test_load_labels():
    labels = load_labels()
    assert labels is not None
    assert len(labels) > 0
    assert isinstance(labels, np.ndarray)

def test_preprocess_img():
    test_img = Image.new('RGB', (200, 200))
    processed = preprocess_img(test_img)
    assert processed.shape == (128, 128, 3)
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0

def test_load_valid_image():
    img = load_image(VALID_IMAGE_URL)
    assert img is not None
    assert isinstance(img, Image.Image)

def test_load_invalid_image():
    img = load_image(INVALID_IMAGE_URL)
    assert img is None

def test_predict_image():
    test_img = Image.new('RGB', (128, 128))
    prediction = predict_image(test_img)
    assert isinstance(prediction, dict)
    assert 'prediction' in prediction
    assert 'probability' in prediction
    assert isinstance(prediction['probability'], float)