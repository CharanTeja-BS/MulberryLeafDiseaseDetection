# src/features.py
import cv2
import numpy as np


def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))

    # Color features
    mean_color = img.mean(axis=(0, 1))

    # Texture features (edges)
    edges = cv2.Canny(img, 100, 200)
    edge_mean = edges.mean()

    # Final feature vector
    features = np.hstack([mean_color, edge_mean])
    return features
