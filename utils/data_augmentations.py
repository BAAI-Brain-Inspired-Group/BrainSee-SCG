import numpy as np
import cv2
from skimage.morphology import dilation, erosion
import random
import sys
from annotator.hed import HEDdetector, nms

def random_threshold(edges, min_threshold=0.1, max_threshold=0.9):
    # threshold = np.random.uniform(min_threshold, max_threshold)
    # return (edges > threshold).astype(np.uint8)
    min_threshold, max_threshold = edges.max()*min_threshold, edges.max()*max_threshold
    mask = (edges >= min_threshold) & (edges <= max_threshold)
    return  np.where(mask, edges, 0).astype(np.uint8)

def scribble(edge, drawing_threshold=10):
    binary_image = np.where(edge > drawing_threshold, 0, 255).astype(np.uint8)
    return binary_image

def random_mask(edges, max_percentage=0.2):
    mask_percentage = np.random.uniform(0, max_percentage)
    mask = np.random.rand(*edges.shape) < mask_percentage
    edges[mask] = 255
    return edges

def random_morphological_transform(edges):
    transformations = [erosion]# [dilation, erosion]
    transform = np.random.choice(transformations)
    return transform(edges)

def random_non_max_suppression(edges, s=10, t=0.3):
# def nms(x, t, s):
    x = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    t = y.max()*t
    z[y > t] = 255
    return z    
    # # This is a simplified version. A proper implementation would need more details.
    # return cv2.dilate(edges, None) - edges

def augment_edges(edge):
    # edges = random_threshold(edge)
    # edges = random_morphological_transform(edges)
    edges = random_mask(edge, random.uniform(0.2, 0.5))
    edges = random_threshold(edges, min_threshold=0.1, max_threshold=0.3)
    edges = scribble(edges, drawing_threshold=0)
    edges = random_morphological_transform(edges)
    # edges = random_non_max_suppression(edges)
    return edges

def generate_scribble(image):
    detected_map = nms(image, 127, 3.0)
    detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
    detected_map[detected_map > 4] = 255
    detected_map[detected_map < 255] = 0
    
    return 255-detected_map

# edges = np.load('tmp.npy')
# edge1 = augment_edges(edges)
# edge2 = generate_scribble(edges)

# a=1