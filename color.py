import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter



def get_dominant_color(image, k=4):
    # Convert image to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))
    
    # Perform k-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the most common cluster
    counts = Counter(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
    
    return tuple(dominant_color.astype(int))

def compare_colors(color1, color2):
    # Calculate Euclidean distance between the two colors
    return np.linalg.norm(np.array(color1) - np.array(color2))



