
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from sklearn.cluster import KMeans

# Load pretrained ResNet
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 1. COLOR DETECTION (K-MEANS)
def get_dominant_colors(image, n_colors=3):
    """Extract dominant colors using K-means clustering"""
    pixels = image.reshape(-1, 3).astype(float)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    color_info = []
    for i in range(n_colors):
        count = np.sum(labels == i)
        percentage = count / len(labels)
        color_info.append({
            'rgb': colors[i],
            'percentage': percentage,
            'name': rgb_to_color_name(colors[i])
        })
    
    return sorted(color_info, key=lambda x: x['percentage'], reverse=True)

def rgb_to_color_name(rgb):
    """Convert RGB to color name using HSV color space"""
    rgb_normalized = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    
    if s < 30:
        if v < 50:
            return "Black"
        elif v > 200:
            return "White"
        else:
            return "Gray"
    
    if h < 10 or h > 160:
        return "Red"
    elif h < 25:
        return "Orange"
    elif h < 40:
        return "Yellow"
    elif h < 80:
        return "Green"
    elif h < 95:
        return "Cyan"
    elif h < 130:
        return "Blue"
    elif h < 160:
        return "Purple"
    
    return "Mixed"

def get_color(image):
    """Get primary color name"""
    colors = get_dominant_colors(image, n_colors=1)
    return colors[0]['name']

# 2. PATTERN DETECTION
def detect_pattern(image):
    """Detect clothing patterns using texture analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=30, maxLineGap=10)
    has_lines = lines is not None and len(lines) > 5
    
    if std < 15 and edge_density < 0.05:
        return "Solid"
    elif has_lines and edge_density > 0.15:
        return "Striped"
    elif edge_density > 0.3:
        return "Patterned"
    elif std > 40:
        return "Textured"
    else:
        return "Plain"


# 3. CATEGORY DETECTION
def classify_category_cv(image):
    """Use computer vision to classify garment category"""
    height, width = image.shape[:2]
    aspect_ratio = height / width
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    vertical_edges = np.sum(np.abs(sobelx))
    horizontal_edges = np.sum(np.abs(sobely))
    edge_ratio = vertical_edges / (horizontal_edges + 1e-5)
    
    if aspect_ratio < 0.8:
        return "shoe"
    elif aspect_ratio < 1.3:
        return "top"
    else:
        if edge_ratio > 1.2:
            return "bottom"
        else:
            return "top"

def get_category_from_filename(name):
    """Fallback: filename-based detection"""
    name = name.lower()
    
    if "shoe" in name or "sneaker" in name or "heel" in name or "boot" in name:
        return "shoe"
    if "pant" in name or "jean" in name or "trouser" in name or "skirt" in name or "short" in name:
        return "bottom"
    if "shirt" in name or "top" in name or "tshirt" in name or "blouse" in name:
        return "top"
    if "dress" in name:
        return "dress"
    if "jacket" in name or "coat" in name:
        return "jacket"
    
    return "unknown"

def get_category(image, filename):
    """Hybrid: Use CV + filename"""
    cv_category = classify_category_cv(image)
    filename_category = get_category_from_filename(filename)
    
    if filename_category != "unknown":
        return filename_category
    
    return cv_category


# FEATURE EXTRACTION
def extract_embedding(image):
    """Extract deep features using ResNet"""
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img)
    return embedding.numpy()[0]


# MAIN ANALYSIS FUNCTION
def analyze_item(image, filename):
    """Complete analysis of clothing item"""
    colors = get_dominant_colors(image, n_colors=3)
    category = get_category(image, filename)
    pattern = detect_pattern(image)
    embedding = extract_embedding(image)
    
    return {
        "category": category,
        "color": colors[0]['name'],
        "colors": colors,
        "pattern": pattern,
        "embedding": embedding,
        "image": image,
        "name": filename
    }

