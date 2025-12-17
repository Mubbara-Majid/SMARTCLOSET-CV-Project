import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from sklearn.cluster import KMeans
import os

# ==========================================
# 1. MODEL LOADING & CONFIGURATION
# ==========================================

# IMPORTANT: The order here must match EXACTLY the alphabetical order 
# of folders in your training data, unless you defined a specific order.
# usually ImageFolder uses alphabetical: ['DRESS', 'JEANS', 'SKIRT', 'T-SHIRT']
# If you are sure it is the order you pasted:
CLASSES = ['JEANS', 'T-SHIRT', 'DRESS', 'SKIRT']

# MAP: Model Output -> App Category (top/bottom/shoe)
# The recommender needs 'top', 'bottom', 'shoe' to work.
CATEGORY_MAP = {
    'JEANS': 'bottom',
    'SKIRT': 'bottom',
    'T-SHIRT': 'top',
    'DRESS': 'body',   # Note: The current recommender might ignore dresses
}

# Define standard transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_custom_model():
    """Loads the custom smart_closet_model.pth if available."""
    print("Loading Smart Closet Model...")
    
    # Initialize ResNet18
    model = models.resnet18(pretrained=False)
    
    # Adjust final layer for 4 classes
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))
    
    try:
        if os.path.exists('smart_closet_model.pth'):
            model.load_state_dict(torch.load('smart_closet_model.pth', map_location=torch.device('cpu')))
            print(f"✅ Custom model loaded for classes: {CLASSES}")
        else:
            raise FileNotFoundError("Model file not found")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not load custom model ({e}).")
        print("   Falling back to generic ResNet18.")
        model = models.resnet18(pretrained=True)

    model.eval()
    return model

model = load_custom_model()


# ==========================================
# 2. COLOR DETECTION (K-MEANS)
# ==========================================

def get_dominant_colors(image, n_colors=3):
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
    rgb_normalized = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    
    if s < 30:
        if v < 50: return "Black"
        elif v > 200: return "White"
        else: return "Gray"
    
    if h < 10 or h > 160: return "Red"
    elif h < 25: return "Orange"
    elif h < 40: return "Yellow"
    elif h < 80: return "Green"
    elif h < 95: return "Cyan"
    elif h < 130: return "Blue"
    elif h < 160: return "Purple"
    
    return "Mixed"

def get_color(image):
    colors = get_dominant_colors(image, n_colors=1)
    return colors[0]['name']


# ==========================================
# 3. PATTERN DETECTION
# ==========================================

def detect_pattern(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=30, maxLineGap=10)
    has_lines = lines is not None and len(lines) > 5
    
    if std < 15 and edge_density < 0.05: return "Solid"
    elif has_lines and edge_density > 0.15: return "Striped"
    elif edge_density > 0.3: return "Patterned"
    elif std > 40: return "Textured"
    else: return "Plain"


# ==========================================
# 4. CATEGORY DETECTION (UPDATED)
# ==========================================

def classify_category_model(image):
    """Predict category using the custom trained model and map to App Category"""
    try:
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Check if we are running the generic model fallback
            if outputs.shape[1] != len(CLASSES):
                return None 

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Retrieve the model class (e.g., 'JEANS')
            model_label = CLASSES[predicted.item()]
            
            # Map 'JEANS' -> 'bottom'
            app_category = CATEGORY_MAP.get(model_label, "unknown")
            
            print(f"Model Prediction: {model_label} ({confidence.item():.2f}) -> Mapped to: {app_category}")
            return app_category

    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def classify_category_cv(image):
    """Fallback CV logic (Useful for Shoes since model doesn't know them)"""
    height, width = image.shape[:2]
    aspect_ratio = height / width
    
    if aspect_ratio < 0.8: return "shoe"
    elif aspect_ratio < 1.3: return "top"
    else: return "bottom"

def get_category_from_filename(name):
    name = name.lower()
    if "shoe" in name or "sneaker" in name or "heel" in name: return "shoe"
    if "jean" in name or "skirt" in name or "pant" in name: return "bottom"
    if "shirt" in name or "top" in name: return "top"
    return "unknown"

def get_category(image, filename):
    # 1. Filename (Strongest signal for shoes)
    fname_cat = get_category_from_filename(filename)
    if fname_cat != "unknown":
        return fname_cat
    
    # 2. Custom Model
    model_cat = classify_category_model(image)
    if model_cat and model_cat != "unknown":
        return model_cat

    # 3. CV Fallback (Catch-all for shoes if filename failed)
    return classify_category_cv(image)


# ==========================================
# 5. FEATURE EXTRACTION
# ==========================================

def extract_embedding(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img)
    return embedding.numpy()[0]

def analyze_item(image, filename):
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