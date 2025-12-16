import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Load pretrained ResNet
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def get_color(image):
    avg = np.mean(image.reshape(-1,3), axis=0)
    b, g, r = avg

    if r > 150 and g < 100:
        return "Red"
    if b > 150:
        return "Blue"
    if r < 80 and g < 80 and b < 80:
        return "Black"
    if r > 200 and g > 200:
        return "White"
    return "Neutral"

def get_category_from_filename(name):
    name = name.lower()

    if "shoe" in name or "sneaker" in name or "heel" in name:
        return "shoe"

    if (
        "pant" in name
        or "jean" in name
        or "trouser" in name
        or "skirt" in name
    ):
        return "bottom"

    if "shirt" in name or "top" in name or "tshirt" in name:
        return "top"

    return "unknown"

    

def extract_embedding(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img)
    return embedding.numpy()[0]

def analyze_item(image, filename):
    return {
        "category": get_category_from_filename(filename),
        "color": get_color(image),
        "embedding": extract_embedding(image),
        "image": image,
        "name": filename
    }
