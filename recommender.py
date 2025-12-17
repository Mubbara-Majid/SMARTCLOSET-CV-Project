from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_color_harmony(color1, color2):
    color_positions = {
        'Red': 0, 'Orange': 30, 'Yellow': 60, 'Green': 120,
        'Cyan': 180, 'Blue': 240, 'Purple': 280,
        'White': -1, 'Black': -1, 'Gray': -1, 'Mixed': -1
    }
    
    pos1 = color_positions.get(color1, -1)
    pos2 = color_positions.get(color2, -1)
    
    if pos1 == -1 or pos2 == -1:
        return 0.9
    
    angle_diff = abs(pos1 - pos2)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    if angle_diff < 30:
        return 0.85
    elif 150 <= angle_diff <= 210:
        return 0.95
    elif 90 <= angle_diff <= 150:
        return 0.75
    else:
        return 0.6

# PATTERN COMPATIBILITY
def pattern_compatibility(pattern1, pattern2):
    """Check if patterns work well together"""
    busy_patterns = {'Striped', 'Patterned'}
    simple_patterns = {'Solid', 'Plain'}
    
    if pattern1 in busy_patterns and pattern2 in busy_patterns:
        return 0.3
    
    if pattern1 in simple_patterns or pattern2 in simple_patterns:
        return 1.0
    
    return 0.7

# MAIN SCORING FUNCTION
def score(item1, item2):
    """Calculate compatibility score between two items"""
    # Visual similarity
    visual_sim = cosine_similarity(
        [item1["embedding"]],
        [item2["embedding"]]
    )[0][0]
    
    # Color harmony
    color_score = calculate_color_harmony(
        item1["color"], 
        item2["color"]
    )
    
    # Pattern compatibility
    pattern_score = pattern_compatibility(
        item1.get("pattern", "Plain"),
        item2.get("pattern", "Plain")
    )
    
    # Weighted combination
    final_score = (
        0.30 * visual_sim +
        0.40 * color_score +
        0.30 * pattern_score
    )
    
    return final_score


# RECOMMENDATION ENGINE (UPDATED FOR DRESSES)
def recommend_outfit(items):
    """Find the best outfit combination (Either Top+Bottom+Shoe OR Dress+Shoe)"""
    tops = [i for i in items if i["category"] == "top"]
    bottoms = [i for i in items if i["category"] == "bottom"]
    shoes = [i for i in items if i["category"] == "shoe"]
    bodies = [i for i in items if i["category"] == "body"] # Dresses/Jumpsuits

    best_outfit = None
    best_score = -1

    # 1. Check Top + Bottom + Shoe Combinations
    for t in tops:
        for b in bottoms:
            for s in shoes:
                # Calculate average compatibility of the trio
                s1 = score(t, b)
                s2 = score(b, s)
                s3 = score(t, s)
                total = (s1 + s2 + s3) / 3 # Average score
                
                if total > best_score:
                    best_score = total
                    best_outfit = (t, b, s)

    # 2. Check Dress + Shoe Combinations
    for d in bodies:
        for s in shoes:
            # Calculate compatibility pair
            total = score(d, s)
            
            # We give a slight bonus (1.05x) to dresses because they are easier to style
            if (total * 1.05) > best_score:
                best_score = total
                best_outfit = (d, s)

    return best_outfit