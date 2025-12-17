
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# COLOR THEORY
def calculate_color_harmony(color1, color2):
    """Calculate how well two colors work together"""
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


# RECOMMENDATION ENGINE
def recommend_outfit(items):
    """Find the best outfit combination"""
    tops = [i for i in items if i["category"] == "top"]
    bottoms = [i for i in items if i["category"] == "bottom"]
    shoes = [i for i in items if i["category"] == "shoe"]

    best = None
    best_score = -1

    for t in tops:
        for b in bottoms:
            for s in shoes:
                total = score(t, b) + score(b, s) + score(t, s)
                if total > best_score:
                    best_score = total
                    best = (t, b, s)

    return best
