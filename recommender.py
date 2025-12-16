from sklearn.metrics.pairwise import cosine_similarity

def score(item1, item2):
    sim = cosine_similarity(
        [item1["embedding"]],
        [item2["embedding"]]
    )[0][0]

    rule = 0
    if item1["color"] != item2["color"]:
        rule += 0.3

    return sim + rule

def recommend_outfit(items):
    tops = [i for i in items if i["category"] == "top"]
    bottoms = [i for i in items if i["category"] == "bottom"]
    shoes = [i for i in items if i["category"] == "shoe"]

    best = None
    best_score = -1

    for t in tops:
        for b in bottoms:
            for s in shoes:
                total = score(t, b) + score(b, s)
                if total > best_score:
                    best_score = total
                    best = (t, b, s)

    return best
