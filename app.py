import os, base64, json, urllib.request, urllib.parse, hashlib, secrets
import numpy as np
import cv2
from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

# Food-101 class name → display name (underscore → space, title case)
# classes loaded from labels.json at startup

def visual_classify(cv_img):
    hsv   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h, w  = cv_img.shape[:2]
    total = h * w

    yellow_pct        = float(np.sum(cv2.inRange(hsv,(15,80,100),(35,255,255))>0))/total
    red_pct           = float(np.sum(cv2.inRange(hsv,(0,100,80),(10,255,255))>0)+np.sum(cv2.inRange(hsv,(165,100,80),(180,255,255))>0))/total
    brown_pct         = float(np.sum(cv2.inRange(hsv,(10,40,40),(25,200,180))>0))/total
    green_pct         = float(np.sum(cv2.inRange(hsv,(36,40,40),(85,255,255))>0))/total
    white_pct         = float(np.sum(cv2.inRange(hsv,(0,0,180),(180,40,255))>0))/total
    orange_pct        = float(np.sum(cv2.inRange(hsv,(10,100,100),(20,255,255))>0))/total
    deep_red_pct      = float(np.sum(cv2.inRange(hsv,(0,150,80),(8,255,255))>0)+np.sum(cv2.inRange(hsv,(168,150,80),(180,255,255))>0))/total
    bright_orange_pct = float(np.sum(cv2.inRange(hsv,(8,180,150),(18,255,255))>0))/total
    ygreen_pct        = float(np.sum(cv2.inRange(hsv,(30,100,100),(50,255,200))>0))/total

    # Fruits — tight thresholds
    if deep_red_pct > 0.40 and brown_pct < 0.10 and yellow_pct < 0.12:
        return "apple", 0.85
    if bright_orange_pct > 0.40 and brown_pct < 0.10 and yellow_pct < 0.15:
        return "orange", 0.82
    if yellow_pct > 0.50 and red_pct < 0.06 and brown_pct < 0.10:
        return "banana", 0.82
    if ygreen_pct > 0.45 and red_pct < 0.06 and brown_pct < 0.10:
        return "apple", 0.75

    # Dishes
    if yellow_pct > 0.28 and brown_pct > 0.10 and red_pct < 0.12 and green_pct < 0.12:
        return "french fries", 0.72
    if yellow_pct > 0.22 and orange_pct > 0.08 and brown_pct > 0.08 and green_pct < 0.14:
        return "fish and chips", 0.70
    if white_pct > 0.25 and orange_pct > 0.03 and green_pct > 0.03:
        return "fried rice", 0.75
    if yellow_pct > 0.18 and brown_pct > 0.08 and red_pct > 0.05:
        return "nachos", 0.68
    if green_pct > 0.30:
        return "salad", 0.65
    if red_pct > 0.18 and yellow_pct > 0.12:
        return "pizza", 0.62
    return "mixed dish", 0.35


# Map COCO class IDs → use visual classifier directly (Food-101 doesn't know raw fruits/veg)
VISUAL_COCO_IDS = {
    47: "apple",
    49: "orange",
    46: "banana",
    50: "broccoli",
    51: "carrot",
}

#TensorFlow - Food-101 trained model (81.9% Top-1, 95.9% Top-5)

TF_OK=False; tf_model=None; FOOD101_CLASSES=[]
try:
    import tensorflow as tf
    _model_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "food101_efficientnetv2b0.keras")
    _labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "food101_classes.json")
    if not os.path.exists(_model_path):
        raise FileNotFoundError("food101_efficientnetv2b0.keras not found — place it next to app.py")
    if not os.path.exists(_labels_path):
        raise FileNotFoundError("food101_classes.json not found — place it next to app.py")
    with open(_labels_path) as f:
        FOOD101_CLASSES = json.load(f)
    print(f"⏳ Loading Food-101 model ({len(FOOD101_CLASSES)} classes)…")
    tf_model = tf.keras.models.load_model(_model_path)
    TF_OK = True
    print(f"Food-101 EfficientNetV2B0 ready  (Top-1: 81.9%  Top-5: 95.9%)")
except FileNotFoundError as e:
    print(f"{e}  →  using OpenCV visual classifier only")
except Exception as e:
    print(f"TF model load failed: {e}")

#YOLOv8 
YOLO_OK=False; yolo_model=None
try:
    from ultralytics import YOLO
    yolo_model=YOLO("yolov8s.pt"); YOLO_OK=True
    print("YOLOv8s ready")
except Exception as e:
    print(f"YOLOv8: {e}")

FOOD_CLASS_IDS={46,47,48,49,50,51,52,53,54,55}


#  PORTION ESTIMATION ENGINE
#  Uses reference objects detected in the image to estimate actual gram weight.
#  Priority: plate > utensil > hand > cup > food bbox fraction


# Real-world reference sizes (in mm)
REFERENCE_SIZES_MM = {
    # Plates
    "dinner plate":    254,   # standard dinner plate diameter
    "side plate":      190,   # side / salad plate
    "bowl":            150,   # average bowl diameter
    # Utensils
    "fork":            190,   # standard fork length
    "knife":           210,   # standard table knife length
    "spoon":           175,   # tablespoon / dessert spoon
    "teaspoon":        125,
    "chopsticks":      240,
    # Hand
    "hand":            180,   # average adult hand length (palm base to fingertip)
    # Cups / glasses
    "cup":             80,    # standard mug diameter
    "glass":           75,    # glass diameter
    "can":             66,    # standard drink can diameter
}

# COCO class IDs for reference objects (YOLOv8 detects these)
REFERENCE_COCO_IDS = {
    # Kitchenware
    40: ("fork",    "utensil",  190),
    41: ("knife",   "utensil",  210),
    42: ("spoon",   "utensil",  175),
    43: ("bowl",    "plate",    150),
    44: ("cup",     "cup",       80),
    # People / hands (COCO detects person bbox — use as hand proxy)
    0:  ("person",  "hand",     180),
    # Bottles / cans
    39: ("bottle",  "can",       66),
    75: ("remote",  None,       None),   # skip
}

# Typical serving weights per food (grams) — used as base when no reference found
TYPICAL_SERVING_G = {
    "pizza":           150,   # 1 large slice
    "hamburger":       200,   # standard burger
    "cheeseburger":    210,
    "hot dog":         130,
    "sandwich":        200,
    "burrito":         250,
    "tacos":           100,   # 1 taco
    "nachos":          150,
    "loaded nachos":   300,
    "loaded fries":    280,
    "french fries":    150,
    "pasta":           250,
    "rice":            180,
    "fried rice":      220,
    "sushi":           150,   # ~6 pieces
    "ramen":           400,   # bowl
    "soup":            350,
    "salad":           200,
    "caesar salad":    200,
    "steak":           220,
    "chicken":         180,
    "chicken wings":   250,
    "salmon":          180,
    "egg":             100,   # 2 eggs
    "omelette":        150,
    "waffle":          130,
    "pancake":         120,
    "donut":            75,
    "donuts":           75,
    "cake":            120,   # 1 slice
    "chocolate cake":  120,
    "ice cream":       150,
    "bread":            80,   # 2 slices
    "bagel":           100,
    "croissant":        60,
    "banana":          120,
    "apple":           182,
    "orange":          131,
    "broccoli":        150,
    "mashed potato":   200,
    "default":         150,
}

def estimate_portion(cv_img, food_bbox, food_label, yolo_results=None):
    """
    Estimate actual portion weight (grams) using:
    1. Reference objects detected by YOLO (plate, fork, spoon, cup, hand)
    2. Food bounding box relative to image size
    3. Food-specific typical serving as fallback

    Returns: (estimated_grams, method_used, confidence)
    """
    h, w = cv_img.shape[:2]
    img_area = h * w
    fx1, fy1, fx2, fy2 = food_bbox
    food_w   = fx2 - fx1
    food_h   = fy2 - fy1
    food_area = food_w * food_h

    best_ref       = None   # (ref_name, ref_px_size, real_mm, ref_type)
    ref_confidence = 0.0

    #scan YOLO results for reference objects
    if yolo_results is not None:
        for box in yolo_results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in REFERENCE_COCO_IDS:
                continue
            ref_name, ref_type, real_mm = REFERENCE_COCO_IDS[cls_id]
            if ref_type is None or real_mm is None:
                continue
            conf = float(box.conf[0])
            if conf < 0.30:
                continue

            rx1, ry1, rx2, ry2 = map(int, box.xyxy[0].tolist())
            ref_px = max(rx2 - rx1, ry2 - ry1)   # largest dimension in pixels

            # Prefer higher-confidence and larger references
            priority = {"plate": 3, "utensil": 2, "cup": 2, "hand": 1}
            score = conf * priority.get(ref_type, 1)

            if best_ref is None or score > ref_confidence:
                best_ref       = (ref_name, ref_px, real_mm, ref_type, rx1, ry1, rx2, ry2)
                ref_confidence = score

    #if we have a reference, compute pixels-per-mm 
    if best_ref is not None:
        ref_name, ref_px, real_mm, ref_type, rx1, ry1, rx2, ry2 = best_ref
        px_per_mm = ref_px / real_mm

        # Estimate food dimensions in mm
        food_mm_w = food_w / px_per_mm
        food_mm_h = food_h / px_per_mm

        # Estimate food volume using food-specific depth assumptions
        depth_mm = _estimate_depth_mm(food_label, food_mm_w, food_mm_h)

        # Volume in cm³ then grams (density ≈ 1.0 g/cm³ for most foods)
        vol_cm3 = (food_mm_w / 10) * (food_mm_h / 10) * (depth_mm / 10)
        density = _food_density(food_label)
        estimated_g = round(vol_cm3 * density)

        # Clamp to sane range
        min_g = TYPICAL_SERVING_G.get(food_label.lower(), TYPICAL_SERVING_G["default"]) * 0.2
        max_g = TYPICAL_SERVING_G.get(food_label.lower(), TYPICAL_SERVING_G["default"]) * 3.5
        estimated_g = max(min_g, min(max_g, estimated_g))

        method = f"reference object: {ref_name} ({real_mm}mm real-world)"
        print(f"  Portion [{food_label}]: {ref_name} ref → {food_mm_w:.0f}×{food_mm_h:.0f}mm → {estimated_g:.0f}g")
        return round(estimated_g), method, min(0.75, ref_confidence)

    #bbox fraction of image as proxy
    area_fraction = food_area / img_area

    # A dinner-plate-sized food filling 40% of frame ≈ full plate portion
    # Scale typical serving by how much of the frame the food occupies
    base_g = TYPICAL_SERVING_G.get(food_label.lower(), TYPICAL_SERVING_G["default"])

    if area_fraction > 0.50:
        scale = 1.3    # very large / close-up
    elif area_fraction > 0.30:
        scale = 1.0    # full plate visible
    elif area_fraction > 0.15:
        scale = 0.75   # partial plate
    elif area_fraction > 0.05:
        scale = 0.55   # small portion or far away
    else:
        scale = 0.40   # tiny in frame

    estimated_g = round(base_g * scale)
    method = f"bbox area ({area_fraction*100:.1f}% of frame) → scaled typical serving"
    print(f"  Portion [{food_label}]: no reference, bbox={area_fraction*100:.1f}% → {estimated_g}g")
    return estimated_g, method, 0.35


def _estimate_depth_mm(food_label: str, width_mm: float, height_mm: float) -> float:
    """Estimate food depth (thickness) in mm based on food type."""
    lbl = food_label.lower()
    # Flat foods
    if any(x in lbl for x in ["pizza", "pancake", "waffle", "crepe", "flatbread"]):
        return 15.0
    # Sandwich / burger (stacked)
    if any(x in lbl for x in ["burger", "sandwich", "burger", "wrap", "burrito"]):
        return 80.0
    # Bowls (soup, ramen, rice, salad)
    if any(x in lbl for x in ["soup", "ramen", "bowl", "salad", "rice", "fried rice", "noodle"]):
        return 60.0
    # Tall / thick items
    if any(x in lbl for x in ["cake", "steak", "chicken", "meat", "fillet"]):
        return 40.0
    # Small / thin
    if any(x in lbl for x in ["sushi", "taco", "nacho", "chip", "fries", "french"]):
        return 20.0
    # Fruit / veg (roughly spherical — use half width as depth)
    if any(x in lbl for x in ["apple", "orange", "banana", "broccoli", "egg"]):
        return min(width_mm, height_mm) * 0.5
    # Default
    return 35.0


def _food_density(food_label: str) -> float:
    """Return approximate density (g/cm³) for volume-to-mass conversion."""
    lbl = food_label.lower()
    if any(x in lbl for x in ["soup", "broth", "juice"]): return 1.0
    if any(x in lbl for x in ["rice", "pasta", "noodle"]): return 0.75
    if any(x in lbl for x in ["salad", "greens", "vegetable"]): return 0.30
    if any(x in lbl for x in ["steak", "chicken", "meat", "pork", "beef", "lamb"]): return 1.05
    if any(x in lbl for x in ["cake", "bread", "muffin", "donut"]): return 0.45
    if any(x in lbl for x in ["cheese"]): return 1.10
    if any(x in lbl for x in ["fries", "chip", "nacho"]): return 0.55
    if any(x in lbl for x in ["ice cream"]): return 0.60
    return 0.80   # general food default

def classify_food(cv_crop, coco_cls_id=None):
    """
    COCO fruit/veg → visual colour classifier (reliable for raw produce).
    Everything else → Food-101 neural model first, visual as last resort.
    """
    if coco_cls_id in VISUAL_COCO_IDS:
        vis_label, vis_score = visual_classify(cv_crop)
        coco_name = VISUAL_COCO_IDS[coco_cls_id]
        if vis_score >= 0.65:
            return vis_label, vis_score
        return coco_name, 0.75

    if TF_OK and tf_model is not None and FOOD101_CLASSES:
        try:
            rgb    = cv2.cvtColor(cv2.resize(cv_crop, (224, 224)), cv2.COLOR_BGR2RGB)
            arr    = np.expand_dims(rgb.astype("float32"), 0)
            preds  = tf_model.predict(arr, verbose=0)
            probs  = tf.nn.softmax(preds, axis=1).numpy()[0]
            top5_i = np.argsort(probs)[::-1][:5]
            top5   = [(FOOD101_CLASSES[i].replace("_", " "), float(probs[i])) for i in top5_i]
            label, score = top5[0]
            print(f"  Food-101: {label} ({score:.3f}) top5:{[t[0] for t in top5]}")
            if score >= 0.05:
                return label, score
        except Exception as e:
            print(f"  TF error: {e}")

    vis_label, vis_score = visual_classify(cv_crop)
    CLEAR_VISUAL = {"apple", "banana", "orange", "salad", "green salad", "french fries", "fish and chips"}
    if vis_label in CLEAR_VISUAL and vis_score >= 0.72:
        return vis_label, vis_score

    return vis_label, vis_score


# Ingredients DB
INGREDIENTS={
    "fried rice":[{"name":"Cooked rice","amount":150,"unit":"g"},{"name":"Egg","amount":50,"unit":"g"},{"name":"Mixed vegetables","amount":60,"unit":"g"},{"name":"Soy sauce","amount":15,"unit":"ml"},{"name":"Sesame oil","amount":10,"unit":"ml"}],
    "pizza":[{"name":"Pizza dough","amount":120,"unit":"g"},{"name":"Tomato sauce","amount":40,"unit":"g"},{"name":"Mozzarella","amount":60,"unit":"g"},{"name":"Olive oil","amount":10,"unit":"ml"},{"name":"Basil","amount":5,"unit":"g"}],
    "hamburger":[{"name":"Beef patty","amount":120,"unit":"g"},{"name":"Burger bun","amount":60,"unit":"g"},{"name":"Cheddar","amount":20,"unit":"g"},{"name":"Lettuce","amount":15,"unit":"g"},{"name":"Tomato","amount":20,"unit":"g"}],
    "cheeseburger":[{"name":"Beef patty","amount":120,"unit":"g"},{"name":"Burger bun","amount":60,"unit":"g"},{"name":"Cheddar cheese","amount":25,"unit":"g"},{"name":"Lettuce","amount":15,"unit":"g"},{"name":"Pickles","amount":10,"unit":"g"}],
    "hot dog":[{"name":"Hot dog sausage","amount":80,"unit":"g"},{"name":"Hot dog bun","amount":50,"unit":"g"},{"name":"Mustard","amount":10,"unit":"g"},{"name":"Ketchup","amount":10,"unit":"g"}],
    "nachos":[{"name":"Tortilla chips","amount":80,"unit":"g"},{"name":"Cheddar cheese","amount":60,"unit":"g"},{"name":"Jalapeños","amount":15,"unit":"g"},{"name":"Sour cream","amount":30,"unit":"g"},{"name":"Salsa","amount":30,"unit":"g"}],
    "loaded nachos":[{"name":"Tortilla chips","amount":80,"unit":"g"},{"name":"Ground beef","amount":80,"unit":"g"},{"name":"Cheese sauce","amount":60,"unit":"g"},{"name":"Sour cream","amount":30,"unit":"g"},{"name":"Corn","amount":20,"unit":"g"},{"name":"Jalapeños","amount":15,"unit":"g"},{"name":"Salsa","amount":25,"unit":"g"}],
    "loaded fries":[{"name":"French fries","amount":150,"unit":"g"},{"name":"Ground beef","amount":80,"unit":"g"},{"name":"Cheese sauce","amount":50,"unit":"g"},{"name":"Sour cream","amount":30,"unit":"g"},{"name":"Spring onions","amount":10,"unit":"g"},{"name":"Paprika","amount":3,"unit":"g"}],
    "fish and chips":[{"name":"Battered fish fillet","amount":180,"unit":"g"},{"name":"Chips (thick cut)","amount":200,"unit":"g"},{"name":"Vegetable oil","amount":20,"unit":"ml"},{"name":"Lemon wedge","amount":15,"unit":"g"}],
    "french fries":[{"name":"Potatoes","amount":200,"unit":"g"},{"name":"Vegetable oil","amount":30,"unit":"ml"},{"name":"Salt","amount":2,"unit":"g"}],
    "burrito":[{"name":"Flour tortilla","amount":70,"unit":"g"},{"name":"Rice","amount":80,"unit":"g"},{"name":"Chicken","amount":80,"unit":"g"},{"name":"Black beans","amount":60,"unit":"g"},{"name":"Cheddar","amount":25,"unit":"g"},{"name":"Sour cream","amount":20,"unit":"g"}],
    "taco":[{"name":"Corn tortilla","amount":30,"unit":"g"},{"name":"Ground beef","amount":60,"unit":"g"},{"name":"Cheddar","amount":15,"unit":"g"},{"name":"Lettuce","amount":10,"unit":"g"},{"name":"Sour cream","amount":10,"unit":"g"}],
    "sushi":[{"name":"Sushi rice","amount":100,"unit":"g"},{"name":"Nori","amount":5,"unit":"g"},{"name":"Salmon","amount":40,"unit":"g"},{"name":"Soy sauce","amount":10,"unit":"ml"}],
    "ramen":[{"name":"Noodles","amount":100,"unit":"g"},{"name":"Pork broth","amount":300,"unit":"ml"},{"name":"Chashu pork","amount":60,"unit":"g"},{"name":"Egg","amount":50,"unit":"g"},{"name":"Spring onions","amount":10,"unit":"g"}],
    "pasta":[{"name":"Pasta","amount":100,"unit":"g"},{"name":"Tomato sauce","amount":80,"unit":"g"},{"name":"Parmesan","amount":15,"unit":"g"},{"name":"Olive oil","amount":10,"unit":"ml"}],
    "pasta carbonara":[{"name":"Spaghetti","amount":100,"unit":"g"},{"name":"Pancetta","amount":60,"unit":"g"},{"name":"Egg yolks","amount":40,"unit":"g"},{"name":"Parmesan","amount":30,"unit":"g"},{"name":"Black pepper","amount":2,"unit":"g"}],
    "steak":[{"name":"Beef steak","amount":200,"unit":"g"},{"name":"Butter","amount":15,"unit":"g"},{"name":"Garlic","amount":5,"unit":"g"},{"name":"Rosemary","amount":3,"unit":"g"}],
    "chicken":[{"name":"Chicken breast","amount":150,"unit":"g"},{"name":"Olive oil","amount":10,"unit":"ml"},{"name":"Garlic powder","amount":2,"unit":"g"},{"name":"Salt & pepper","amount":2,"unit":"g"}],
    "curry":[{"name":"Meat/chicken","amount":150,"unit":"g"},{"name":"Curry sauce","amount":150,"unit":"g"},{"name":"Onion","amount":60,"unit":"g"},{"name":"Garlic","amount":10,"unit":"g"},{"name":"Spice blend","amount":8,"unit":"g"}],
    "rice dish":[{"name":"White rice","amount":150,"unit":"g"},{"name":"Water","amount":200,"unit":"ml"},{"name":"Salt","amount":2,"unit":"g"}],
    "salad":[{"name":"Mixed greens","amount":60,"unit":"g"},{"name":"Tomato","amount":40,"unit":"g"},{"name":"Cucumber","amount":40,"unit":"g"},{"name":"Olive oil","amount":15,"unit":"ml"}],
    "green salad":[{"name":"Mixed greens","amount":80,"unit":"g"},{"name":"Cucumber","amount":40,"unit":"g"},{"name":"Avocado","amount":40,"unit":"g"},{"name":"Olive oil","amount":15,"unit":"ml"}],
    "waffle":[{"name":"Flour","amount":100,"unit":"g"},{"name":"Milk","amount":120,"unit":"ml"},{"name":"Egg","amount":50,"unit":"g"},{"name":"Butter","amount":30,"unit":"g"},{"name":"Sugar","amount":20,"unit":"g"}],
    "pancake":[{"name":"Flour","amount":100,"unit":"g"},{"name":"Milk","amount":150,"unit":"ml"},{"name":"Egg","amount":50,"unit":"g"},{"name":"Butter","amount":20,"unit":"g"},{"name":"Maple syrup","amount":30,"unit":"ml"}],
    "donut":[{"name":"Flour","amount":80,"unit":"g"},{"name":"Sugar","amount":40,"unit":"g"},{"name":"Butter","amount":20,"unit":"g"},{"name":"Egg","amount":25,"unit":"g"},{"name":"Glaze","amount":30,"unit":"g"}],
    "cake":[{"name":"Flour","amount":100,"unit":"g"},{"name":"Sugar","amount":80,"unit":"g"},{"name":"Butter","amount":60,"unit":"g"},{"name":"Eggs","amount":50,"unit":"g"},{"name":"Milk","amount":60,"unit":"ml"}],
    "ice cream":[{"name":"Cream","amount":100,"unit":"ml"},{"name":"Milk","amount":80,"unit":"ml"},{"name":"Sugar","amount":50,"unit":"g"},{"name":"Vanilla","amount":3,"unit":"ml"}],
    "banana":[{"name":"Banana","amount":118,"unit":"g"}],
    "apple":[{"name":"Apple","amount":182,"unit":"g"}],
    "orange":[{"name":"Orange","amount":131,"unit":"g"}],
    "broccoli":[{"name":"Broccoli","amount":150,"unit":"g"},{"name":"Olive oil","amount":5,"unit":"ml"}],
    "eggs":[{"name":"Eggs","amount":100,"unit":"g"},{"name":"Butter","amount":5,"unit":"g"}],
    "omelette":[{"name":"Eggs (3)","amount":150,"unit":"g"},{"name":"Butter","amount":10,"unit":"g"},{"name":"Cheese","amount":20,"unit":"g"}],
    "bread":[{"name":"Flour","amount":120,"unit":"g"},{"name":"Water","amount":80,"unit":"ml"},{"name":"Yeast","amount":3,"unit":"g"},{"name":"Salt","amount":2,"unit":"g"}],
    "bagel":[{"name":"Bread flour","amount":120,"unit":"g"},{"name":"Water","amount":70,"unit":"ml"},{"name":"Yeast","amount":3,"unit":"g"},{"name":"Salt","amount":3,"unit":"g"}],
    "croissant":[{"name":"Flour","amount":100,"unit":"g"},{"name":"Butter","amount":60,"unit":"g"},{"name":"Milk","amount":50,"unit":"ml"},{"name":"Yeast","amount":3,"unit":"g"}],
    "mixed dish":[{"name":"Mixed ingredients","amount":200,"unit":"g"}],
}

def get_ingredients(food_name):
    key=food_name.lower().strip()
    if key in INGREDIENTS: return INGREDIENTS[key]
    for k,v in INGREDIENTS.items():
        if k in key or key in k: return v
    n=key
    if any(x in n for x in ["chicken","turkey"]): return [{"name":"Chicken","amount":150,"unit":"g"},{"name":"Oil","amount":10,"unit":"ml"},{"name":"Seasoning","amount":3,"unit":"g"}]
    if any(x in n for x in ["beef","steak","pork","lamb","meat"]): return [{"name":"Meat","amount":180,"unit":"g"},{"name":"Oil","amount":10,"unit":"ml"},{"name":"Salt & pepper","amount":2,"unit":"g"}]
    if any(x in n for x in ["fish","salmon","tuna","cod","shrimp"]): return [{"name":food_name.title(),"amount":150,"unit":"g"},{"name":"Lemon juice","amount":10,"unit":"ml"},{"name":"Olive oil","amount":8,"unit":"ml"}]
    if any(x in n for x in ["pasta","noodle","spaghetti"]): return [{"name":"Pasta","amount":100,"unit":"g"},{"name":"Sauce","amount":80,"unit":"g"},{"name":"Parmesan","amount":15,"unit":"g"}]
    if any(x in n for x in ["rice","risotto"]): return [{"name":"Rice","amount":150,"unit":"g"},{"name":"Water","amount":250,"unit":"ml"},{"name":"Salt","amount":2,"unit":"g"}]
    if any(x in n for x in ["salad","greens"]): return [{"name":"Mixed greens","amount":80,"unit":"g"},{"name":"Tomato","amount":40,"unit":"g"},{"name":"Dressing","amount":20,"unit":"ml"}]
    if any(x in n for x in ["cake","pie","pastry","muffin"]): return [{"name":"Flour","amount":100,"unit":"g"},{"name":"Sugar","amount":80,"unit":"g"},{"name":"Butter","amount":60,"unit":"g"}]
    if any(x in n for x in ["soup","broth","stew"]): return [{"name":"Broth","amount":300,"unit":"ml"},{"name":"Vegetables","amount":100,"unit":"g"}]
    return [{"name":food_name.title(),"amount":150,"unit":"g"}]

MACRO_DB={
    "fried rice":{"cal":163,"protein":5,"carbs":28,"fat":4},
    "pizza":{"cal":266,"protein":11,"carbs":33,"fat":10},
    "hamburger":{"cal":295,"protein":17,"carbs":24,"fat":14},
    "cheeseburger":{"cal":310,"protein":18,"carbs":25,"fat":16},
    "hot dog":{"cal":290,"protein":11,"carbs":22,"fat":18},
    "nachos":{"cal":346,"protein":10,"carbs":36,"fat":19},
    "loaded nachos":{"cal":380,"protein":16,"carbs":32,"fat":22},
    "loaded fries":{"cal":370,"protein":14,"carbs":38,"fat":18},
    "french fries":{"cal":312,"protein":4,"carbs":41,"fat":15},
    "fish and chips":{"cal":290,"protein":15,"carbs":32,"fat":12},
    "sandwich":{"cal":250,"protein":12,"carbs":30,"fat":8},
    "burrito":{"cal":206,"protein":10,"carbs":23,"fat":8},
    "taco":{"cal":226,"protein":12,"carbs":20,"fat":11},
    "sushi":{"cal":150,"protein":10,"carbs":20,"fat":3},
    "ramen":{"cal":180,"protein":11,"carbs":26,"fat":4},
    "pasta":{"cal":158,"protein":6,"carbs":30,"fat":2},
    "pasta carbonara":{"cal":371,"protein":14,"carbs":42,"fat":16},
    "steak":{"cal":271,"protein":26,"carbs":0,"fat":18},
    "chicken":{"cal":165,"protein":31,"carbs":0,"fat":4},
    "curry":{"cal":150,"protein":10,"carbs":12,"fat":7},
    "rice dish":{"cal":130,"protein":3,"carbs":28,"fat":0},
    "salad":{"cal":50,"protein":2,"carbs":8,"fat":2},
    "green salad":{"cal":45,"protein":2,"carbs":6,"fat":2},
    "waffle":{"cal":291,"protein":8,"carbs":37,"fat":13},
    "pancake":{"cal":227,"protein":7,"carbs":34,"fat":7},
    "donut":{"cal":415,"protein":5,"carbs":51,"fat":20},
    "cake":{"cal":350,"protein":4,"carbs":55,"fat":13},
    "ice cream":{"cal":207,"protein":3,"carbs":24,"fat":11},
    "bread":{"cal":265,"protein":9,"carbs":49,"fat":3},
    "bagel":{"cal":270,"protein":11,"carbs":53,"fat":2},
    "croissant":{"cal":406,"protein":9,"carbs":46,"fat":21},
    "banana":{"cal":89,"protein":1,"carbs":23,"fat":0},
    "apple":{"cal":52,"protein":0,"carbs":14,"fat":0},
    "orange":{"cal":47,"protein":1,"carbs":12,"fat":0},
    "broccoli":{"cal":34,"protein":3,"carbs":7,"fat":0},
    "eggs":{"cal":155,"protein":13,"carbs":1,"fat":11},
    "omelette":{"cal":154,"protein":11,"carbs":0,"fat":12},
    "mixed dish":{"cal":200,"protein":8,"carbs":25,"fat":8},
    "loaded fries":{"cal":370,"protein":14,"carbs":38,"fat":18},
    "chicken wings":{"cal":290,"protein":27,"carbs":0,"fat":20},
    "chicken wing":{"cal":290,"protein":27,"carbs":0,"fat":20},
    "pad thai":{"cal":181,"protein":10,"carbs":26,"fat":5},
    "fish and chips":{"cal":290,"protein":15,"carbs":32,"fat":12},
    "spring rolls":{"cal":220,"protein":6,"carbs":26,"fat":10},
    "dumplings":{"cal":180,"protein":8,"carbs":22,"fat":7},
    "curry":{"cal":150,"protein":10,"carbs":12,"fat":7},
    "bibimbap":{"cal":490,"protein":25,"carbs":78,"fat":8},
    "pho":{"cal":350,"protein":25,"carbs":45,"fat":5},
    "gyoza":{"cal":200,"protein":9,"carbs":20,"fat":9},
    "takoyaki":{"cal":200,"protein":8,"carbs":24,"fat":8},
    "miso soup":{"cal":40,"protein":3,"carbs":4,"fat":2},
    "tiramisu":{"cal":280,"protein":5,"carbs":30,"fat":15},
    "cheesecake":{"cal":321,"protein":5,"carbs":26,"fat":22},
    "macarons":{"cal":420,"protein":5,"carbs":64,"fat":16},
    "hummus":{"cal":166,"protein":8,"carbs":14,"fat":10},
    "pork chop":{"cal":231,"protein":27,"carbs":0,"fat":13},
    "ceviche":{"cal":100,"protein":14,"carbs":5,"fat":2},
    "scallops":{"cal":111,"protein":20,"carbs":5,"fat":1},
    "baby back ribs":{"cal":290,"protein":25,"carbs":5,"fat":19},
    "clam chowder":{"cal":186,"protein":7,"carbs":16,"fat":10},
    "edamame":{"cal":122,"protein":11,"carbs":10,"fat":5},
    "default":{"cal":200,"protein":8,"carbs":25,"fat":8},
}

app=Flask(__name__,template_folder="templates")
app.secret_key=secrets.token_hex(32)
USERS_FILE="users.json"

def load_users():
    if not os.path.exists(USERS_FILE): return {}
    with open(USERS_FILE) as f: return json.load(f)
def save_users(u):
    with open(USERS_FILE,"w") as f: json.dump(u,f,indent=2)
def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

@app.route("/")
def index(): return redirect(url_for("dashboard") if session.get("user") else url_for("login_page"))

@app.route("/login")
def login_page():
    if session.get("user"): return redirect(url_for("dashboard"))
    return send_from_directory("templates","login.html")

@app.route("/dashboard")
def dashboard():
    if not session.get("user"): return redirect(url_for("login_page"))
    return send_from_directory("templates","dashboard.html")

@app.route("/api/login",methods=["POST"])
def api_login():
    d=request.get_json(silent=True) or {}
    email,pw=d.get("email","").strip().lower(),d.get("password","")
    if not email or not pw: return jsonify({"ok":False,"error":"Enter email and password"}),400
    users=load_users(); user=users.get(email)
    if not user or user["password"]!=hash_pw(pw): return jsonify({"ok":False,"error":"Invalid email or password"}),401
    session["user"]={"email":email,"name":user["name"]}
    return jsonify({"ok":True,"name":user["name"]})

@app.route("/api/register",methods=["POST"])
def api_register():
    d=request.get_json(silent=True) or {}
    name,email,pw=d.get("name","").strip(),d.get("email","").strip().lower(),d.get("password","")
    if not name or not email or not pw: return jsonify({"ok":False,"error":"All fields required"}),400
    if len(pw)<6: return jsonify({"ok":False,"error":"Password must be 6+ characters"}),400
    users=load_users()
    if email in users: return jsonify({"ok":False,"error":"Account already exists"}),409
    users[email]={"name":name,"password":hash_pw(pw)}; save_users(users)
    session["user"]={"email":email,"name":name}
    return jsonify({"ok":True,"name":name})

@app.route("/api/logout",methods=["POST"])
def api_logout(): session.clear(); return jsonify({"ok":True})

@app.route("/api/me")
def api_me(): u=session.get("user"); return jsonify({"ok":bool(u),"user":u})

@app.route("/api/analyse",methods=["POST"])
def analyse():
    if not session.get("user"): return jsonify({"ok":False,"error":"Not authenticated"}),401
    try:
        d=request.get_json(silent=True) or {}
        img_b64=d.get("image","")
        if not img_b64: return jsonify({"ok":False,"error":"No image provided"}),400
        if "," in img_b64: img_b64=img_b64.split(",",1)[1]
        raw=base64.b64decode(img_b64)
        nparr=np.frombuffer(raw,np.uint8)
        cv_img=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        if cv_img is None: return jsonify({"ok":False,"error":"Cannot decode image"}),400
        h,w=cv_img.shape[:2]

        lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
        l=clahe.apply(l)
        enhanced=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)

        detections=[]
        if YOLO_OK and yolo_model:
            results=yolo_model(enhanced, conf=0.30, iou=0.35, verbose=False)
            raw_boxes=[]
            for box in results[0].boxes:
                cls_id=int(box.cls[0])
                if cls_id not in FOOD_CLASS_IDS: continue
                confidence=float(box.conf[0])
                x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                raw_boxes.append((x1,y1,x2,y2,confidence,cls_id,results[0].names[cls_id]))

            # NMS — suppress overlapping boxes (IoU > 0.35)
            def iou(a,b):
                ix1,iy1=max(a[0],b[0]),max(a[1],b[1])
                ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
                inter=max(0,ix2-ix1)*max(0,iy2-iy1)
                ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
                return inter/ua if ua>0 else 0

            raw_boxes.sort(key=lambda x:-x[4])
            kept=[]
            for box in raw_boxes:
                if all(iou(box,k)<0.35 for k in kept):
                    kept.append(box)

            # Classify each kept box
            seen_labels={}  # deduplicate: keep best confidence per food label
            for x1,y1,x2,y2,confidence,cls_id,cls_name in kept:
                crop=enhanced[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
                if crop.size==0: continue
                food_label,food_score=classify_food(crop, coco_cls_id=cls_id)
                # If same food label detected multiple times, keep highest confidence only
                if food_label in seen_labels:
                    if confidence <= seen_labels[food_label]:
                        continue
                seen_labels[food_label]=confidence
                est_g,portion_method,portion_conf=estimate_portion(enhanced,(x1,y1,x2,y2),food_label,results)
                nutrition=_usda_scaled(food_label,est_g)
                ingredients=get_ingredients(food_label)
                detections.append({
                    "id":len(detections),"yolo_class":cls_name,
                    "tf_class":food_label,"tf_score":round(food_score,3),
                    "confidence":round(confidence,3),
                    "food_name":food_label.title(),
                    "bbox":[x1,y1,x2,y2],
                    "nutrition":nutrition,"ingredients":ingredients,
                    "portion_g":est_g,"portion_method":portion_method,
                    "portion_confidence":round(portion_conf,2),
                })

        #Whole-image fallback (YOLO found nothing)
        if not detections:
            food_label,food_score=classify_food(enhanced)
            est_g,portion_method,portion_conf=estimate_portion(enhanced,(0,0,w,h),food_label,yolo_results=None)
            nutrition=_usda_scaled(food_label,est_g)
            ingredients=get_ingredients(food_label)
            detections.append({
                "id":0,"yolo_class":"whole image",
                "tf_class":food_label,"tf_score":round(food_score,3),
                "confidence":round(food_score,3),
                "food_name":food_label.title(),
                "bbox":[0,0,w,h],
                "nutrition":nutrition,"ingredients":ingredients,
                "portion_g":est_g,"portion_method":portion_method,
                "portion_confidence":round(portion_conf,2),
            })

        totals={"cal":round(sum(d["nutrition"]["cal"] for d in detections),1),"protein":round(sum(d["nutrition"]["protein"] for d in detections),1),"carbs":round(sum(d["nutrition"]["carbs"] for d in detections),1),"fat":round(sum(d["nutrition"]["fat"] for d in detections),1)}
        return jsonify({"ok":True,"detections":detections,"totals":totals,"annotated":_draw_boxes(cv_img.copy(),detections),"mode":"yolo" if len(detections)>1 or detections[0]["yolo_class"]!="whole image" else "whole_image"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/api/search")
def search():
    if not session.get("user"): return jsonify({"results":[]}),401
    q=request.args.get("q","").strip()
    if not q: return jsonify({"results":[]})
    try:
        params=urllib.parse.urlencode({"query":q,"api_key":"DEMO_KEY","pageSize":8,"dataType":"SR Legacy,Foundation,Branded"})
        req=urllib.request.Request(f"https://api.nal.usda.gov/fdc/v1/foods/search?{params}",headers={"User-Agent":"MacroLens/4.0"})
        with urllib.request.urlopen(req,timeout=7) as resp: data=json.loads(resp.read())
        results=[{"name":f.get("description",""),"cal":_nn(f,1008),"protein":_nn(f,1003),"carbs":_nn(f,1005),"fat":_nn(f,1004),"source":"USDA","ingredients":get_ingredients(f.get("description",""))} for f in data.get("foods",[])[:8]]
        return jsonify({"results":results})
    except Exception as e: return jsonify({"results":[],"error":str(e)})

#Spoonacular

SPOONACULAR_KEY = "236739ab27264ffebe1b5e125d61b158"

@app.route("/api/recipes")
def recipes():
    if not session.get("user"):
        return jsonify({"ok": False, "error": "Not authenticated"}), 401

    food = request.args.get("food", "").strip()
    if not food:
        return jsonify({"ok": False, "error": "No food specified"}), 400

    if SPOONACULAR_KEY == "YOUR_SPOONACULAR_API_KEY":
        return jsonify({"ok": False, "error": "Spoonacular API key not set"}), 503

    try:
        # Search recipes by ingredient / food name
        params = urllib.parse.urlencode({
            "apiKey":  SPOONACULAR_KEY,
            "query":   food,
            "number":  6,
            "addRecipeInformation": True,
            "fillIngredients": False,
            "instructionsRequired": True,
        })
        req = urllib.request.Request(
            f"https://api.spoonacular.com/recipes/complexSearch?{params}",
            headers={"User-Agent": "MacroLens/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        recipes_out = []
        for r in data.get("results", []):
            recipes_out.append({
                "id":           r.get("id"),
                "title":        r.get("title", ""),
                "image":        r.get("image", ""),
                "ready_in":     r.get("readyInMinutes"),
                "servings":     r.get("servings"),
                "source_url":   r.get("sourceUrl", ""),
                "summary":      _strip_html(r.get("summary", ""))[:200],
                "diets":        r.get("diets", []),
                "calories":     _spoon_cal(r),
            })

        return jsonify({"ok": True, "food": food, "recipes": recipes_out})

    except urllib.error.HTTPError as e:
        if e.code == 402:
            return jsonify({"ok": False, "error": "Spoonacular daily quota exceeded"}), 402
        return jsonify({"ok": False, "error": f"Spoonacular error {e.code}"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

def _strip_html(text):
    """Remove HTML tags from Spoonacular summaries."""
    import re
    return re.sub(r"<[^>]+>", "", text or "")

def _spoon_cal(recipe):
    """Extract calories from Spoonacular nutrition if present."""
    nutrients = recipe.get("nutrition", {}).get("nutrients", [])
    cal = next((n["amount"] for n in nutrients if n.get("name") == "Calories"), None)
    return round(cal) if cal else None

def _draw_boxes(img,detections):
    COLORS=[(217,95,59),(46,127,163),(196,154,39),(107,79,160),(58,140,92),(217,59,150)]
    for d in detections:
        x1,y1,x2,y2=d["bbox"]; col=COLORS[d["id"]%len(COLORS)]
        cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
        label=f"{d['food_name']} {int(d['confidence']*100)}%"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
        cv2.rectangle(img,(x1,y1-th-8),(x1+tw+6,y1),col,-1)
        cv2.putText(img,label,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    _,buf=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,88])
    return "data:image/jpeg;base64,"+base64.b64encode(buf).decode()

def _nn(food,nid):
    n=next((x for x in food.get("foodNutrients",[]) if str(x.get("nutrientId",""))==str(nid) or str(x.get("nutrientNumber",""))==str(nid)),None)
    return round(float(n["value"]),1) if n else 0.0

def _usda(label):
    """Look up nutrition per 100g from USDA, fall back to built-in DB."""
    query=label.replace("_"," ")
    try:
        params=urllib.parse.urlencode({"query":query,"api_key":"DEMO_KEY","pageSize":5,"dataType":"SR Legacy,Foundation"})
        req=urllib.request.Request(f"https://api.nal.usda.gov/fdc/v1/foods/search?{params}",headers={"User-Agent":"MacroLens/5.0"})
        with urllib.request.urlopen(req,timeout=6) as resp: data=json.loads(resp.read())
        foods=[f for f in data.get("foods",[]) if f.get("foodNutrients")]
        if foods:
            food=foods[0]
            return {"name":food.get("description",query),"cal":_nn(food,1008),"protein":_nn(food,1003),"carbs":_nn(food,1005),"fat":_nn(food,1004),"source":"USDA"}
    except Exception as e:
        print(f"  USDA: {e}")
    entry=MACRO_DB.get(label.lower(),MACRO_DB["default"])
    return {"name":query.title(),"source":"built-in","cal":float(entry["cal"]),"protein":float(entry["protein"]),"carbs":float(entry["carbs"]),"fat":float(entry["fat"])}

def _usda_scaled(label: str, grams: float) -> dict:
    base = _usda(label)
    if base.get("cal", 0) == 0:
        entry = MACRO_DB.get(label.lower(), MACRO_DB["default"])
        base = {"name": label.title(), "source": "built-in",
                "cal": float(entry["cal"]), "protein": float(entry["protein"]),
                "carbs": float(entry["carbs"]), "fat": float(entry["fat"])}
    factor = max(0.0, grams) / 100.0
    return {
        "name":     base["name"],
        "source":   base["source"],
        "per_100g": {"cal": base["cal"], "protein": base["protein"],
                     "carbs": base["carbs"], "fat": base["fat"]},
        "cal":      round(base["cal"]     * factor, 1),
        "protein":  round(base["protein"] * factor, 1),
        "carbs":    round(max(0.0, base["carbs"] * factor), 1),
        "fat":      round(max(0.0, base["fat"]   * factor), 1),
        "portion_g": grams,
    }

REPORTS_FILE = "reports.json"

@app.route("/api/report", methods=["POST"])
def report_prediction():
    if not session.get("user"):
        return jsonify({"ok": False, "error": "Not authenticated"}), 401
    try:
        d = request.get_json(silent=True) or {}
        wrong_label = d.get("wrong_label", "").strip()
        correct_label = d.get("correct_label", "").strip()
        detection_id = d.get("detection_id")
        if not wrong_label:
            return jsonify({"ok": False, "error": "Missing prediction label"}), 400
        reports = []
        if os.path.exists(REPORTS_FILE):
            with open(REPORTS_FILE) as f:
                reports = json.load(f)
        import datetime
        reports.append({
            "user": session["user"]["email"],
            "wrong_label": wrong_label,
            "correct_label": correct_label,
            "detection_id": detection_id,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        with open(REPORTS_FILE, "w") as f:
            json.dump(reports, f, indent=2)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__=="__main__":
    print("\nMacroLens → http://localhost:5000\n")
    app.run(debug=True,host="0.0.0.0",port=5000)
