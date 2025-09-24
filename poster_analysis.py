import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deepface import DeepFace
import easyocr
import openai
from collections import defaultdict

# --- Use standalone Keras for TF 2.20+ ---
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8 model."""
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_ocr_model():
    """Loads the EasyOCR reader model."""
    return easyocr.Reader(['en'])

# --- Core Computer Vision Functions ---
def analyze_persons(image_np, person_boxes):
    """Analyzes each detected person for gender and emotion with robust handling."""
    person_details = []
    for i, raw_box in enumerate(person_boxes):
        try:
            x1, y1, x2, y2 = map(int, raw_box)
        except Exception:
            x1, y1, x2, y2 = 0, 0, image_np.shape[1], image_np.shape[0]

        h = y2 - y1
        w = x2 - x1
        # Expand crop to include more of the head/face
        face_y1 = max(0, y1 - int(h * 0.4))
        face_y2 = min(image_np.shape[0], y2 + int(h * 0.4))
        face_x1 = max(0, x1 - int(w * 0.4))
        face_x2 = min(image_np.shape[1], x2 + int(w * 0.4))
        cropped = image_np[face_y1:face_y2, face_x1:face_x2]

        # Convert to RGB if needed
        if cropped.ndim == 2 or (cropped.ndim == 3 and cropped.shape[2] == 1):
            cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
        elif cropped.ndim == 3 and cropped.shape[2] == 4:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_RGBA2RGB)

        detail = {
            "id": i,
            "box": [x1, y1, x2, y2],
            "gender": "Unknown",
            "emotion": "Unknown",
            "area": max(1, (x2 - x1) * (y2 - y1)),
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "analysis_error": None,
            "crop_dims": f"{cropped.shape}"
        }

        if cropped.size == 0 or cropped.shape[0] < 48 or cropped.shape[1] < 48:
            detail["analysis_error"] = f"Face region too small: {cropped.shape}"
            person_details.append(detail)
            continue

        if cropped.dtype != np.uint8:
            cropped = (np.clip(cropped, 0, 255)).astype(np.uint8)

        try:
            analysis = DeepFace.analyze(
                img_path=cropped,
                actions=['gender', 'emotion'],
                enforce_detection=False,
                detector_backend='retinaface',  # Try 'retinaface', 'mtcnn', or 'opencv'
                prog_bar=False
            )

            # Debug: log DeepFace output
            print(f"DeepFace output for person {i}: {analysis}")

            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]

            if isinstance(analysis, dict):
                gender_val = analysis.get('dominant_gender') or analysis.get('gender') or analysis.get('sex')
                emotion_val = analysis.get('dominant_emotion') or analysis.get('emotion')

                if isinstance(gender_val, dict):
                    gender_val = max(gender_val.items(), key=lambda kv: kv[1])[0]

                if gender_val:
                    g = str(gender_val).strip().lower()
                    if 'male' in g or 'man' in g:
                        detail['gender'] = 'Man'
                    elif 'female' in g or 'woman' in g:
                        detail['gender'] = 'Woman'
                    else:
                        detail['gender'] = gender_val

                if emotion_val:
                    if isinstance(emotion_val, dict):
                        emotion_val = max(emotion_val.items(), key=lambda kv: kv[1])[0]
                    detail['emotion'] = str(emotion_val)

                detail['analysis_error'] = "DeepFace:ok"
            else:
                detail['analysis_error'] = "DeepFace:unexpected-format"

        except Exception as e:
            detail['analysis_error'] = f"DeepFace exception: {repr(e)}"

        if detail['gender'] == 'Unknown':
            detail['analysis_error'] += " | fallback-needed"

        person_details.append(detail)
    return person_details

def detect_persons_and_text(image_np, yolo_model, ocr_reader):
    """Detects persons using YOLO and text using EasyOCR."""
    # YOLO detection
    results = yolo_model(image_np)
    person_boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # Class 0 is 'person' in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append([x1, y1, x2, y2])
    # EasyOCR detection
    ocr_results = ocr_reader.readtext(image_np)
    detected_text = " ".join([res[1] for res in ocr_results])
    return person_boxes, detected_text

# --- Metrics Calculation ---
def calculate_metrics(person_details, image_width):
    metrics = defaultdict(float)
    male_areas, female_areas = [], []
    center_region = (image_width * 0.3, image_width * 0.7)

    for p in person_details:
        if p['gender'] == 'Man':
            metrics['male_count'] += 1
            male_areas.append(p['area'])
            if center_region[0] < p['center'][0] < center_region[1]:
                metrics['males_in_center'] += 1
        elif p['gender'] == 'Woman':
            metrics['female_count'] += 1
            female_areas.append(p['area'])
            if center_region[0] < p['center'][0] < center_region[1]:
                metrics['females_in_center'] += 1

    metrics['avg_male_area'] = np.mean(male_areas) if male_areas else 0
    metrics['avg_female_area'] = np.mean(female_areas) if female_areas else 0
    return metrics

# --- Rule-Based Suggestions ---
def generate_rule_based_suggestions(metrics):
    suggestions = []
    if metrics['male_count'] > metrics['female_count'] * 1.5:
        suggestions.append("Poster heavily male-dominated. Consider featuring more female characters prominently.")
    if metrics['avg_male_area'] > metrics['avg_female_area'] * 1.5:
        suggestions.append("Male figures are shown larger than female figures. Balance sizes.")
    if metrics['males_in_center'] > metrics['females_in_center']:
        suggestions.append("Males dominate the center of the poster. Place female characters more centrally.")
    if not suggestions:
        suggestions.append("Poster shows balanced gender representation.")
    return suggestions

# --- AI Suggestions ---
def generate_ai_suggestions(metrics, openai_api_key):
    if not openai_api_key:
        return "No API key provided. Cannot generate AI suggestions."

    openai.api_key = openai_api_key
    prompt = f"""
    Analyze the following poster metrics for gender stereotypes and suggest improvements:
    {metrics}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an AI assistant helping remove gender stereotypes."},
                      {"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"AI Suggestion Error: {repr(e)}"

# --- Drawing Function ---
def draw_annotations(image_np, person_details):
    annotated = image_np.copy()
    for p in person_details:
        x1, y1, x2, y2 = p['box']
        label = f"{p['gender']} | {p['emotion']}"
        color = (0, 255, 0) if p['gender'] == 'Woman' else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated

# --- Streamlit App ---
st.title("🎬 Bollywood Poster Gender Stereotype Analyzer")

uploaded_file = st.file_uploader("Upload a movie poster", type=["jpg", "jpeg", "png"])
openai_api_key = st.text_input("Enter OpenAI API Key (optional)", type="password")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    yolo_model = load_yolo_model()
    ocr_reader = load_ocr_model()

    person_boxes, detected_text = detect_persons_and_text(image_np, yolo_model, ocr_reader)
    person_details = analyze_persons(image_np, person_boxes)
    metrics = calculate_metrics(person_details, image_np.shape[1])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Uploaded Poster")
        st.image(image, use_column_width=True)

    rule_suggestions = generate_rule_based_suggestions(metrics)
    ai_suggestions = generate_ai_suggestions(metrics, openai_api_key)
    annotated_image = draw_annotations(image_np, person_details)

    with col2:
        st.subheader("📊 Analysis Results")
        st.image(annotated_image, use_column_width=True)
        st.markdown("---")
        st.write("#### Metrics")
        st.metric("Men Detected", metrics["male_count"])
        st.metric("Women Detected", metrics["female_count"])
        st.markdown(f"Avg Male Size: `{metrics['avg_male_area']:.0f}` px")
        st.markdown(f"Avg Female Size: `{metrics['avg_female_area']:.0f}` px")
        st.markdown(f"Men in Center: `{metrics['males_in_center']}`")
        st.markdown(f"Women in Center: `{metrics['females_in_center']}`")
        
        with st.expander("🔍 Debug Info"):
            for person in person_details:
                if person.get('analysis_error'):
                    st.error(f"Person {person['id']}: {person['analysis_error']} | Crop: {person.get('crop_dims', 'N/A')}")
                else:
                    st.success(f"Person {person['id']}: Analysis successful | Crop: {person.get('crop_dims', 'N/A')}")
            st.info(f"Total persons detected by YOLO: {len(person_boxes)}")
            successful_analyses = sum(1 for p in person_details if p['gender'] != 'Unknown')
            st.info(f"Successful DeepFace analyses: {successful_analyses}/{len(person_details)}")
        
        with st.expander("📝 Detected Text"):
            st.write(detected_text)
        st.markdown("---")
        st.write("#### Rule-Based Suggestions")
        for sug in rule_suggestions:
            st.warning(f"💡 {sug}")
        st.markdown("---")
        st.write("#### 🤖 AI Suggestions")
        st.info(ai_suggestions)
else:
    st.info("Upload a poster image to begin analysis.")