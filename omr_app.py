import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------------
# 1. Load CNN Model
# ----------------------------
cnn_model = load_model("model/bubble_cnn.h5")

# ----------------------------
# 2. Define Answer Key
# ----------------------------
ANSWER_KEY = {1:2, 2:0, 3:3, 4:1, 5:2}  # Example

# ----------------------------
# 3. Function: Score Calculation
# ----------------------------
def calculate_score(student_answers):
    correct = 0
    for q, ans in student_answers.items():
        if q in ANSWER_KEY and ans == ANSWER_KEY[q]:
            correct += 1
    total = len(ANSWER_KEY)
    score = (correct / total) * 100
    return score, correct, total

# ----------------------------
# 4. Function: Extract Answers & Visual Feedback
# ----------------------------
def extract_answers(sheet_img, answer_key=None):
    display_img = sheet_img.copy()
    gray = cv2.cvtColor(sheet_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_cnts = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])  # top-to-bottom

    num_choices = 4
    answers = {}

    for q, i in enumerate(range(0, len(bubble_cnts), num_choices)):
        cnts = bubble_cnts[i:i+num_choices]
        filled = None
        max_conf = 0

        for j, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            bubble_roi = sheet_img[y:y+h, x:x+w]

            roi_resized = cv2.resize(bubble_roi, (28,28))
            roi_norm = roi_resized.astype("float32") / 255.0
            roi_norm = roi_norm.reshape(1,28,28,3)

            prob = cnn_model.predict(roi_norm, verbose=0)[0][0]

            if prob > max_conf:
                max_conf = prob
                filled = j

        answers[q+1] = filled

        # Visual feedback
        color = (255,0,0)  # Blue = selected
        if answer_key and q+1 in answer_key:
            if filled == answer_key[q+1]:
                color = (0,255,0)  # Green = correct
            else:
                color = (0,0,255)  # Red = wrong
        cv2.drawContours(display_img, [cnts[filled]], -1, color, 2)

    return answers, display_img

# ----------------------------
# 5. Streamlit UI
# ----------------------------
st.title("Automated OMR Scoring System 🚀")

uploaded_file = st.file_uploader("Upload OMR Sheet", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    sheet_img = cv2.imdecode(file_bytes, 1)

    # Extract answers & visualize
    answers, vis_img = extract_answers(sheet_img, ANSWER_KEY)

    # Calculate score
    score, correct, total = calculate_score(answers)

    # Display results
    st.subheader("Score")
    st.write(f"{score}% ({correct} / {total})")

    st.subheader("Answers Selected")
    st.write(answers)

    st.subheader("OMR Sheet Visualization")
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    st.image(vis_img_rgb, use_column_width=True)