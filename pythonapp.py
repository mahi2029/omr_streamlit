from flask import Flask, render_template, request
import os
from scoring import score_sheet

app = Flask(_name_)

# Example Answer Key (Change this for your test/exam)
answer_key = {
    1: 2,  # Q1 = C
    2: 0,  # Q2 = A
    3: 1,  # Q3 = B
    4: 3   # Q4 = D
}

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "omr_sheet" not in request.files:
            return "⚠️ No file uploaded"

        file = request.files["omr_sheet"]
        if file.filename == "":
            return "⚠️ No selected file"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Score the uploaded sheet
        score, total, results = score_sheet(filepath, answer_key, debug=False)

        return render_template("result.html", score=score, total=total, results=results)

    return render_template("index.html")

if _name_ == "_main_":
    app.run(debug=True)