from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot  as plt
import io
import base64

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a strong secret key
login_manager = LoginManager(app)
login_manager.login_view = "login"

# Default credentials
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"

# Flask-Login User model
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    if user_id == "1":
        return User("1", DEFAULT_USERNAME)
    return None

# Paths
student_images_folder = os.path.join(os.path.dirname(__file__), "student_images")
attendance_folder = "attendance_records"

if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

# Global variables
video_capture = cv2.VideoCapture(0)
current_frame = None

def get_attendance_file(subject):
    """Return the Excel file path for a specific subject."""
    return os.path.join(attendance_folder, f"{subject}_attendance.xlsx")

def load_reference_encodings(folder_path):
    """Load face encodings from reference images."""
    encodings = {}
    for file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file)
        try:
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            name = os.path.splitext(file)[0]  # Filename without extension
            encodings[name] = face_encoding
        except IndexError:
            print(f"No face detected in {file}. Skipping...")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    print(f"Loaded {len(encodings)} reference encodings.")
    return encodings

def mark_attendance_in_excel(name, rollno, subject):
    """Mark attendance in the subject-specific Excel file."""
    file_path = get_attendance_file(subject)
    print(f"Marking attendance for subject: {subject}, File: {file_path}")

    today = datetime.now().day
    column_name = f"Day{today}"

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        columns = ["RollNo", "Name"] + [f"Day{i}" for i in range(1, 31)]
        df = pd.DataFrame(columns=columns)

    if rollno in df["RollNo"].values:
        df.loc[df["RollNo"] == rollno, column_name] = "P"
    else:
        new_row = {"RollNo": rollno, "Name": name, column_name: "P"}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(file_path, index=False)

def generate_frames():
    """Generate video frames from webcam."""
    global current_frame
    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture video frame")
            break
        current_frame = frame  # Update the global variable
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            login_user(User("1", username))
            return redirect(url_for("protected"))
        else:
            flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    """Logout the user."""
    logout_user()
    return redirect(url_for("login"))

@app.route("/protected")
@login_required
def protected():
    """Protected page for attendance."""
    return render_template("protected.html", username=current_user.username)

@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capture", methods=["POST"])
def capture():
    """Capture frame and mark attendance for a specific subject."""
    global current_frame
    subject = request.args.get("subject")
    print(f"Received request to mark attendance for subject: {subject}")

    if not subject:
        return jsonify({"message": "Subject not specified.", "success": False})

    if current_frame is None:
        return jsonify({"message": "No frame captured from the webcam. Try again.", "success": False})

    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)
    if not face_encodings:
        return jsonify({"message": "No face detected. Please ensure your face is clearly visible to the webcam.", "success": False})

    reference_encodings = load_reference_encodings(student_images_folder)
    for name, encoding in reference_encodings.items():
        match = face_recognition.compare_faces([encoding], face_encodings[0], tolerance=0.6)
        if match[0]:
            rollno, student_name = name.split("_", 1)  # Assume filenames are "RollNo_Name.jpg"
            mark_attendance_in_excel(student_name, rollno, subject)
            return jsonify({"message": f"Attendance taken successfully for {student_name} (RollNo: {rollno}) in {subject}.", "success": True})

    return jsonify({"message": "No matching face found in the reference images.", "success": False})

@app.route("/export_csv", methods=["GET"])
@login_required
def export_csv():
    """Export attendance data for a subject to CSV."""
    subject = request.args.get("subject")
    if not subject:
        return "Subject not specified", 400

    file_path = get_attendance_file(subject)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Attendance data not found for the specified subject", 404


@app.route("/attendance_insights")
@login_required
def attendance_insights():
    """Attendance Insights Dashboard for a specific subject."""
    subject = request.args.get("subject", "python") 
    if not subject:
        return "Subject not specified", 400

    file_path = get_attendance_file(subject)
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df["Attendance Percentage"] = df.iloc[:, 2:].apply(
            lambda row: (row == "P").sum() / len(row) * 100, axis=1
        )

        low_attendance = df[df["Attendance Percentage"] < 75]

        # Generate a bar chart for low attendance
        fig, ax = plt.subplots()
        ax.bar(low_attendance["Name"], low_attendance["Attendance Percentage"], color="red")
        ax.set_xlabel("Student Name")
        ax.set_ylabel("Attendance Percentage")
        ax.set_title(f"Low Attendance for {subject} (< 75%)")

        # Save the chart to a temporary image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode("utf-8")
        plt.close()

        return render_template(
            "attendance_insights.html",
            img_b64=img_b64,
            low_attendance=low_attendance,
            subject=subject,
        )
    return "No attendance data found for the specified subject", 404

if __name__ == "__main__":
    app.run(debug=True)
