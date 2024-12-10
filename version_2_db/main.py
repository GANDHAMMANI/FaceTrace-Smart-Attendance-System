from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import face_recognition
import os
import sqlite3
from datetime import datetime
import pandas as pd

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
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Retrieve all subjects from the database
    cursor.execute("SELECT * FROM subjects")
    subjects = cursor.fetchall()

    conn.close()
    
    # Pass the subjects to the template
    return render_template("protected.html", username=current_user.username, subjects=subjects)

# Paths
student_images_folder = os.path.join(os.path.dirname(__file__), "student_images")

# Global variables
video_capture = cv2.VideoCapture(0)
current_frame = None

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

def mark_attendance_in_db(name, rollno, subject_id=1):
    """Mark attendance in the database."""
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Check if record already exists for today
    cursor.execute(
        """
        SELECT * FROM attendance WHERE roll_no = ? AND subject_id = ? AND date = ?
        """,
        (rollno, subject_id, today),
    )
    existing_record = cursor.fetchone()

    if not existing_record:
        # Insert new record
        cursor.execute(
            """
            INSERT INTO attendance (roll_no, subject_id, date, status, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (rollno, subject_id, today, "P", timestamp),
        )
        conn.commit()

    conn.close()

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

@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capture", methods=["POST"])
def capture():
    """Capture frame and mark attendance."""
    subject_id = request.form.get("subject_id")  # Get the subject ID from the form submission
    if not subject_id:
        return jsonify({"message": "No subject selected. Please select a subject."})

    # Capture the frame
    global current_frame
    if current_frame is None:
        return jsonify({"message": "No frame captured from the webcam. Try again."})
    
    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)
    if not face_encodings:
        return jsonify({"message": "No face detected. Please ensure your face is clearly visible to the webcam."})
    
    # Load reference encodings
    reference_encodings = load_reference_encodings(student_images_folder)
    
    for name, encoding in reference_encodings.items():
        match = face_recognition.compare_faces([encoding], face_encodings[0], tolerance=0.6)
        if match[0]:
            rollno, student_name = name.split("_", 1)  # Assume filenames are "RollNo_Name.jpg"
            
            # Mark attendance in the database
            mark_attendance_in_db(student_name, rollno, subject_id)
            return jsonify({"message": f"Attendance taken successfully for {student_name} (RollNo: {rollno}) in the selected subject."})
    
    return jsonify({"message": "No matching face found in the reference images."})


@app.route("/attendance_insights")
@login_required
def attendance_insights():
    """Attendance Insights Dashboard."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Retrieve attendance data with calculated attendance percentage
    cursor.execute("""
        SELECT s.roll_no, s.name, 
               (COUNT(a.attendance_id) * 100.0 / 30) AS attendance_percentage
        FROM attendance a
        JOIN students s ON s.roll_no = a.roll_no
        GROUP BY s.roll_no, s.name
    """)
    insights = cursor.fetchall()

    # Debugging: Check if insights data is correct
    print("Insights:", insights)

    # Filter students with low attendance (below 75%)
    low_attendance = [student for student in insights if student[2] < 75]
    # Debugging: Check low attendance students
    print("Low Attendance:", low_attendance)

    # Get the top students with highest attendance (e.g., top 10 students)
    top_attendance = sorted(insights, key=lambda x: x[2], reverse=True)[:10]
    # Debugging: Check top attendance students
    print("Top Attendance:", top_attendance)

    conn.close()

    # Pass data to the template
    return render_template(
        "attendance_insights.html",
        insights=insights,
        low_attendance=low_attendance,
        top_attendance=top_attendance
    )



@app.route("/export_csv")
@login_required
def export_csv():
    """Export attendance data to CSV."""
    conn = sqlite3.connect("attendance.db")
    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    conn.close()
    csv_path = "attendance_data.csv"
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
