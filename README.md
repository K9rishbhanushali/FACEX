# Smart Attendance

Real-Time Multi-Face Recognition Based Smart Attendance System built using Python, Flask, OpenCV, InsightFace, and SQLite.

## Description

Smart Attendance is an AI-powered attendance management project designed to automate classroom attendance using face recognition. The system detects and recognizes multiple students in real time, marks attendance automatically, stores records in a local database, and provides attendance analytics through reports, graphs, and defaulter tracking.

The project was developed to solve common problems in traditional attendance systems such as:

- manual roll-call consuming lecture time
- proxy attendance
- human error in record keeping
- limited reporting and analytics
- contact-based biometric limitations

## Key Features

- Teacher login and class management
- Student registration using face samples
- Real-time multi-face attendance marking
- Student attendance report page
- Attendance graphs and defaulter list
- Backend liveness validation to reduce proxy attendance
- Anti-photo / anti-video proxy checks
- Adaptive lighting correction for better recognition in bright or uneven rooms
- Teacher attendance editing for missed or late attendance cases

## Tech Stack

- Python
- Flask
- OpenCV
- NumPy
- Pandas
- OpenPyXL
- InsightFace
- Ultralytics YOLO
- SQLite

## Project Structure

```text
smart_attendance/
├── app.py
├── fastapi_app.py
├── requirements.txt
├── database.db
├── encodings.pkl
├── templates/
└── static/
```

## How It Works

1. Teacher logs into the system.
2. Teacher creates a class and registers students.
3. Student face data is stored as embeddings.
4. During attendance, the system captures live camera frames.
5. Faces are detected and matched with registered encodings.
6. Backend validation checks liveness and proxy-risk patterns.
7. Valid attendance entries are stored in the database.
8. Teachers can view reports, graphs, and defaulter analysis.

## Installation

1. Clone or download the repository.
2. Open terminal in the project folder.
3. Create a virtual environment:

```bash
python -m venv attendance_env
```

4. Activate the environment:

```bash
attendance_env\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the application:

```bash
python app.py
```

7. Open in browser:

```text
http://127.0.0.1:5000
```

## Notes

- This project is mainly designed for local deployment because live attendance requires direct camera access.
- Cloud deployment is possible for dashboard and report modules, but the live camera-based recognition pipeline would need architectural changes.
- A dedicated `best.pt` mask model is not currently included in this repository.

## Future Scope

- Cloud-based attendance storage
- Mobile app integration
- ERP / college management system integration
- Stronger masked-face recognition
- CCTV-based classroom deployment
- Improved anti-spoofing and liveness models

## Authors

- Krish Bhanushali
- Deepisha Chugh
- Het Finavia
- Nit Jain
