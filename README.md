# Anomaly-Frame-Work-for-Ambient-Assistance-Living-System-using-AR
# Overview
This project focuses on Anomaly Frame Workfor in Ambient Assisted Living environments. Using augmented reality and machine learning, it supports elderly or disabled individuals by identifying their actions and offering real-time assistance based on those actions.

# Features
- Personalized 3D avatars for user interaction
- Live video monitoring with AWS cloud storage
- Action recognition and anomaly detection
- Health recommendations based on actions
- Web interface built with Django and HTML/CSS

# Tools Used
- Backend: Python, Django
- Frontend: HTML, CSS, JavaScript
- Machine Learning: Action recognition algorithm
- Cloud: AWS (for live video storage)
- Others: OpenCV, Mediapipe, NumPy, Pandas

# Setup Instructions
**1. Create a Virtual Environment**
```python -m venv env```
```source env\Scripts\activate```

**2. Configure Django**
Make sure the settings.py file has correct database configurations and MEDIA_URL, STATIC_URL paths set properly.

**3. Apply Migrations**
```python manage.py makemigrations```
```python manage.py migrate```

**4. Run the Development Server**
```python manage.py runserver```

# How to Use
- Register/login as a user.
- Access the live video monitoring page.
- Perform actions; the system will recognize and process them.
- View health recommendation or alerts based on activity.
