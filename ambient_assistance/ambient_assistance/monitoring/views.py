# monitoring/views.py
import os
import subprocess
import pandas as pd
from django.utils import timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import boto3
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from rest_framework import viewsets
from rest_framework.response import Response
from .models import ElderParent, Profile, HealthRecord, HealthTask, Anomaly, HealthRecommendation,TaskInput
from .forms import LoginForm, ElderParentForm, ProfileUpdateForm, HealthRecordForm, HealthTaskForm, UserForm,TaskInputForm
from .anomaly_detection import detect_anomalies
from .serializers import AnomalySerializer
import cv2


# Load datasets
description_df = pd.read_csv(r'C:\Users\SRAVA\Downloads\archive (5)\description.csv')
diets_df = pd.read_csv(r'C:\Users\SRAVA\Downloads\archive (5)\diets.csv')
medications_df = pd.read_csv(r'C:\Users\SRAVA\Downloads\archive (5)\medications.csv')
precautions_df = pd.read_csv(r'C:\Users\SRAVA\Downloads\archive (5)\precautions_df.csv')
symptoms_df = pd.read_csv(r'C:\Users\SRAVA\Downloads\archive (5)\symtoms_df.csv')
workout_df = pd.read_csv(r'C:\Users\SRAVA\Downloads\archive (5)\workout_df.csv')

# Combine symptoms into a single string for each disease
disease_symptoms = symptoms_df.groupby('Disease').apply(
    lambda x: ' '.join(x.iloc[:, 2:].dropna().astype(str))
).reset_index()
disease_symptoms.columns = ['Disease', 'Symptoms']

# Convert symptoms to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(disease_symptoms['Symptoms'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
disease_similarity_df = pd.DataFrame(cosine_sim, index=disease_symptoms['Disease'], columns=disease_symptoms['Disease'])

@login_required
def recommend_diseases(disease_name, top_n=5):
    if disease_name not in disease_similarity_df.index:
        return []
    
    # Get similarity scores
    sim_scores = disease_similarity_df[disease_name]
    
    # Get the top N similar diseases
    top_diseases = sim_scores.sort_values(ascending=False).head(top_n + 1).index
    return [disease for disease in top_diseases if disease != disease_name]

@login_required
def get_health_recommendations(disease_name):
    if disease_name not in disease_similarity_df.index:
        return {}
    
    # Get disease recommendations
    recommended_diseases = recommend_diseases(disease_name)
    
    # Initialize recommendations dictionary
    recommendations = {}
    
    for disease in [disease_name] + recommended_diseases:
        desc = description_df[description_df['Disease'] == disease]['Description'].values
        diet = diets_df[diets_df['Disease'] == disease]['Diet'].values
        medication = medications_df[medications_df['Disease'] == disease]['Medication'].values
        precaution = precautions_df[precautions_df['Disease'] == disease].iloc[0, 2:].dropna().values
        symptom = symptoms_df[symptoms_df['Disease'] == disease].iloc[0, 2:].dropna().values
        workout = workout_df[workout_df['Disease'] == disease]['Workout'].values
        
        recommendations[disease] = {
            'description': desc[0] if len(desc) > 0 else "",
            'diet': diet[0] if len(diet) > 0 else "",
            'medication': medication[0] if len(medication) > 0 else "",
            'precaution': ", ".join(precaution) if len(precaution) > 0 else "",
            'symptom': ", ".join(symptom) if len(symptom) > 0 else "",
            'workout': workout[0] if len(workout) > 0 else ""
        }
    
    return recommendations

@login_required
def create_avatar_from_images(elder_parent_id):
    elder_parent = ElderParent.objects.get(id=elder_parent_id)
    
    front_image_path = os.path.join(settings.MEDIA_ROOT, elder_parent.front_image.name)
    back_image_path = os.path.join(settings.MEDIA_ROOT, elder_parent.back_image.name)
    left_image_path = os.path.join(settings.MEDIA_ROOT, elder_parent.left_image.name)
    right_image_path = os.path.join(settings.MEDIA_ROOT, elder_parent.right_image.name)

    output_path = os.path.join(settings.MEDIA_ROOT, 'avatars', f'{elder_parent_id}_avatar.glb')

    subprocess.run([
        'blender',
        '--background',
        '--python', r'C:\Users\SRAVA\Downloads\blender_script.py',
        '--',
        front_image_path,
        back_image_path,
        left_image_path,
        right_image_path,
        output_path
    ])

    elder_parent.avatar_model = os.path.join('avatars', f'{elder_parent_id}_avatar.glb')
    elder_parent.save()

def login_view(request):
    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    return render(request, 'login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('profile')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

@login_required
def profile_view(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        parent_form = ElderParentForm(request.POST, request.FILES, instance=request.user.elderparent)
        profile_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if user_form.is_valid() and parent_form.is_valid() and profile_form.is_valid():
            user_form.save()
            parent_form.save()
            profile_form.save()
            return redirect('home')
    else:
        user_form = UserForm(instance=request.user)
        parent_form = ElderParentForm(instance=request.user.elderparent)
        profile_form = ProfileUpdateForm(instance=request.user.profile)
    return render(request, 'profile.html', {'user_form': user_form, 'parent_form': parent_form, 'profile_form': profile_form})

@login_required
def home_view(request):
    return render(request, 'home.html')

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def ar_view(request):
    return render(request, 'ar_view.html')

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if success:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return None

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@login_required
def video_feed(request):
    latest_anomaly = Anomaly.objects.filter(user=request.user).order_by('-detected_at').first()
    if latest_anomaly:
        return StreamingHttpResponse(gen(Camera()), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        context = {
            'message': 'Mobile camera is turned off'
        }
        return render(request, 'video_feed.html', context)


from django.core.serializers.json import DjangoJSONEncoder
import json
import logging
logger = logging.getLogger(__name__)

@login_required
def health_recommendations_view(request):
    logger.debug(f"Request object type: {type(request)}")
    logger.debug(f"Request.user: {request.user} (Type: {type(request.user)})")

    try:
        elder_parent = ElderParent.objects.get(user=request.user)
    except ElderParent.DoesNotExist:
        elder_parent = None

    if request.method == 'POST':
        task_form = TaskInputForm(request.POST)
        health_record_form = HealthRecordForm(request.POST)

        if task_form.is_valid():
            task_input = task_form.save(commit=False)
            task_input.user = request.user
            task_input.elder_parent = elder_parent
            task_input.save()
            return redirect('health_recommendations')

        if health_record_form.is_valid():
            health_record = health_record_form.save(commit=False)
            health_record.user = request.user
            health_record.save()

            logger.debug(f"Health Record Disease: {health_record.health_disease} (Type: {type(health_record.health_disease)})")
            health_recommendations = get_health_recommendations(health_record.health_disease)

            recommendation_obj = HealthRecommendation(
                user=request.user,
                health_disease=health_record.health_disease,
                recommendations=json.dumps(health_recommendations, cls=DjangoJSONEncoder)
            )
            recommendation_obj.save()

            return redirect('health_recommendations')

        else:
            logger.error(f"Health Record Form Errors: {health_record_form.errors}")

    else:
        task_form = TaskInputForm()
        health_record_form = HealthRecordForm()

    recommendations = HealthRecommendation.objects.filter(user=request.user).last()
    tasks = TaskInput.objects.filter(user=request.user)

    return render(request, 'health_recommendations.html', {
        'task_form': task_form,
        'form': health_record_form,
        'recommendations': recommendations,
        'tasks': tasks,
    })

@login_required
def upload_anomaly_view(request):
    if request.method == 'POST':
        video = request.FILES['video']
        elder_parent = ElderParent.objects.get(user=request.user)
        anomaly_path = os.path.join(settings.MEDIA_ROOT, 'anomalies', video.name)
        
        with open(anomaly_path, 'wb+') as destination:
            for chunk in video.chunks():
                destination.write(chunk)
        
        detected_action = detect_anomalies(anomaly_path)
        anomaly_clip_path = anomaly_path.replace('.mp4', '_anomaly.avi')
        
        s3_client = boto3.client('s3')
        s3_client.upload_file(anomaly_clip_path, settings.AWS_STORAGE_BUCKET_NAME, anomaly_clip_path.split('/')[-1])

        anomaly = Anomaly(user=request.user, elder_parent=elder_parent, video=anomaly_clip_path, detected_at=timezone.now(), action=detected_action)
        anomaly.save()

        return redirect('anomaly_success')
    return render(request, 'upload_anomaly.html')

@login_required
def health_task_view(request):
    if request.method == 'POST':
        form = HealthTaskForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('health_task_success')
    else:
        form = HealthTaskForm()
    return render(request, 'health_task.html', {'form': form})

@login_required
def view_anomalies_view(request):
    anomalies = Anomaly.objects.filter(user=request.user)
    return render(request, 'view_anomalies.html', {'anomalies': anomalies})

class AnomalyViewSet(viewsets.ModelViewSet):
    queryset = Anomaly.objects.all()
    serializer_class = AnomalySerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data, files=request.FILES)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=201, headers=headers)

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)

@login_required
def delete_task_view(request, task_id):
    task = TaskInput.objects.get(id=task_id, user=request.user)
    task.delete()
    return redirect('health_recommendations')

def update_profile_view(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        parent_form = ElderParentForm(request.POST, request.FILES, instance=request.user.elderparent)
        profile_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        
        if user_form.is_valid() and parent_form.is_valid() and profile_form.is_valid():
            user_form.save()
            parent_form.save()
            profile_form.save()
            return redirect('home')
    else:
        user_form = UserForm(instance=request.user)
        parent_form = ElderParentForm(instance=request.user.elderparent)
        profile_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'form': user_form,  # You might want to customize this if using multiple forms
        'parent_form': parent_form,
        'profile_form': profile_form,
    }
    return render(request, 'update_profile.html', context)


# views.py
import cv2
import numpy as np
import base64
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync

# Load the trained model
model = tf.keras.models.load_model(r'C:/Users/SRAVA/action_recog.h5')

# Define categories
categories = ['wave', 'walk', 'turn', 'throw', 'talk', 'stand', 'smile', 'situp', 'sit', 'shake_hands',
              'run', 'push', 'punch', 'pour', 'pick', 'laugh', 'jump', 'hug', 'hit', 'handstand',
              'fall_floor', 'eat', 'drink', 'dribble', 'climb_stairs', 'climb', 'clap', 'chew',
              'catch', 'brush_hair']

def live_web_monitoring(request):
    return render(request, 'live_monitoring.html')

class LiveMonitoringConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        frame_data = text_data.split(",")[1]
        frame_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process the frame and predict action
        action = self.predict_action(frame)
        async_to_sync(self.send)(text_data=json.dumps({'action': action}))

    def predict_action(self, frame):
        resized_frame = cv2.resize(frame, (64, 64))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = gray_frame.reshape(1, 64, 64, 1).astype('float32') / 255.0

        predictions = model.predict(processed_frame)
        predicted_class = np.argmax(predictions[0])
        return categories[predicted_class]
