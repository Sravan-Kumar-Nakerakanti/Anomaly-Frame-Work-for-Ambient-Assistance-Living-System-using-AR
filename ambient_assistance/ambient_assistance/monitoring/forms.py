# monitoring/forms.py
from django import forms
from django.contrib.auth.models import User
from .models import HealthRecord, HealthRecommendation, ElderParent, Profile, HealthTask,TaskInput

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class ElderParentForm(forms.ModelForm):
    class Meta:
        model = ElderParent
        fields = ['name', 'health_conditions', 'front_image', 'back_image', 'left_image', 'right_image']

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['first_name', 'last_name', 'email', 'profile_image']

class HealthRecordForm(forms.ModelForm):
    class Meta:
        model = HealthRecord
        fields = ['elder_parent_name', 'anthropometric_measurements', 'health_disease', 'task_specific_actions']

class HealthRecommendationForm(forms.ModelForm):
    class Meta:
        model = HealthRecommendation
        fields = ['recommendations']

class HealthTaskForm(forms.ModelForm):
    class Meta:
        model = HealthTask
        fields = ['task_name', 'description', 'date']

class TaskInputForm(forms.ModelForm):
    class Meta:
        model = TaskInput
        fields = ['task_name', 'description']