# monitoring/models.py

from django.db import models
from django.contrib.auth.models import User

class ElderParent(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    health_conditions = models.TextField()
    front_image = models.ImageField(upload_to='elder_parent_images/')
    back_image = models.ImageField(upload_to='elder_parent_images/')
    left_image = models.ImageField(upload_to='elder_parent_images/')
    right_image = models.ImageField(upload_to='elder_parent_images/')
    avatar_model = models.FileField(upload_to='avatars/', null=True, blank=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Elder Parent'
        verbose_name_plural = 'Elder Parents'


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    email = models.EmailField()
    profile_image = models.ImageField(upload_to='profile_images/', blank=True, null=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class HealthRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    elder_parent_name = models.CharField(max_length=100)
    anthropometric_measurements = models.TextField()
    health_disease = models.CharField(max_length=100)  # Changed to CharField
    task_specific_actions = models.TextField()

    def __str__(self):
        return f"{self.elder_parent_name}'s Health Record"


class HealthRecommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    health_disease = models.CharField(max_length=255)
    recommendations = models.JSONField()  # Or a TextField if you plan to store JSON as a string

    def __str__(self):
        return f'Recommendations for {self.health_disease} by {self.user.username}'


class HealthTask(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    elder_parent = models.ForeignKey(ElderParent, on_delete=models.CASCADE)
    task_name = models.CharField(max_length=100)
    description = models.TextField()
    date = models.DateField()

    def __str__(self):
        return f"Task: {self.task_name} for {self.elder_parent.name}"

    def as_dict(self):
        return {
            'task_name': self.task_name,
            'description': self.description,
            'date': self.date.strftime('%Y-%m-%d')
        }


class Anomaly(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    elder_parent = models.ForeignKey(ElderParent, on_delete=models.CASCADE)
    video = models.FileField(upload_to='anomalies/')
    detected_at = models.DateTimeField()
    action = models.CharField(max_length=50)
    task = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Anomaly {self.id} for {self.elder_parent.name}"

class TaskInput(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    elder_parent = models.ForeignKey('ElderParent', on_delete=models.CASCADE)
    task_name = models.CharField(max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Task: {self.task_name} for {self.elder_parent.name}"