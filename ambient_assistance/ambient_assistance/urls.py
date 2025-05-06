"""
URL configuration for ambient_assistance project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# ambient_assitance/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from monitoring import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.login_view, name='login'),
    path('profile/', views.profile_view, name='profile'),
    path('update_profile/', views.update_profile_view, name='update_profile'),
    path('health_recommendations/', views.health_recommendations_view, name='health_recommendations'),
    path('upload_anomaly/', views.upload_anomaly_view, name='upload_anomaly'),
    path('health_task/', views.health_task_view, name='health_task'),
    path('anomalies/', views.view_anomalies_view, name='view_anomalies'),
    path('logout/', views.logout_view, name='logout'),
    path('ar_view/', views.ar_view, name='ar_view'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('register/', views.register_view, name='register'),
    path('home/', views.home_view, name='home'),
    path('delete-task/<int:task_id>/', views.delete_task_view, name='delete_task'),
    path('live_web_monitoring/', views.live_web_monitoring, name='live_web_monitoring'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)