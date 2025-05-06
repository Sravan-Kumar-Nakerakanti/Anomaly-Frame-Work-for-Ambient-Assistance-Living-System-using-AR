# monitoring/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AnomalyViewSet

router = DefaultRouter()
router.register(r'anomalies', AnomalyViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
