# monitoring/routing.py

from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/tasks/', consumers.TaskConsumer.as_asgi()),
    path(r'ws/live_monitoring/$', consumers.LiveMonitoringConsumer.as_asgi()),
]
