"""
ASGI config for ambient_assistance project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""
#asgi.py

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ambient_assistance.settings')

# application = get_asgi_application()
# asgi.py
import channels
import channels.routing
import channels.auth
import channels.generic.websocket

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import monitoring.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ambient_assistance.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            monitoring.routing.websocket_urlpatterns
        )
    ),
})

