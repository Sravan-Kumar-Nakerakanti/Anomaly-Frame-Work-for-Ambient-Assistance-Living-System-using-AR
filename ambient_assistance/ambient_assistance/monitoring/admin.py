# monitoring/admin.py

from django.contrib import admin
from .models import ElderParent, Profile

admin.site.register(ElderParent)
admin.site.register(Profile)
