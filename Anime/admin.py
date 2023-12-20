from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

# Register your models here.
from .models import Anime
from .models import MyUser

admin.site.register(MyUser, UserAdmin)
admin.site.register(Anime)
