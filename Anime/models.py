from django.db import models
from django.contrib.auth.models import AbstractUser

class Anime(models.Model):
    title = models.CharField(max_length=255)
    rating = models.IntegerField()
    genres = models.CharField(max_length=255)

    def __str__(self):
        return self.title
    
class MyUser(AbstractUser):
    last_watched_anime = models.CharField(max_length=255, null=True, blank=True)