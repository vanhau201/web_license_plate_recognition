from django.db import models
from django.contrib.auth.models import User
# Ceate your models here.


class Image(models.Model):
    image = models.ImageField(upload_to="image/")
    confidences = models.FloatField()
    result = models.CharField(max_length=10)
    status = models.BooleanField(default=True)
    time_create = models.DateTimeField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
