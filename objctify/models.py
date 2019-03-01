from django.db import models


class ImageProcessingRequest(models.Model):
    date = models.DateTimeField(auto_now_add=True, blank=True)
    type = models.TextField()
    ip = models.GenericIPAddressField()


class UploadedImage(models.Model):
    filename = models.TextField()
    image = models.ImageField(upload_to="uploaded/")
    request = models.ForeignKey(ImageProcessingRequest, on_delete=models.CASCADE)
