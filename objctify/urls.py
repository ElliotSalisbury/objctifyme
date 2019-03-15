from django.urls import path

from objctify import views

urlpatterns = [
    path(r'', views.index, name='home'),
    path(r'about/', views.about, name='about'),
    path(r'author/', views.author, name='author'),

    path(r'3D/AverageFaces', views.averageFaces, name='averageFaces'),
    path(r'3D/morphFaces', views.morphFaces, name='morphFaces'),

    path(r'uploadImage', views.upload, name='upload'),
    path(r'api/startProcess', views.startImageProcessing, name='api_StartProcess'),
    path(r'api/getResults', views.getSwap, name='api_GetResults'),
]