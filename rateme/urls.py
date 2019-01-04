from django.urls import path

from rateme import views

urlpatterns = [
    path(r'', views.index, name='home'),
    path(r'about/', views.about, name='about'),
    path(r'author/', views.author, name='author'),

    path(r'3D/AverageFaces', views.index, name='averageFaces'),
    path(r'3D/morphFaces', views.index, name='morphFaces'),

    path(r'uploadImage', views.index, name='upload'),
    path(r'api/startProcess', views.index, name='api_StartProcess'),
    path(r'api/getResults', views.index, name='api_GetResults'),

    path(r'submissions/', views.SubmissionListView.as_view(), name='submission-list'),
]