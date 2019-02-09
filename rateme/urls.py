from django.urls import path

from rateme import views

urlpatterns = [
    path(r'', views.index, name='home'),
    path(r'about/', views.about, name='about'),
    path(r'author/', views.author, name='author'),

    path(r'3D/AverageFaces', views.morphFaces, name='averageFaces'),
    path(r'3D/morphFaces', views.morphFaces, name='morphFaces'),

    path(r'uploadImage', views.index, name='upload'),
    path(r'api/startProcess', views.index, name='api_StartProcess'),
    path(r'api/getResults', views.index, name='api_GetResults'),

    path(r'submissions/', views.SubmissionListView.as_view(), name='submission-list'),
    path(r'submission/<pk>/', views.SubmissionDetailView.as_view(), name='submission-detail'),
    path(r'author/<pk>/', views.AuthorDetailView.as_view(), name='author-detail'),

    path(r'graph/', views.graph, name='graph'),

    path(r'api/submissions', views.API_SubmissionListView.as_view(), name='api-submissions')
]