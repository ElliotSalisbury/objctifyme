from django.urls import path

from rateme import views

urlpatterns = [
    path(r'submissions/', views.SubmissionListView.as_view(), name='submission-list'),
    path(r'submission/<pk>/', views.SubmissionDetailView.as_view(), name='submission-detail'),
    path(r'author/<pk>/', views.AuthorDetailView.as_view(), name='author-detail'),

    path(r'graph/', views.graph, name='graph'),

    path(r'api/submissions', views.API_SubmissionListView.as_view(), name='api-submissions')
]