from django.urls import path

from rateme.views import SubmissionListView

urlpatterns = [
    path('', SubmissionListView.as_view(), name='submission-list'),
]