from django.db.models import Count, Q, F, Avg
from django.shortcuts import render
from django.views.generic.list import ListView

from rateme.models import Submission

# Create your views here.
class SubmissionListView(ListView):

    model = Submission
    # paginate_by = 100  # if pagination is desired

    def get_queryset(self):
        return Submission.objects.filter(has_images=True)\
            .annotate(usable_comments_count=Count("comments", filter=Q(comments__rating__isnull=False)))\
            .filter(usable_comments_count__gte=10)\
            .annotate(avg_rating=Avg("comments__rating", filter=Q(comments__rating__isnull=False)))\
            .order_by("-avg_rating")