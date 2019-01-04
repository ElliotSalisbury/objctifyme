from django.db.models import Count, Q, F, Avg
from django.shortcuts import render, render_to_response
from django.views.generic.list import ListView

from rateme.models import Submission

def index(request):
    return render_to_response('objctify/index.html')
def about(request):
    return render_to_response('objctify/about.html')
def author(request):
    return render_to_response('objctify/author.html')

# Create your views here.
class SubmissionListView(ListView):

    model = Submission
    # paginate_by = 100  # if pagination is desired

    def get_queryset(self):
        return Submission.objects.filter(has_images=True, calculated_rating__isnull=False)\
            .annotate(usable_comments_count=Count("comments", filter=Q(comments__rating__isnull=False)))\
            .filter(usable_comments_count__gte=10) \
            .order_by("-calculated_rating")
            # .annotate(avg_rating=Avg("comments__rating", filter=Q(comments__rating__isnull=False)))\
            # .order_by("-calculated_rating")