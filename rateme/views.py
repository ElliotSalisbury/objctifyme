from django.db.models import Count, Q
from django.shortcuts import render_to_response
from django.views.generic import DetailView
from django.views.generic.list import ListView
from rest_framework.generics import ListAPIView
from rest_framework.pagination import PageNumberPagination

from rateme.models import Submission, User
from rateme.serializers import SubmissionSerializer

def graph(request):
    return render_to_response('rateme/submission_graph.html')

# Create your views here.
class SubmissionListView(ListView):
    model = Submission

    paginate_by = 100

    def get_queryset(self):
        return Submission.objects.filter(has_images=True, calculated_rating__isnull=False, gender='F') \
            .annotate(usable_comments_count=Count("comments", filter=Q(comments__rating__isnull=False))) \
            .filter(usable_comments_count__gte=10) \
            .order_by("-calculated_rating")


# Create your views here.
class SubmissionDetailView(DetailView):
    model = Submission

    # def get_context_data(self, **kwargs):
    #     context = super().get_context_data(**kwargs)
    #     context['now'] = timezone.now()
    #     return context

class AuthorDetailView(DetailView):
    model = User


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 1000
    page_size_query_param = 'page_size'
    max_page_size = 1000


class API_SubmissionListView(ListAPIView):
    serializer_class = SubmissionSerializer
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        submissions = Submission.objects.filter(has_images=True, calculated_rating__isnull=False)

        gender = self.request.query_params.get('gender', None)
        if gender is not None:
            submissions = submissions.filter(gender=gender)

        submissions = submissions.annotate(
            usable_comments_count=Count("comments", filter=Q(comments__rating__isnull=False))) \
            .filter(usable_comments_count__gte=10) \
            .order_by("-calculated_rating")

        return submissions
