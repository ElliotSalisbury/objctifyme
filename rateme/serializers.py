from rest_framework import serializers
from rateme.models import Submission
import time

class TimestampField(serializers.Field):
    def to_representation(self, value):
        return int(time.mktime(value.timetuple()))

class MySerializer(serializers.ModelSerializer):
    ts = TimestampField(source="my_fieldname")

class SubmissionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Submission
        fields = ('id', 'author_id', 'age', 'gender', 'title', 'calculated_rating', 'created', 'permalink', 'score')

    created = TimestampField()