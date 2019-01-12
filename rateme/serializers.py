import json
import time

from rest_framework import serializers

from rateme.models import Submission, SubmissionImage, FaceProcessing


class TimestampField(serializers.Field):
    def to_representation(self, value):
        return int(time.mktime(value.timetuple())) * 1000


class JSONField(serializers.Field):
    def to_representation(self, value):
        return json.loads(value)


class FaceProcessingSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceProcessing
        fields = ('shape_coefficients', 'color_coefficients', 'expression_coefficients',)

    shape_coefficients = JSONField()
    color_coefficients = JSONField()
    expression_coefficients = JSONField()


class SubmissionImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubmissionImage
        fields = ('face_processings',)

    face_processings = FaceProcessingSerializer(many=True)


class SubmissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Submission
        fields = (
            'id', 'author_id', 'age', 'gender', 'title', 'calculated_rating', 'created', 'permalink', 'score', 'images')

    created = TimestampField()
    images = SubmissionImageSerializer(many=True)
