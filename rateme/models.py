from django.db import models
from worker_reliability import RATINGS_COUNT_THRESH, ALL_MEAN, ALL_STD, NORM_MEAN, NORM_STD
import numpy as np

class User(models.Model):
    id = models.CharField(max_length=256, primary_key=True)

    def __str__(self):
        return self.id

    def get_mean_std(self):
        ratings = []
        for comment in self.comments.filter(rating__isnull=False):
            ratings.append(comment.rating)

        if len(ratings) < RATINGS_COUNT_THRESH:
            mean, std = ALL_MEAN, ALL_STD
        else:
            mean, std = np.mean(ratings), np.std(ratings)
            std = max(std, 0.0001)

        return mean, std


class Submission(models.Model):
    id = models.CharField(max_length=256, primary_key=True)
    title = models.CharField(max_length=256)
    has_images = models.BooleanField()
    age = models.IntegerField(null=True)
    gender = models.CharField(max_length=1, null=True)
    author = models.ForeignKey("User", on_delete=models.CASCADE, related_name="submissions")
    created = models.DateTimeField()
    permalink = models.URLField()
    score = models.IntegerField()
    upvote_ratio = models.FloatField()

    calculated_rating = models.FloatField(null=True)

    def __str__(self):
        return self.title

    @property
    def hotness(self):
        if self.calculated_rating is None:
            return self.calculate_rating()

        return self.calculated_rating

    @property
    def usable_comments(self):
        return self.comments.filter(rating__isnull=False)

    def calculate_rating(self):
        actual_ratings = []
        for comment in self.usable_comments:
            rating = comment.rating
            user_mean, user_std = comment.author.get_mean_std()

            actual_rating = (rating - user_mean) / user_std
            actual_rating = (actual_rating * NORM_STD) + NORM_MEAN

            actual_ratings.append(actual_rating)

        if len(actual_ratings) > 0:
            actual_rating = np.mean(actual_ratings)
        else:
            actual_rating = None

        self.calculated_rating = actual_rating
        self.save()

        return self.calculated_rating


class SubmissionImage(models.Model):
    submission = models.ForeignKey("Submission", on_delete=models.CASCADE, related_name="images")
    image = models.ImageField()
    face_count = models.IntegerField()


class ImageProcessing(models.Model):
    image = models.OneToOneField("SubmissionImage", on_delete=models.CASCADE, primary_key=True,
                                 related_name="processing")
    texture = models.ImageField()
    shape_coefficients = models.CharField(max_length=4096)
    color_coefficients = models.CharField(max_length=4096)
    expression_coefficients = models.CharField(max_length=4096)
    pitch = models.FloatField()
    yaw = models.FloatField()
    roll = models.FloatField()


class FaceProcessing(models.Model):
    image = models.ForeignKey("SubmissionImage", on_delete=models.CASCADE, related_name="face_processings")
    texture = models.ImageField()
    shape_coefficients = models.CharField(max_length=4096)
    color_coefficients = models.CharField(max_length=4096)
    expression_coefficients = models.CharField(max_length=4096)
    pitch = models.FloatField()
    yaw = models.FloatField()
    roll = models.FloatField()


class Comment(models.Model):
    id = models.CharField(max_length=256, primary_key=True)
    submission = models.ForeignKey("Submission", on_delete=models.CASCADE, related_name="comments")
    body = models.CharField(max_length=4096)
    rating = models.IntegerField(null=True)
    decimal = models.FloatField(null=True)
    author = models.ForeignKey("User", on_delete=models.CASCADE, related_name="comments")
    created = models.DateTimeField()
    permalink = models.URLField()
    score = models.IntegerField()

    def __str__(self):
        return "{}: {}/10  ('{}')".format(self.author, self.rating, self.body)
