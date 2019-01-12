from django.db import models
import numpy as np

class User(models.Model):
    id = models.CharField(max_length=256, primary_key=True)

    NORM_MEAN = 5
    NORM_STD = 3
    RATINGS_COUNT_THRESH = 3
    ALL_MEAN = 6.929547654727015
    ALL_STD = 1.675662345628553

    def __str__(self):
        return self.id

    def get_mean_std(self):
        ratings = []
        for comment in self.comments.filter(rating__isnull=False):
            ratings.append(comment.rating)

        if len(ratings) < self.RATINGS_COUNT_THRESH:
            mean, std = self.ALL_MEAN, self.ALL_STD
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
        usable_comments = self.usable_comments
        if len(usable_comments) > 0:
            actual_ratings = 0
            total_weighting = 0
            for comment in usable_comments:
                rating = comment.rating
                user_mean, user_std = comment.author.get_mean_std()

                actual_rating = (rating - user_mean) / user_std
                actual_rating = (actual_rating * User.NORM_STD) + User.NORM_MEAN

                weight = user_std**2
                weighted_rating = weight * actual_rating

                actual_ratings += weighted_rating
                total_weighting += weight

            actual_rating = actual_ratings / total_weighting
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
    texture_coefficients = models.CharField(max_length=4096, null=True)
    pitch = models.FloatField()
    yaw = models.FloatField()
    roll = models.FloatField()


class Comment(models.Model):
    id = models.CharField(max_length=256, primary_key=True)
    submission = models.ForeignKey("Submission", on_delete=models.CASCADE, related_name="comments")
    body = models.TextField()
    rating = models.IntegerField(null=True)
    decimal = models.FloatField(null=True)
    author = models.ForeignKey("User", on_delete=models.CASCADE, related_name="comments")
    created = models.DateTimeField()
    permalink = models.URLField()
    score = models.IntegerField()

    def __str__(self):
        return "{}: {}/10  ('{}')".format(self.author, self.rating, self.body)
