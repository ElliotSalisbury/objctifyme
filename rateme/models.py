from django.db import models

class User(models.Model):
    id = models.CharField(max_length=256, primary_key=True)

    def __str__(self):
        return self.id

class Submission(models.Model):
    id = models.CharField(max_length=256, primary_key=True)
    title = models.CharField(max_length=256)
    has_images = models.BooleanField()
    age = models.IntegerField()
    gender = models.CharField(max_length=1)
    author = models.ForeignKey("User", on_delete=models.CASCADE, related_name="submissions")
    created = models.DateTimeField()
    permalink = models.URLField()
    score = models.IntegerField()
    upvote_ratio = models.FloatField()

    def __str__(self):
        return self.title

    @property
    def hotness(self):
        score = 0
        comments = self.usable_comments
        size = comments.count()
        if size:
            for comment in comments:
                score += comment.rating
            return score / comments.count()
        return 0

    @property
    def usable_comments(self):
        return self.comments.filter(rating__isnull=False)

class SubmissionImage(models.Model):
    submission = models.ForeignKey("Submission", on_delete=models.CASCADE, related_name="images")
    image = models.ImageField()


class Comment(models.Model):
    id = models.CharField(max_length=256, primary_key=True)
    submission = models.ForeignKey("Submission", on_delete=models.CASCADE, related_name="comments")
    body = models.CharField(max_length=4096)
    rating = models.IntegerField(null=True)
    decimal = models.FloatField(null=True)
    author = models.ForeignKey("User", on_delete=models.CASCADE, related_name="comments")
    created = models.DateTimeField()
    permalink = models.URLField()
    score =  models.IntegerField()

    def __str__(self):
        return "{}: {}/10  ('{}')".format(self.author, self.rating, self.body)