import os
import numpy as np



def get_user_mean(user):
    global user_means

    if user.id not in user_means:
        ratings = []
        for comment in user.comments.filter(rating__isnull=False):
            ratings.append(comment.rating)

        if len(ratings) < User.RATINGS_COUNT_THRESH:
            mean, std = User.ALL_MEAN, User.ALL_STD
        else:
            mean, std = np.mean(ratings), np.std(ratings)
            std = max(std, 0.0001)

        user_means[user.id] = mean, std

    return user_means[user.id]

def calculate_submission_ratings():
    submissions = Submission.objects.filter(has_images=True, calculated_rating__isnull=True) \
        .annotate(usable_comments_count=Count("comments", filter=Q(comments__rating__isnull=False))) \
        .filter(usable_comments_count__gte=10)  # prefetch_related("comments")
    submissions_len = len(submissions)
    for i, submission in enumerate(submissions):
        usable_comments = submission.usable_comments
        if len(usable_comments) > 0:
            actual_ratings = 0
            total_weighting = 0
            for comment in usable_comments:
                rating = comment.rating
                # user_id = comment.author_id
                user_mean, user_std = get_user_mean(comment.author)

                actual_rating = (rating - user_mean) / user_std
                actual_rating = (actual_rating * User.NORM_STD) + User.NORM_MEAN


                weight = user_std**2
                weighted_rating = weight * actual_rating

                actual_ratings += weighted_rating
                total_weighting += weight

            actual_rating = actual_ratings / total_weighting
        else:
            actual_rating = None

        submission.calculated_rating = actual_rating
        submission.save()

if __name__ == '__main__':
    from django.db.models import Count, Q

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ratemescraper.settings")
    import django

    django.setup()
    from rateme.models import User, Submission

    user_means = {}
    # users = User.objects.all()#prefetch_related("comments")
    # all_users = User.objects.all()
    # all_users_len = len(all_users)
    # all_ratings = []
    # for i, user in enumerate(all_users):
    #     print("user {}/{}".format(i, all_users_len))
    #     ratings = []
    #     for comment in user.comments.filter(rating__isnull=False):
    #         ratings.append(comment.rating)
    #
    #     if len(ratings) > RATINGS_COUNT_THRESH:
    #         mean, std = np.mean(ratings), np.std(ratings)
    #         std = max(std, 0.0001)
    #         user_means[user.id] = mean, std
    #     all_ratings.extend(ratings)
    # ALL_MEAN, ALL_STD = np.mean(all_ratings), np.std(all_ratings)
    user_means["deleted"] = User.ALL_MEAN, User.ALL_STD

    calculate_submission_ratings()