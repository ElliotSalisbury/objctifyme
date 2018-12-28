import datetime
import os
import re
import time
import traceback
import urllib
from urllib.error import HTTPError

import cv2
import dlib
import praw
from imgurpython import ImgurClient

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ratemescraper.settings")
import django

django.setup()

from rateme.models import Submission, Comment, SubmissionImage, User
from django.conf import settings

dstFolder = settings.MEDIA_ROOT

detector = dlib.get_frontal_face_detector()

reAgeGender = re.compile("(\d+)[\s]*([MF])")
reGenderAge = re.compile("([MF])[\s]*(\d+)")
reRatingSlash = re.compile("(\d+)(\.\d+)?\/10")
reImgurAlbum = re.compile("imgur\.com\/a\/(\w+)")
reImgurGallery = re.compile("imgur\.com\/gallery\/(\w+)")

author_blacklist = ['AutoModerator', ]


class TitleNoParse(Exception):
    pass


time_since_last_imgur = datetime.datetime.now()


def getImgUrlsFromAlbum(imgur, albumId, is_gallery=False):
    global time_since_last_imgur

    imgurls = []
    try:
        now = datetime.datetime.now()
        seconds_since = (now - time_since_last_imgur).total_seconds()
        seconds_to_wait = max(8 - seconds_since, 0)
        time.sleep(seconds_to_wait)  # prevent rate limit
        time_since_last_imgur = now

        if is_gallery:
            album = imgur.gallery_item(albumId)
        else:
            album = imgur.get_album(albumId)

        for image in album.images:
            imgurls.append(image['link'])
    except Exception as e:
        print(str(e))
    return imgurls


def image_resize(image, max_width=512, max_height=512, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # check to see if the width is None
    if h > w:
        # calculate the ratio of the height and construct the
        # dimensions
        r = max_height / float(h)
        dim = (int(w * r), max_height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = max_width / float(w)
        dim = (max_width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def check_image_usable(filepath):
    im = cv2.imread(filepath)
    if im is not None:
        im = image_resize(im)
        rects = detector(im, 1)
        return len(rects)
    return 0


def downloadImages(dstPath, imgurls):
    filepaths = []

    for imageurl in imgurls:
        filename = imageurl.split('/')[-1]
        filepath = os.path.join(dstPath, filename)
        urllib.request.urlretrieve(imageurl, filepath)

        faces_count = check_image_usable(filepath)
        if faces_count > 0:
            path_n_count = (filepath[len(dstFolder) + 1:], faces_count)
            filepaths.append(path_n_count)
        else:
            os.remove(filepath)
    return filepaths


def parse_title(title):
    title = re.sub("[\[\{\(\)\}\]\\\/]", "", title)
    title = title.upper()
    result = reAgeGender.search(title)
    if result:
        age = result.group(1)
        gender = result.group(2)
    else:
        result = reGenderAge.search(title)
        if result:
            age = result.group(2)
            gender = result.group(1)
        else:
            raise TitleNoParse("Title cannot be parsed: {}".format(title))

    return int(age), gender


def get_author_name(author):
    author_name = "deleted"
    if author is not None:
        author_name = author.name

    try:
        db_user = User.objects.get(pk=author_name)
    except User.DoesNotExist as e:
        db_user = User(id=author_name)
        db_user.save()

    return db_user


def download_photos(imgur, submission):
    dstPath = os.path.join(dstFolder, submission.id)

    imgurls = None
    if "imgur.com/a/" in submission.url:
        url = submission.url
        albumId = url.split("/")[-1]
        imgurls = getImgUrlsFromAlbum(imgur, albumId)
    elif "imgur.com/gallery/" in submission.url:
        url = submission.url
        albumId = url.split("/")[-1]
        imgurls = getImgUrlsFromAlbum(imgur, albumId, is_gallery=True)
    elif submission.is_self:
        result = reImgurAlbum.search(submission.selftext)
        if result:
            imgurls = getImgUrlsFromAlbum(imgur, result.group(1))
        else:
            result = reImgurGallery.search(submission.selftext)
            if result:
                imgurls = getImgUrlsFromAlbum(imgur, result.group(1), is_gallery=True)
    elif ".jpg" in submission.url or ".png" in submission.url:
        imgurls = [submission.url, ]
    else:
        print("\t" + submission.url)

    filepaths = []
    if imgurls:
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)

        filepaths = downloadImages(dstPath, imgurls)

        # clean up the folder if we did not keep any of the images
        if len(filepaths) == 0:
            os.rmdir(dstPath)

    return dstPath, filepaths


def parse_submission(imgur, submission):
    age, gender = parse_title(submission.title)

    # check if we have this submission in our db
    try:
        db_submission = Submission.objects.get(pk=submission.id)

        print("\t\tSubmission found, retrying to download images")
        # download the submissions photos
        img_dir_path, filepaths = download_photos(imgur, submission)

        print("\t\thas {} usable images".format(len(filepaths)))
        db_submission.has_images = len(filepaths) > 0
        db_submission.score = submission.score
        db_submission.upvote_ratio = submission.upvote_ratio
        db_submission.save()

        # save all the submission images
        for path_n_count in filepaths:
            filepath = path_n_count[0]
            count = path_n_count[1]
            SubmissionImage.objects.update_or_create(submission=db_submission, image=filepath,
                                                     defaults={'face_count': count})

    except Submission.DoesNotExist as e:
        submission_created = datetime.datetime.fromtimestamp(submission.created_utc, tz=datetime.timezone.utc)
        submission_title = submission.title
        submission_author = get_author_name(submission.author)

        # download the submissions photos
        img_dir_path, filepaths = download_photos(imgur, submission)

        db_submission = Submission(id=submission.id,
                                   title=submission_title,
                                   age=age,
                                   gender=gender,
                                   has_images=len(filepaths) > 0,
                                   author=submission_author,
                                   created=submission_created,
                                   permalink=submission.permalink,
                                   score=submission.score,
                                   upvote_ratio=submission.upvote_ratio)
        db_submission.save()

        # save all the submission images
        for path_n_count in filepaths:
            filepath = path_n_count[0]
            count = path_n_count[1]
            image = SubmissionImage(submission=db_submission, image=filepath, face_count=count)
            image.save()

    return db_submission


def parse_comment(db_submission, comment):
    comment_id = comment.id

    try:
        db_comment = Comment.objects.get(pk=comment_id)

        db_comment.score = comment.score
        db_comment.save()
    except Comment.DoesNotExist as e:
        comment_body = comment.body
        comment_created = datetime.datetime.fromtimestamp(comment.created_utc, tz=datetime.timezone.utc)
        comment_author = get_author_name(comment.author)
        if comment_author in author_blacklist:
            return

        result = reRatingSlash.search(comment.body)
        rating = None
        decimal = None

        if result:
            rating = int(result.group(1))

            if int(rating) <= 10:
                decimal = result.group(2)
                if decimal is not None:
                    decimal = float(decimal)
            else:
                rating = None

        db_comment = Comment(id=comment_id,
                             body=comment_body,
                             rating=rating,
                             decimal=decimal,
                             author=comment_author,
                             submission_id=db_submission.id,
                             created=comment_created,
                             permalink=comment.permalink,
                             score=comment.score)
        db_comment.save()
        print("\tadded new comment: {}".format(db_comment))


def scrape():
    reddit = praw.Reddit(user_agent='RateMeScraper',
                         client_id='mi6RmDXyraCd0g', client_secret="sXoQ7BRVaX0uvm_bfvK-N-vcaDk")
    subreddit = reddit.subreddit('rateme')

    imgur = ImgurClient("fc9299b2b6b8315", "e68c8491aa0be70333539be249096e940689c3bb")

    existing_ids = Submission.objects.filter(has_images=True).values_list('id', flat=True)
    to_check_ids = set(Submission.objects.filter(has_images=False).values_list('id', flat=True))
    with open("sub_ids2.txt", "r") as f:
        for submission_id in f.readlines():
            to_check_ids.add(submission_id.strip())

    to_check_ids = to_check_ids - set(existing_ids)
    to_check_len = len(to_check_ids)

    for i, submission_id in enumerate(to_check_ids):
        submission = reddit.submission(submission_id)
        try:
            print("{}/{}: {}".format(i, to_check_len, submission.title))
            db_submission = parse_submission(imgur, submission)

            # skip grabbing the comments if the submission doesnt have any usable images
            if not db_submission.has_images:
                continue

            # parse the comments in that submission
            for top_level_comment in submission.comments:
                db_comment = parse_comment(db_submission, top_level_comment)

        except (TitleNoParse, HTTPError) as e:
            print(e)
            continue
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

    for submission in subreddit.hot(limit=1000):
        try:
            db_submission = parse_submission(imgur, submission)

            # skip grabbing the comments if the submission doesnt have any usable images
            if not db_submission.has_images:
                continue

            # parse the comments in that submission
            for top_level_comment in submission.comments:
                db_comment = parse_comment(db_submission, top_level_comment)

        except (TitleNoParse, HTTPError) as e:
            print(e)
            continue
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

    for comment in subreddit.stream.comments():
        try:
            submission = comment.submission

            db_submission = parse_submission(imgur, submission)

            # skip grabbing the comments if the submission doesnt have any usable images
            if not db_submission.has_images:
                continue

            # parse the comments in that submission
            for top_level_comment in submission.comments:
                db_comment = parse_comment(db_submission, top_level_comment)

        except (TitleNoParse, HTTPError) as e:
            print(e)
            continue
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue


if __name__ == '__main__':
    scrape()
