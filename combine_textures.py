import os

import cv2
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ratemescraper.settings")
import django

django.setup()

from django.conf import settings
from rateme.models import Submission

TEXTURE_SIZE = 1024
MEDIA_ROOT = settings.MEDIA_ROOT

submissions = Submission.objects.filter(has_images=True, images__processing__isnull=True)
for submission in submissions:
    images = submission.images.all()
    processings = [image.processing for image in images]

    combined_texture = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.float)
    total_weight = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.float) + 0.00001
    for processing in processings:
        relative_im_path = processing.image.image.name
        im_path = os.path.join(MEDIA_ROOT, relative_im_path)
        image_orig = cv2.imread(im_path)

        relative_im_path = processing.texture.name
        im_path = os.path.join(MEDIA_ROOT, relative_im_path)
        texture = cv2.imread(im_path)

        mask = np.zeros_like(texture)
        mask[np.where(
            np.logical_and(np.logical_and(texture[:, :, 0] > 0, texture[:, :, 1] > 0), texture[:, :, 2] > 0))] = (
        1, 1, 1)

        texture = cv2.resize(texture, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_NEAREST)

        yscale = 1 - abs(processing.yaw * 2)
        pscale = 1  # 1-abs(processing.pitch*2)
        weight = max(yscale * pscale, 0)
        weighted_mask = mask * weight
        total_weight += weighted_mask

        combined_texture += texture * weighted_mask

    combined_texture /= total_weight
    combined_texture = combined_texture.astype(np.uint8)

    combined_texture_path = os.path.join(os.path.dirname(im_path), "{}_combinedtexture.jpg".format(submission.id))
    cv2.imwrite(combined_texture_path, combined_texture)
