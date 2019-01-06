import os

import cv2
import numpy as np
from django.db.models import Count, F

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ratemescraper.settings")
import django

django.setup()

from django.conf import settings
from rateme.models import Submission, SubmissionImage

TEXTURE_SIZE = 1024
MEDIA_ROOT = settings.MEDIA_ROOT
EXPECTED_AREA = TEXTURE_SIZE * TEXTURE_SIZE

if __name__ == '__main__':
    processed_images = SubmissionImage.objects.all().annotate(face_processing_count=Count("face_processings")) \
                .filter(face_processing_count__gte=F("face_count")).values('id')
    submissions = Submission.objects.filter(has_images=True, images__in=processed_images)

    for submission in submissions:
        images = submission.images.all()

        combined_texture = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.float)
        total_weight = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.float) + 0.00001
        processing_count = 0
        for image in images:
            face_processings = image.face_processings.all()
            for processing in face_processings:
                relative_im_path = processing.texture.name
                im_path = os.path.join(MEDIA_ROOT, relative_im_path)
                texture = cv2.imread(im_path)

                tex_area = texture.shape[0] * texture.shape[1]
                area_weight = min(1.0, tex_area / EXPECTED_AREA)

                mask = np.zeros_like(texture)
                mask[np.where(
                    np.logical_and(np.logical_and(texture[:, :, 0] > 0, texture[:, :, 1] > 0), texture[:, :, 2] > 0))] = (
                1, 1, 1)

                texture = cv2.resize(texture, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_NEAREST)

                yscale = 1 - abs(processing.yaw * 2)
                pscale = 1  # 1-abs(processing.pitch*2)
                weight = max(yscale * pscale, 0) * area_weight
                weighted_mask = mask * weight
                total_weight += weighted_mask

                combined_texture += texture * weighted_mask
                processing_count += 1

        combined_texture /= total_weight
        combined_texture = combined_texture.astype(np.uint8)

        combined_texture_path = os.path.join(os.path.dirname(im_path), "{}_combinedtexture.jpg".format(submission.id))
        cv2.imwrite(combined_texture_path, combined_texture)

        normalized_weight = total_weight / processing_count
        normalized_weight_path = os.path.join(os.path.dirname(im_path), "{}_weight.npz".format(submission.id))
        np.save(normalized_weight_path, normalized_weight)

