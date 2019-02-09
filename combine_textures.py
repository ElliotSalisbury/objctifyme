import os
import traceback

import cv2
import numpy as np
from django.db.models import Count, F, Q

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ratemescraper.settings")
import django

django.setup()

from django.conf import settings
from rateme.models import Submission, SubmissionImage

TEXTURE_SIZE = 256
MEDIA_ROOT = settings.MEDIA_ROOT
EXPECTED_AREA = TEXTURE_SIZE * TEXTURE_SIZE

if __name__ == '__main__':
    processed_images = SubmissionImage.objects.all().annotate(face_processing_count=Count("face_processings")) \
        .filter(face_processing_count__gte=F("face_count")).values_list('id', flat=True)
    submissions = Submission.objects.filter(has_images=True) \
        .annotate(image_count=Count('images')) \
        .annotate(processed_image_count=Count("images", filter=Q(images__id__in=processed_images))) \
        .filter(image_count=F("processed_image_count"))

    submissions_len = len(submissions)
    for i, submission in enumerate(submissions):
        try:
            submission_dir = os.path.join(MEDIA_ROOT, submission.id)

            combined_texture_path = os.path.join(submission_dir, "{}_combinedtexture.jpg".format(submission.id))
            normalized_weight_path = os.path.join(submission_dir, "{}_weight.npy".format(submission.id))

            print("{}/{}  {}".format(i, submissions_len, combined_texture_path))
            if os.path.exists(normalized_weight_path):
                print("\tskipping")
                continue

            images = submission.images.all()

            combined_texture = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.float)
            total_weight = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 1), dtype=np.float) + 0.00001
            processing_count = 0
            for image in images:
                face_processings = image.face_processings.all()
                for processing in face_processings:
                    relative_im_path = processing.texture.name
                    im_path = os.path.join(MEDIA_ROOT, relative_im_path)
                    texture = cv2.imread(im_path)

                    relative_mask_path = processing.texture.name.replace("texture.jpg", "texture_mask.jpg")
                    mask_path = os.path.join(MEDIA_ROOT, relative_mask_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    tex_area = texture.shape[0] * texture.shape[1]
                    area_weight = min(1.0, tex_area / EXPECTED_AREA)

                    texture = cv2.resize(texture, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_NEAREST)
                    mask = np.expand_dims(mask, 2)
                    mask = mask / 255.0

                    yscale = 1 - abs(processing.yaw * 2)
                    pscale = 1  # 1-abs(processing.pitch*2)
                    weight = max(yscale * pscale, 0) * area_weight
                    weighted_mask = mask * weight
                    total_weight += weighted_mask

                    combined_texture += texture * weighted_mask
                    processing_count += 1

            combined_texture /= total_weight
            combined_texture = combined_texture.astype(np.uint8)

            cv2.imwrite(combined_texture_path, combined_texture)

            normalized_weight = total_weight / processing_count
            np.save(normalized_weight_path, normalized_weight)
        except Exception as e:
            traceback.print_exc()
            continue
