# Generated by Django 2.1.4 on 2018-12-28 01:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rateme', '0007_auto_20181225_0240'),
    ]

    operations = [
        migrations.AddField(
            model_name='submissionimage',
            name='face_count',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
    ]
