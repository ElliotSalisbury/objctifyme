# Generated by Django 2.1.4 on 2018-12-11 11:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rateme', '0003_submission_has_images'),
    ]

    operations = [
        migrations.AlterField(
            model_name='comment',
            name='rating',
            field=models.IntegerField(null=True),
        ),
    ]
