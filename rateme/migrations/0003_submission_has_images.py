# Generated by Django 2.1.4 on 2018-12-11 10:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rateme', '0002_auto_20181210_1622'),
    ]

    operations = [
        migrations.AddField(
            model_name='submission',
            name='has_images',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
    ]