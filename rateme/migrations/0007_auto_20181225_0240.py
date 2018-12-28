# Generated by Django 2.1.4 on 2018-12-25 02:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rateme', '0006_imageprocessing'),
    ]

    operations = [
        migrations.AddField(
            model_name='imageprocessing',
            name='pitch',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='imageprocessing',
            name='roll',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='imageprocessing',
            name='yaw',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]
