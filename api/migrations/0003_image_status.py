# Generated by Django 3.2.3 on 2021-11-24 06:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_alter_image_time_create'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='status',
            field=models.BooleanField(default=False),
        ),
    ]
