# Generated by Django 4.1.7 on 2023-05-31 15:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("product", "0007_alter_emoji_set_num"),
    ]

    operations = [
        migrations.AlterField(
            model_name="emoji", name="set_num", field=models.IntegerField(),
        ),
    ]
