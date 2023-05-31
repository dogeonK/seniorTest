from django.db import models


class Product(models.Model):
    name = models.CharField(max_length=70)
    price = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Style(models.Model):
    request_id = models.CharField(max_length=50, null=False)
    tag_name = models.CharField(max_length=50, null=False)
    img_url = models.CharField(max_length=256, null=False)
    img = models.BinaryField(null=False)

    class Meta:
        db_table = 'style'


class Emoji(models.Model):
    request_id = models.CharField(max_length=50, null=False)
    tag_name = models.CharField(max_length=50, null=False)
    emoji_tag = models.CharField(max_length=50, null=False)
    emoji_url = models.CharField(max_length=256, null=False)
    emoji = models.BinaryField(null=False)
    set_num = models.IntegerField(null=False)

    class Meta:
        db_table = "emoji"
