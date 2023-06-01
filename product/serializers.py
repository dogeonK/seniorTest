from django.utils.baseconv import base64
from rest_framework import serializers
from .models import Product, Style, Emoji


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'


import base64
from django.core.files import File


class StyleSerializer(serializers.ModelSerializer):
    #img = serializers.ImageField(use_url=True)

    class Meta:
        model = Style
        fields = ('request_id', 'tag_name', "img_url")

class EmojiSerializer(serializers.ModelSerializer):
    #emoji = serializers.ImageField(use_url=True)

    class Meta:
        model = Emoji
        fields = ('request_id', 'tag_name', "emoji_tag", "emoji_url")
