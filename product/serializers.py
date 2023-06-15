from django.utils.baseconv import base64
from rest_framework import serializers
from .models import Product, Style, Emoji, asynctest


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
        fields = ('requestId', 'tagName', "imgUrl")

class EmojiSerializer(serializers.ModelSerializer):
    #emoji = serializers.ImageField(use_url=True)

    class Meta:
        model = Emoji
        fields = ('requestId', 'tagName', "emojiTag", "emojiUrl")

class AsyncSerializer(serializers.ModelSerializer):
    #emoji = serializers.ImageField(use_url=True)

    class Meta:
        model = asynctest
        fields = '__all__'