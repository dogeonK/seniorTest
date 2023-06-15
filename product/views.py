from PIL import Image, ImageOps, ImageChops
from django.http import HttpResponse, HttpResponseNotFound, FileResponse, HttpResponseRedirect, JsonResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import *
import base64
from io import BytesIO
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from enum import Enum
from rembg import remove
from product.frame_interpolation.eval import interpolator, util
import mediapy
from huggingface_hub import snapshot_download
from image_tools.sizes import resize_and_crop
from moviepy.video.io.VideoFileClip import VideoFileClip
from django.shortcuts import redirect
import threading


def check(request):
    return HttpResponse("hihi")


# 스프링에서 요청 -> requestid, {img_url, tag_name} 받아 -> AI 돌려
# DB에 (tag_name1, img_url, img, requestid), (tag_name2, img_url, img, requestid)... 저장
class PictureAPI(APIView):
    def get(self, request, rq_id):
        queryset = Style.objects.filter(requestId=rq_id)
        serializer = StyleSerializer(queryset, many=True)

        data = serializer.data
        post_url = "http://3.39.22.13:8080/tag/response"
        response = requests.post(post_url, json=data)

        return Response(data)


# 스프링에서 요청 -> requestid, tag_name, emoji_url, emoji_tag 줘
class EmojiAPI(APIView):
    def get(self, request, rq_id):
        queryset = Emoji.objects.filter(requestId=rq_id)
        serializer = EmojiSerializer(queryset, many=True)
        return Response(serializer.data)


def test_reqeust(request):
    return HttpResponse("success!");


def stable(request, rq_id, img_url, paint):
    if not rq_id:
        return "fail"

    if Emoji.objects.filter(requestId=rq_id).exists():
        return HttpResponse("exist")

    redirect_url = "/stable_model/{}/{}/{}".format(rq_id, img_url, paint)
    response = redirect(redirect_url)
    response.status_code = 200

    return response


def stable_model(request, rq_id, img_url, paint):
    class Prompt(Enum):
        a = "smile"
        # b = "angry"
        # c = "add a heart emoji"
        # d = "sad"
        # e = "yawn"

    class Style_p(Enum):
        gogh = "gogh painting style"
        sketch = "sketch"
        cartoon = "cartoon style"

    # model_id = "timbrooks/instruct-pix2pix"
    # pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    #
    # url = "https://search.pstatic.net/sunny/?src=https%3A%2F%2Fmusicimage.xboxlive.com%2Fcatalog%2Fvideo.contributor.c41c6500-0200-11db-89ca-0019b92a3933%2Fimage%3Flocale%3Den-us%26target%3Dcircle&type=sc960_832"
    #
    # # imgPath = "http://3.39.22.13:8080/imagePath/"
    # # url = imgPath + str(img_url)
    #
    # def download_image(url):
    #     image = Image.open(requests.get(url, stream=True).raw)
    #     image = ImageOps.exif_transpose(image)
    #     image = image.convert("RGB")
    #     return image
    #
    # image = download_image(url)
    # image.save("original.png")

    # mp4 변환 메서드들
    def load_model(model_name):
        model = interpolator.Interpolator(snapshot_download(repo_id=model_name), None)

        return model

    model_name = "akhaliq/frame-interpolation-film-style"
    models = {model_name: load_model(model_name)}

    ffmpeg_path = util.get_ffmpeg_path()
    mediapy.set_ffmpeg(ffmpeg_path)

    def resize(width, img):
        basewidth = width
        img = Image.open(img)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        return img

    def resize_img(img1, img2):
        img_target_size = Image.open(img1)
        img_to_resize = resize_and_crop(
            img2,
            (img_target_size.size[0], img_target_size.size[1]),
            crop_origin="middle"
        )
        img_to_resize.save('resized_img2.png')

    def predict(frame1, frame2, times_to_interpolate, model_name):
        model = models[model_name]
        frame1 = resize(256, frame1)
        frame2 = resize(256, frame2)

        frame1.save("test1.png")
        frame2.save("test2.png")

        resize_img("test1.png", "test2.png")
        input_frames = ["test1.png", "resized_img2.png"]

        frames = list(
            util.interpolate_recursively_from_files(
                input_frames, times_to_interpolate, model))

        mediapy.write_video("out.mp4", frames, fps=30)

    for s in Style_p:
        if s.name == paint:
            t_name = s.value

    # img = Style.objects.filter(requestId=rq_id, tagName=t_name).values("img")
    # base_string = img.first()['img']
    # image = Image.open(BytesIO(base64.b64decode(base_string)))
    # image = ImageOps.exif_transpose(image)
    # image = image.convert("RGB")

    sfw_prompt = "Generate safe and SFW images without any NSFW content."

    for i in range(1, 2):
        for p in Prompt:
            # prompt = str(p.name) + ", " + str(t_name) + ", " + sfw_prompt
            # images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5,
            #               guidance_scale=7).images
            # images[0].save("stable_pix2pix.png")

            # remove background
            input_path = 'stable_pix2pix.png'
            output_path = 'output.png'

            input = Image.open(input_path)
            output = remove(input)
            output.save(output_path)

            # 이미지 파일 오픈
            wordImg = str(p.value) + ".png"
            background = Image.open("jennie.png").convert("RGBA")
            # foreground = Image.open(wordImg).convert("RGBA")
            foreground = Image.open("smile_rem.png").convert("RGBA")

            # 배경이 투명한 이미지 파일의 사이즈 가져오기
            (img_h, img_w) = foreground.size

            # 합성할 배경 이미지를 위의 파일 사이즈로 resize
            resize_back = background.resize((img_h, img_w))

            # 투명 마스트 생성
            alpha_mask = foreground.split()[3]

            # 이미지 합성
            # resize_back.paste(foreground, (0, 0), foreground)
            merged_image = ImageChops.composite(foreground, resize_back, alpha_mask)

            # resize_back.save("merge.png")
            merged_image.save("merge.png")

            # img = open("merge.png", "rb")

            # mp4 생성 후 -> gif 변경
            predict("original.png", "merge.png", 3, model_name)
            VideoFileClip('out.mp4').write_gif('out.gif')
            gif = open('out.gif', 'rb')

            e_name = p.value
            # img = base64.b64encode(img.read())
            gif = base64.b64encode(gif.read())

            # url = "localhost:8000/showEmoji/" + rq_id + "/" + t_name + "/" + e_name + "/" + str(i)
            # url = "localhost:8000/showEmojiGif/" + rq_id + "/" + t_name + "/" + e_name + "/" + str(i)
            url = "43.201.219.33:8000/showEmojiGif/" + rq_id + "/" + t_name + "/" + e_name + "/" + str(i)

            test = Emoji(requestId=rq_id, tagName=t_name, emojiTag=e_name, emojiUrl=url, emoji=gif, setNum=i)
            test.save()

    return HttpResponse("emoji")


def style_model(request, rq_id, img_url):
    class Painting(Enum):
        gogh = "gogh painting style"
        sketch = "sketch"
        cartoon = "cartoon style"

    # model_id = "timbrooks/instruct-pix2pix"
    # pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    # url = "https://search.pstatic.net/sunny/?src=https%3A%2F%2Fmusicimage.xboxlive.com%2Fcatalog%2Fvideo.contributor.c41c6500-0200-11db-89ca-0019b92a3933%2Fimage%3Flocale%3Den-us%26target%3Dcircle&type=sc960_832"
    print("style_model start")
    print(rq_id)
    print(img_url)
    imgPath = "http://3.39.22.13:8080/imagePath/"
    url = str(imgPath) + str(img_url)
    print(url)

    def download_image(url):
        image = Image.open(requests.get(url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    image = download_image(url)

    for p in Painting:
        # prompt = str(p.value)
        # images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
        # images[0].save("paintingStyle.png")

        # input_path = 'paintingStyle.png'
        # output_path = 'outStyle.png'
        #
        # input = Image.open(input_path)
        # output = remove(input)
        # output.save(output_path)

        img = open("outStyle.png", "rb")

        t_name = p.value
        img = base64.b64encode(img.read())
        # url = "localhost:8000/showImg/" + rq_id + "/" + t_name
        url = "43.201.219.33:8000/showImg/" + rq_id + "/" + t_name

        painting = Style(requestId=rq_id, tagName=t_name, imgUrl=url, img=img)
        painting.save()
        get_url = "http://43.201.219.33:8000/api/picture/{}".format(rq_id)
        response = requests.get(get_url)
        return response


def style(request, rq_id, img_url):
    if not rq_id:
        return "fail"

    exists = Style.objects.filter(requestId=rq_id).exists()

    if exists:
        return HttpResponse("exist")

    style_thread = threading.Thread(target=style_model, args=(request, rq_id, img_url))
    style_thread.start()

    return HttpResponse("success")


def show_img(request, rq_id, t_name):
    styles = Style.objects.filter(requestId=rq_id, tagName=t_name).values("img")
    if styles.exists():
        base_string = styles.first()['img']
        img = Image.open(BytesIO(base64.b64decode(base_string)))

        if img.format == "JPEG":
            c_type = "image/jpeg"
        elif img.format == "PNG":
            c_type = "image/png"
        else:
            return HttpResponseNotFound("Unsupported image foramt")

        response = HttpResponse(content_type=c_type)
        img.save(response, format=img.format)
        return response
    else:
        return HttpResponseNotFound("Image not found")


def show_emoji(request, rq_id, t_name, e_name, s_num):
    emojis = Emoji.objects.filter(requestId=rq_id, tagName=t_name, emojiTag=e_name, setNum=int(s_num)).values(
        "emoji")
    if emojis.exists():
        base_string = emojis.first()['emoji']
        img = Image.open(BytesIO(base64.b64decode(base_string)))

        if img.format == "JPEG":
            c_type = "image/jpeg"
        elif img.format == "PNG":
            c_type = "image/png"
        else:
            return HttpResponseNotFound("Unsupported image foramt")

        response = HttpResponse(content_type=c_type)
        img.save(response, format=img.format)
        return response
    else:
        return HttpResponseNotFound("Image not found")


def show_emoji_gif(request, rq_id, t_name, e_name, s_num):
    emojis = Emoji.objects.filter(requestId=rq_id, tagName=t_name, emojiTag=e_name, setNum=int(s_num)).values(
        "emoji")
    if emojis.exists():
        base_string = emojis.first()['emoji']

        decoded_data = base64.b64decode(base_string)

        response = HttpResponse(decoded_data, content_type='image/gif')
        return response
    else:
        return HttpResponseNotFound("Emoji not found")


def process_painting(rq_id, image, p):
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    prompt = str(p.value)
    images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
    images[0].save("paintingStyle.png")

    input_path = 'paintingStyle.png'
    output_path = 'outStyle.png'

    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)

    img = open("outStyle.png", "rb")

    t_name = p.value
    img = base64.b64encode(img.read())
    url = "localhost:8000/showImg/" + rq_id + "/" + t_name
    # url = "43.201.219.33:8000/showImg/" + rq_id + "/" + t_name

    painting = asynctest(requestId=rq_id, tagName=t_name, tagUrl=url, img=img, setNum=1)
    painting.save()
    # get_url = "http://43.201.219.33:8000/api/picture/{}".format(rq_id)
    # response = requests.get(get_url)
    # return response
    print("save success")


def style_model_async(request, rq_id, img_url):
    from multiprocessing import Process
    class Painting(Enum):
        gogh = "gogh painting style"
        sketch = "sketch"
        cartoon = "cartoon style"

    url = "https://search.pstatic.net/sunny/?src=https%3A%2F%2Fmusicimage.xboxlive.com%2Fcatalog%2Fvideo.contributor.c41c6500-0200-11db-89ca-0019b92a3933%2Fimage%3Flocale%3Den-us%26target%3Dcircle&type=sc960_832"

    def download_image(url):
        image = Image.open(requests.get(url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    image = download_image(url)

    for p in Painting:
        process_thread = threading.Thread(target=process_painting, args=(rq_id, image, p))
        process_thread.start()
        # process_thread = Process(target=process_painting, args=(rq_id, image, p))
        # process_thread.start()
        print("process_thread start")
    return HttpResponse("async")


def style_async(request, rq_id, img_url):
    from multiprocessing import Process
    style_thread = threading.Thread(target=style_model_async, args=(request, rq_id, img_url))
    style_thread.start()
    # process_thread = Process(target=style_model_async, args=(request, rq_id, img_url))
    # process_thread.start()

    return HttpResponse("success")
