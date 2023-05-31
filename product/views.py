from PIL import Image
from django.http import FileResponse, HttpResponse, HttpResponseNotFound
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import *
from .serializers import *
import base64
from io import BytesIO


def check(request):
    return HttpResponse("hihi")


# class ProductListAPI(APIView):
#     def get(self, request):
#         queryset = Product.objects.all()
#         print(queryset)
#         serializer = ProductSerializer(queryset, many=True)
#         return Response(serializer.data)
#

# 스프링에서 요청 -> requestid, {img_url, tag_name} 받아 -> AI 돌려
# DB에 (tag_name1, img_url, img, requestid), (tag_name2, img_url, img, requestid)... 저장
class PictureAPI(APIView):
    def get(self, request, rq_id):
        queryset = Style.objects.filter(request_id=rq_id)
        serializer = StyleSerializer(queryset, many=True)
        return Response(serializer.data)


# 스프링에서 요청 -> requestid, tag_name, emoji_url, emoji_tag 줘
class EmojiAPI(APIView):
    def get(self, request, rq_id):
        queryset = Emoji.objects.filter(request_id=rq_id)
        serializer = EmojiSerializer(queryset, many=True)
        return Response(serializer.data)


# def image_grid(imgs, rows, cols):
#     assert len(imgs) == rows * cols
#
#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols * w, rows * h))
#     grid_w, grid_h = grid.size
#
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i % cols * w, i // cols * h))
#     return grid

def test_reqeust(request):
    return HttpResponse("success!");


def stable(request, rq_id, paint):
    # #text2img
    # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    # prompt = "happy man"
    #
    # image = pipe(prompt).images[0]
    # image.save(f"a_r_h.png")

    # #img2img
    # device = "cuda"
    # pipei2i = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    #
    # url = "https://search.pstatic.net/common/?src=http%3A%2F%2Fimage.nmv.naver.net%2Fblog_2023_03_16_293%2Fc80c6734-c3bb-11ed-ac0f-505dac8c37f3_01.jpg&type=sc960_832"
    # #url = "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMTA0MDFfOTIg%2FMDAxNjE3MjgzMTUwODg3.psjeJ22C3hugmGATUiHJxeiewFzZbrFrP7qjpGY8vXYg.bJTZVLOhwiG6MYR_MZMQkv1QEKdkWVuI8qMCRiV2x_gg.JPEG.hulk35%2F20210323%25A3%25DF115129.jpg&type=sc960_832"
    # response = requests.get(url)
    # init_image = Image.open(BytesIO(response.content)).convert("RGB")
    # init_image.thumbnail((768, 768))
    #
    # #prompt = "Artwork by Pablo Picasso, masterpiece, intricately detailed, cubism, cubist, detailed painting, Pablo Picasso"
    #
    # prompt = "sticker illustration"
    # images = pipei2i(prompt=prompt, image=init_image, strength=0.5, guidance_scale=7.5, num_images_per_prompt=3).images
    #
    # images[0].save("picasso_happy.png")
    # img = open('picasso_happy.png', 'rb')
    # response = FileResponse(img)

    # #depth2img
    #
    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    #
    # url = "https://search.pstatic.net/common/?src=http%3A%2F%2Fimage.nmv.naver.net%2Fblog_2023_03_16_293%2Fc80c6734-c3bb-11ed-ac0f-505dac8c37f3_01.jpg&type=sc960_832"
    # response = requests.get(url)
    # init_image = Image.open(BytesIO(response.content)).convert("RGB")
    # init_image.thumbnail((768, 768))
    #
    # prompt = "illustration"
    # images = pipe(prompt=prompt, image=init_image, strength=0.7).images
    #
    # images[0].save('depth2img.png')
    # img = open('depth2img.png', 'rb')
    # response = FileResponse(img)

    # #ControlNet_Canny
    # image = load_image("https://item.kakaocdn.net/do/27ae59565c69dd75f514e99629c4e4598f324a0b9c48f77dbce3a43bd11ce785")
    # import cv2
    # from PIL import Image
    # import numpy as np
    #
    # image = np.array(image)
    #
    # low_threshold = 100
    # high_threshold = 200
    #
    # image = cv2.Canny(image, low_threshold, high_threshold)
    # image = image[:, :, None]
    # image = np.concatenate([image, image, image], axis=2)
    # canny_image = Image.fromarray(image)
    # canny_image.save("canny.png")
    #
    # #Canny -> stable
    # from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    # import torch
    #
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    # )
    #
    # from diffusers import UniPCMultistepScheduler
    #
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    #
    # pipe.enable_model_cpu_offload()
    #
    # generator = torch.manual_seed(0)
    #
    # prompt = "best quality, extremely detailed"
    #
    # images = pipe(prompt=prompt, num_inference_steps=20, generator=generator, image=canny_image).images
    #
    # images[0].save('stable.png')
    # img = open('stable.png', 'rb')
    # response = FileResponse(img)

    # pix2pix
    import PIL
    import requests
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from enum import Enum

    class Prompt(Enum):
        a = "smile"
        b = "angry"
        c = "add a heart emoji"
        d = "sad"
        e = "yawn"

    class Style_p(Enum):
        gogh = "goth painting style"
        sketch = "sketch"
        cartoon = "cartoon style"

    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    url = "https://search.pstatic.net/sunny/?src=https%3A%2F%2Fmusicimage.xboxlive.com%2Fcatalog%2Fvideo.contributor.c41c6500-0200-11db-89ca-0019b92a3933%2Fimage%3Flocale%3Den-us%26target%3Dcircle&type=sc960_832"

    def download_image(url):
        image = PIL.Image.open(requests.get(url, stream=True).raw)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    image = download_image(url)

    for s in Style_p:
        if s.name == paint:
            t_name = s.value

    for i in range(1, 4):
        for p in Prompt:
            prompt = str(p.value) + str(t_name)
            images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5,
                          guidance_scale=7).images
            images[0].save("stable_pix2pix.png")
            img = open("stable_pix2pix.png", "rb")

            e_name = p.value
            img = base64.b64encode(img.read())
            url = "localhost:8000/showEmoji/" + rq_id + "/" + t_name + "/" + e_name + "/" + str(i)

            test = Emoji(request_id=rq_id, tag_name=t_name, emoji_tag=e_name, emoji_url=url, emoji=img, set_num=i)
            test.save()

    # prompt = "smile"
    # images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
    # images[0].save("stable_pix2pix.png")
    #
    # img = open("stable_pix2pix.png", 'rb')
    # #response = FileResponse(img)
    #
    # rq_id = "maro"
    # t_name = "test_name"
    # e_name = "smile"
    # img = base64.b64encode(img.read())
    # url = "localhost:8000/showEmoji/" + rq_id + "/" + t_name + "/" + e_name
    #
    # test = Emoji(request_id=rq_id, tag_name=t_name, emoji_tag=e_name, emoji_url=url, emoji=img)
    # test.save()

    return HttpResponse("emoji")


def style(request, rq_id):
    # from gradio_client import Client
    #
    # client = Client("https://aravinds1811-neural-style-transfer.hf.space/")
    # result = client.predict(
    #     "https://search.pstatic.net/common/?src=http%3A%2F%2Fimage.nmv.naver.net%2Fblog_2023_03_16_293%2Fc80c6734-c3bb-11ed-ac0f-505dac8c37f3_01.jpg&type=sc960_832",
    #     # str representing filepath or URL to image in 'Content Image' Image component
    #     "https://search.pstatic.net/common/?src=http%3A%2F%2Fimgnews.naver.net%2Fimage%2F469%2F2022%2F12%2F20%2F0000713735_003_20221220061216428.jpg&type=sc960_832",
    #     # str representing filepath or URL to image in 'Style Image' Image component
    #     api_name="/predict"
    # )
    # return HttpResponse(result)
    import os
    import PIL
    import requests
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from enum import Enum

    class Painting(Enum):
        gogh = "goth painting style"
        sketch = "sketch"
        cartoon = "cartoon style"

    # terminal_command = "neural-style -style_image test_dog.jpg -content_image face.jpg -output_image profile.png -model_file models/vgg19-d01eb7cb.pth -image_size 256 -backend cudnn -cudnn_autotune -optimizer adam -save_iter 0"
    # os.system(terminal_command)
    # img = open("profile.png", 'rb')

    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    url = "https://search.pstatic.net/sunny/?src=https%3A%2F%2Fmusicimage.xboxlive.com%2Fcatalog%2Fvideo.contributor.c41c6500-0200-11db-89ca-0019b92a3933%2Fimage%3Flocale%3Den-us%26target%3Dcircle&type=sc960_832"

    def download_image(url):
        image = PIL.Image.open(requests.get(url, stream=True).raw)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    image = download_image(url)

    for p in Painting:
        prompt = str(p.value)
        images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
        images[0].save("paintingStyle.png")
        img = open("paintingStyle.png", "rb")

        t_name = p.value
        img = base64.b64encode(img.read())
        url = "localhost:8000/showImg/" + rq_id + "/" + t_name

        painting = Style(request_id=rq_id, tag_name=t_name, img_url=url, img=img)
        painting.save()

    # # 이미지 저장 test
    # rq_id = "maro"
    # t_name = "test_name"
    # img = base64.b64encode(img.read())
    # url = "localhost:8000/showImg/" + rq_id + "/" + t_name
    #
    # test = Style(request_id=rq_id, tag_name=t_name, img_url=url, img=img)
    # test.save()

    return HttpResponse("style")


def show_img(request, rq_id, t_name):
    styles = Style.objects.filter(request_id=rq_id, tag_name=t_name).values("img")
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
    emojis = Emoji.objects.filter(request_id=rq_id, tag_name=t_name, emoji_tag=e_name, set_num=int(s_num)).values("emoji")
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

#
# def stylepack(request):
#     # # 입력 이미지와 스타일 이미지 경로
#     # content_image_path = "content.jpg"
#     # style_image_path = "style.jpg"
#     #
#     # # NeuralStyle 객체 생성
#     # neural_style = NeuralStyle()
#     #
#     # # 모델 파라미터 설정
#     # neural_style.set_content_image_path(content_image_path)
#     # neural_style.set_style_image_path(style_image_path)
#     # neural_style.set_output_image_path("output.jpg")
#     # neural_style.set_num_iterations(1000)
#     #
#     # # 스타일 전이 실행
#     # neural_style.run()
#     # --------------------------------------------------------------
#     # # from neuralstyle.methods import run_style_transfer
#     # #
#     # # content_image_path = "face.png"
#     # # style_image_path = "StarryNight.png"
#     # # run_style_transfer("/StarryNight.png", "/face.png", "output/")
#
#     return HttpResponse("A")
# def emojiStyle(request, rq_id):
#     import os
#     import PIL
#     import requests
#     import torch
#     from diffusers import StableDiffusionInstructPix2PixPipeline
#     from enum import Enum
#
#
#     model_id = "timbrooks/instruct-pix2pix"
#     pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
#
#     image = Style.objects.filter(request_id=rq_id, tag_name=t_name).values("img")
#
#     for p in Painting:
#         prompt = p.value
#         images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
#         images[0].save("paintingStyle.png")
#         img = open("paintingStyle.png")
#
#         rq_id = "maro"
#         t_name = "test_name"
#         img = base64.b64encode(img.read())
#         url = "localhost:8000/showImg/" + rq_id + "/" + t_name
#
#         test = Style(request_id=rq_id, tag_name=t_name, img_url=url, img=img)
#         test.save()
#
#     return HttpResponse("style")
