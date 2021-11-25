from typing import final
from django.shortcuts import render
from rest_framework import serializers, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import ImageSerializer
from .models import Image
import cv2
import numpy as np
import tensorflow as tf
from .utils import cat_bien_so, lay_ki_tu, predict_bienso, img_to_base64
import datetime
import uuid
from base64 import b64decode
from django.core.files.base import ContentFile

net = cv2.dnn.readNet('./models/yolov4-tiny.cfg',
                      './models/yolov4-tiny_3000.weights')
model = tf.keras.models.load_model("./models/cnn_bienso.h5")


@api_view(["GET"])
def apiOverview(request):

    api_url = {
        'predict': '/predict/',
        'list': '/list_predict/'
    }

    return Response(api_url)


@api_view(["GET"])
def listImage(request):
    lists = Image.objects.all()
    list_obj = ImageSerializer(lists, many=True, context={'request': request})
    return Response(list_obj.data)


@api_view(["GET"])
def listImageToDay(request):
    today = datetime.date.today()
    lists = Image.objects.filter(time_create__year=today.year,
                                 time_create__month=today.month, time_create__day=today.day)
    list_obj = ImageSerializer(lists, many=True, context={'request': request})
    return Response(list_obj.data)


@api_view(["POST"])
def checkIn(request):
    if request.method == "POST":
        img_base64 = request.data['base64']
        typeSelect = request.data['type']
        print(typeSelect)
        lisencePlate_confidence = cat_bien_so(img_base64, net)
        if lisencePlate_confidence is not None:
            img = lisencePlate_confidence[0]
            # print(img.shape)
            # cv2.imshow("a",img)
            # cv2.imwrite('./media/image/'+str(uuid.uuid4())+".jpg",img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            confidence = lisencePlate_confidence[1]
            if typeSelect == 'daytime':
                top_bot = lay_ki_tu(img, 'daytime')
            if typeSelect == 'evening':
                top_bot = lay_ki_tu(img, 'evening')
            if top_bot is not None:
                top = top_bot[0]
                bot = top_bot[1]
                rs = predict_bienso(top, bot, model)

                # luu anh base64 vao model
                image_data = b64decode(img_to_base64(img))
                image_name = str(uuid.uuid4())+".jpg"
                # tao model
                image = ContentFile(image_data, image_name)
                Image(image=image, confidences=confidence, result=rs,
                      time_create=datetime.datetime.now()).save()

                obj_last = Image.objects.last()
                serializer_obj = ImageSerializer(
                    obj_last, many=False, context={'request': request})

                return Response(serializer_obj.data)
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def checkOut(request):
    if request.method == "POST":
        img_base64 = request.data['base64']
        typeSelect = request.data['type']
        lisencePlate_confidence = cat_bien_so(img_base64, net)
        if lisencePlate_confidence is not None:
            img = lisencePlate_confidence[0]

            confidence = lisencePlate_confidence[1]

            if typeSelect == 'daytime':
                top_bot = lay_ki_tu(img, 'daytime')
            if typeSelect == 'evening':
                top_bot = lay_ki_tu(img, 'evening')
            if top_bot is not None:
                top = top_bot[0]
                bot = top_bot[1]
                rs = predict_bienso(top, bot, model)
                final = {
                    "image": img_to_base64(img),
                    "result": rs,
                    "confidences": confidence
                }
                image = Image.objects.filter(result=rs)

                if len(image) > 0:
                    for i in image:
                        data_dict = {
                            "id": i.id,
                            "image": i.image,
                            "confidences": i.confidences,
                            "result": i.result,
                            "status": False,
                            "time_create": i.time_create
                        }
                        serializer = ImageSerializer(
                            instance=i, data=data_dict)
                        if serializer.is_valid():
                            serializer.save()
                else:
                    return Response(final, status=status.HTTP_404_NOT_FOUND)

                return Response(final)
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(['DELETE'])
def deleteImage(request, pk):
    image = Image.objects.get(id=pk)
    image.delete()
    return Response("Successful delete !")


@api_view(['POST'])
def updateCheckIn(request, pk):
    image = Image.objects.get(id=pk)
    data_dict = {
        "image": image.image,
        "confidences": image.confidences,
        "result": request.data["result"],
        "status": image.status,
        "time_create": image.time_create
    }
    serializer = ImageSerializer(instance=image, data=data_dict)
    if serializer.is_valid():
        serializer.save()
    else:
        print("Error update !!!")
    return Response(serializer.data)


@api_view(['POST'])
def updateCheckOut(request):
    number = request.data['result']
    image = Image.objects.filter(result=number)
    if len(image) > 0:
        for i in image:
            data_dict = {
                "id": i.id,
                "image": i.image,
                "confidences": i.confidences,
                "result": i.result,
                "status": False,
                "time_create": i.time_create
            }
            serializer = ImageSerializer(
                instance=i, data=data_dict)
            if serializer.is_valid():
                serializer.save()
    else:
        return Response(status=status.HTTP_404_NOT_FOUND)

    return Response(status=status.HTTP_200_OK)
