import cv2
import numpy as np
import base64
import math
# import tensorflow as tf
# net = cv2.dnn.readNet('../model/yolov4-tiny.cfg', '../model/yolov4-tiny_3000.weights')
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# with tf.device('/cpu:0'):
#    model = tf.keras.models.load_model("../model/cnn_bienso.h5")


def base64_to_img(base64_data):
    im_bytes = base64.b64decode(base64_data)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img


def img_to_base64(img):
    string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    return string


def cat_bien_so(base64, net):
    img = base64_to_img(base64)
    font = cv2.FONT_HERSHEY_PLAIN
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i]*100, 2))
            bien_so = img[round(y):round(y+h), round(x):round(x+w)]

        return [bien_so, confidence]
    return None


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, np.abs(angle), 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def compute_skew(src_img):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 7)
    edges = cv2.Canny(img,  threshold1=30,  threshold2=100,
                      apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30,
                            minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0
    nlines = lines.size

    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:  # excluding extreme rotations
            angle += ang
            cnt += 1
    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi


def xoayAnh(src_img):
    return rotate_image(src_img, compute_skew(src_img))


def lay_ki_tu(img, typeSelect):
    try:

        img = xoayAnh(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize bien so 400x450
        img = cv2.resize(img, (450, 400))
        if typeSelect == 'daytime':
            otsu = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)[1]
        if typeSelect == 'evening':
            otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
        k = np.ones((5, 5), dtype='uint8')
        otsu = cv2.dilate(otsu, k, iterations=1)
        otsu = cv2.erode(otsu, k, iterations=1)
        # lay tat ca cac doi tuong co trong bien so
        doituong = []
        contours, _ = cv2.findContours(
            otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            doituong.append([x, y, w, h])

        # lọc lấy các số bỏ đối tượng không mong muốn
        doituong_loc = []
        for x, y, w, h in doituong:
            tile_h = h/otsu.shape[0]
            tile_w = w/otsu.shape[1]
            if 0.25 < tile_h < 0.4 and 0.042 < tile_w < 0.18:
                doituong_loc.append([x, y, w, h])
        # tính giá trị trung bình trục y và chia biển số thành phần trên và phần dưới

        mean_y = np.mean(np.array(doituong_loc)[:, 1])

        top = []
        bot = []
        for i in doituong_loc:
            if i[1] < mean_y:
                top.append(i)
            else:
                bot.append(i)
        # Sắp xếp các số theo thứ tự theo trục x
        top = sorted(top, key=lambda x: x[0])
        bot = sorted(bot, key=lambda x: x[0])

        # resize về 32x32 để predict
        img_bot = []
        img_top = []
        for x, y, w, h in top:
            im = otsu[y:y+h, x:x+w]
            im = 255-im

            # them border 4 huong
            im = cv2.copyMakeBorder(im, 1, 1, 2, 2, cv2.BORDER_CONSTANT)
            # resize 28x28
            im = cv2.resize(im, (64, 64))
            im = 255-im
            img_top.append(im)

        for x, y, w, h in bot:
            im = otsu[y:y+h, x:x+w]
            im = 255-im

            # them border 4 huong
            im = cv2.copyMakeBorder(im, 1, 1, 2, 2, cv2.BORDER_CONSTANT)
            # resize 28x28
            im = cv2.resize(im, (64, 64))
            im = 255-im
            img_bot.append(im)

    except:
        return None
    return [img_top, img_bot]


# def predict_bienso(top, bot, model):
#     labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
#               20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}
#     top = np.array(top)
#     top = top.reshape(-1, 64, 64, 1)
#     bot = np.array(bot)
#     bot = bot.reshape(-1, 64, 64, 1)
#     pre_top = model.predict(top)
#     pre_bot = model.predict(bot)
#     rs_top = [labels[np.argmax(i)] for i in pre_top]
#     rs_bot = [labels[np.argmax(i)] for i in pre_bot]
#     rs = rs_top + ["-"] + rs_bot
#     final = ""
#     final = final.join(rs)
#     return final

def predict_bienso(top, bot, model_chuso, model_chucai):
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
              15: 'F', 16: 'G', 17: 'H', 18: 'K', 19: 'L', 20: 'M', 21: 'N', 22: 'P', 23: 'S', 24: 'T', 25: 'U', 26: 'V', 27: 'X', 28: 'Y', 29: 'Z'}

    rs = ""
    top = np.array(top).reshape(-1, 64, 64, 1)
    bot = np.array(bot).reshape(-1, 64, 64, 1)
    if len(top) == 4 or len(top) == 3:
        rs_top_dau = model_chuso.predict(top[:-2])
        rs_top_duoi = model_chucai.predict(top[-2:])
        rs_top_dau = np.argmax(rs_top_dau, axis=1)
        rs_top_duoi = np.argmax(rs_top_duoi, axis=1)
        rs_top = np.concatenate((rs_top_dau, rs_top_duoi))
        rs += "".join(labels[i] for i in rs_top)
        rs += "-"
    else:
        rs_top = model_chucai.predict(top)
        rs_top = np.argmax(rs_top, axis=1)
        rs += "".join(labels[i] for i in rs_top)
        rs += "-"

    rs_bot = model_chuso.predict(bot)
    rs_bot = np.argmax(rs_bot, axis=1)
    rs += "".join(labels[i] for i in rs_bot)
    return rs
