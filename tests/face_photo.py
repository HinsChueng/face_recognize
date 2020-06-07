# coding=utf-8
# 人脸识别+模型覆盖
import os
import cv2
from PIL import Image
import numpy as np

# 分类器存放路径
base_path = 'F:/python38/Lib/site-packages/opencv_python-4.2.0.34.dist-info/data'
# 人脸识别分类器名称
face_xml = os.path.join(base_path, 'haarcascade_frontalface_alt.xml')
# 人眼识别分类器名称
eye_xml = os.path.join(base_path, 'haarcascade_eye_tree_eyeglasses.xml')
# 人脸识别分类器名称2
face_xml_default = os.path.join(base_path, 'haarcascade_frontalface_default.xml')


def get_face(img):
    # 人脸识别数据
    face_cascade = cv2.CascadeClassifier(face_xml)
    # 人眼识别数据
    # eye_cascade = cv2.CascadeClassifier(eye_xml)
    # 二值化,变为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取人脸识别数据
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    for (x, y, w, h) in faces:
        # (x,y)为脸部矩形左下角的坐标，w为矩形的宽，h为矩形的长
        # 根据人脸识别数据添加头像
        img = christmas(img, x, y, w, h)
    return img


def christmas(img, x, y, w, h):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 你的贴纸地址
    mark = Image.open("../model_selections/1.png")
    height = int(h * PHOTO_RATE)
    weight = int(w * PHOTO_RATE)
    mark = mark.resize((weight, height))
    # 新生成一张图片，配置色彩模式，图片大小和颜色
    layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    # 将读取到的图片复制到新建图片中，传入位置为矩形左下角的坐标（文档错了！！不是右上角，哭辽）
    layer.paste(mark, (int(x - (PHOTO_RATE - 1) * w / 2), int(y - (PHOTO_RATE - 1) * h / 2)))
    out = Image.composite(layer, im, layer)
    img = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    return img


def write_video():
    cap = cv2.VideoCapture(0)
    video_writer = cv2.VideoWriter('testwrite.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1000, 750))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            # 从新定义图片大小
            img = cv2.resize(frame, (1000, 750))
            # 实时识别
            img = get_face(img)
            # 视频显示
            cv2.imshow('frame', img)
            # 保存视频
            video_writer.write(img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("退出视频")
                break
        else:
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def write_photo(url):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            # 从新定义图片大小
            img = cv2.resize(frame, (1000, 750))
            # 实时识别
            img = get_face(img)
            # 视频显示
            cv2.imshow('frame', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                # 保存视频
                cv2.imwrite(url, img)
                print('图片保存在%s成功!' % url)
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 图片和脸轮廓矩形的比例
    PHOTO_RATE = 1.5
    # write_video()
    photo_url = 'E:/ruk.jpg'
    write_photo(photo_url)
