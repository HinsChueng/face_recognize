# coding=utf-8
# 人物照片添加贴纸
import os
import cv2
from PIL import Image
import numpy as np

# 分类器存放路径
base_path = 'F:/python38/Lib/site-packages/opencv_python-4.2.0.34.dist-info/data'
# 人脸识别分类器名称
face_xml = os.path.join(base_path, 'haarcascade_frontalface_alt.xml')


class FacePrc:
    def __init__(self, save_url, model_url=None, photo_rate=1.5):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.photo_rate = photo_rate
        self.save_url = save_url
        self.model_url = model_url

    def get_face(self, img):
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
            img = self.christmas(img, x, y, w, h)
        return img

    def christmas(self, img, x, y, w, h):
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 你的贴纸地址
        mark = Image.open(self.model_url)
        mark = mark.resize((int(w * PHOTO_RATE), int(h * PHOTO_RATE)))
        # 新生成一张图片，配置色彩模式，图片大小和颜色
        layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
        # 将读取到的图片复制到新建图片中，传入位置为矩形左下角的坐标（文档错了！！不是左上角，哭辽）
        layer.paste(mark, (int(x - (PHOTO_RATE - 1) * w / 2), int(y - (PHOTO_RATE - 1) * h / 2)))
        out = Image.composite(layer, im, layer)
        img = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
        return img

    def write_video(self):
        video_writer = cv2.VideoWriter(self.save_url, cv2.VideoWriter_fourcc(*'MJPG'), 30, (1000, 750))
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                # 从新定义图片大小
                img = cv2.resize(frame, (1000, 750))
                # 实时识别
                img = self.get_face(img)
                # 视频显示
                cv2.imshow('frame', img)
                # 保存视频
                video_writer.write(img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("退出视频")
                    break
            else:
                break
        video_writer.release()

    def write_photo(self):
        if not self.model_url:
            raise Exception('预选模型图片路径无效')

        print('键盘输入：%s 进行拍照' % TAKE_PHOTO_KEY)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                # 重新定义摄像区域大小
                frame = cv2.resize(frame, (1440, 1080))
                # 实时识别
                img = self.get_face(frame)
                # 视频显示
                cv2.imshow('frame', img)
                if cv2.waitKey(10) & 0xFF == ord(TAKE_PHOTO_KEY):
                    # 保存图片
                    cv2.imwrite(self.save_url, img)
                    print('图片保存在%s成功!' % self.save_url)
                    break
            else:
                break

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 贴纸保存目录
    PASTER_DIR = 'model_selections'
    # 自定义拍照按键
    TAKE_PHOTO_KEY = 'q'
    # 图片和脸轮廓矩形的比例
    PHOTO_RATE = 2

    # 预选贴纸存放路径， ./ 表示当前目录平级的位置
    paster_list = os.listdir(PASTER_DIR)
    for i in range(len(paster_list)):
        print('%s -> %s' % (i, paster_list[i]))
    num = input('选择贴纸序号:')
    try:
        num = int(num)
    except Exception as e:
        print('index type error: should be `int`')
        exit()
    MODEL_URL = os.path.join(PASTER_DIR, paster_list[int(num) - 1])

    # 图片保存路径
    SAVE_URL = 'E:/ruk6.jpg'
    # MODEL_URL = './model_selections/6.png'
    face = FacePrc(
        save_url=SAVE_URL,
        photo_rate=PHOTO_RATE,
        model_url=MODEL_URL
    )
    face.write_photo()
    face.close()
