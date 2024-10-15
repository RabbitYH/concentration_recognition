import cv2
import dlib
import numpy as np

# 初始化 Dlib 的面部检测器和面部关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

# 定义一个函数来计算面部表情
def detect_face_expression(landmarks):
    # 这里可以定义一些逻辑来判断面部表情
    # 比如，根据眼睛和嘴巴的相对位置来判断
    # 这里只是简单示例
    left_eye = landmarks[36:42]  # 左眼的关键点
    right_eye = landmarks[42:48]  # 右眼的关键点
    mouth = landmarks[48:60]  # 嘴巴的关键点

    # 计算眼睛的高度和嘴巴的高度
    eye_height = np.mean(left_eye[:, 1]) + np.mean(right_eye[:, 1])
    mouth_height = np.mean(mouth[:, 1])
    # print("left_eye[:, 1]",left_eye[:, 1],"right_eye[:, 1]",right_eye[:, 1])
    # print("np.mean(left_eye[:, 1])",np.mean(left_eye[:, 1]),"np.mean(right_eye[:, 1])",np.mean(right_eye[:, 1]))
    # print("eye_height:",eye_height,"mouth_height:",mouth_height)
    # print("mouth[:, 1]",mouth[:, 1])
    # print("np.mean(mouth[:, 1])",np.mean(mouth[:, 1]))
    # print("mouth_height", mouth_height)
    # 判断表情（这里只是一个简单的例子）
    if mouth_height < eye_height:
        return "Surprised"
    else:
        return "Neutral"

# 加载图像
img = cv2.imread(r"E:\Python_workspace\known_faces\d.jpg")

# 将图像转换为灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = detector(gray)

for face in faces:
    # 获取面部特征点
    shape = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # 绘制面部关键点
    for (x, y) in landmarks:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # 识别面部表情
    expression = detect_face_expression(landmarks)
    print(f"Detected expression: {expression}")

    # 绘制人脸矩形
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 显示结果
cv2.imshow("Face Expression Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
