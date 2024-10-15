import dlib
import cv2

# 加载人脸检测器和特征点检测模型
detector = dlib.get_frontal_face_detector()
predictor_path = './model/shape_predictor_68_face_landmarks.dat'  # 替换为你的模型路径
predictor = dlib.shape_predictor(predictor_path)

# 读取图像
image_path = r"E:\Python_workspace\known_faces\a.jpg"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = detector(gray_image)

# 遍历检测到的人脸
for face in faces:
    # 提取面部特征点
    landmarks = predictor(gray_image, face)
    
    # 在图像上绘制特征点
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 绿色圆点

# 显示结果
cv2.imshow('Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
