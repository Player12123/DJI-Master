import cv2
import mediapipe as mp
import math
import time
import threading
from robomaster import robot
from robomaster import camera

# 初始化MediaPipe姿态检测模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 摄像头参数（这些值需要根据实际摄像头进行校准）
# focal_length和pixel_size的值需要根据你的摄像头进行设定
focal_length = 800  # 焦距，单位通常是像素
pixel_size = 1024  # 像素大小，单位通常是米/像素
PTime=0
# 假设的人体肩宽，单位：米
assumed_body_width = 0.5

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        if self.integral > 1:
            self.integral = 1
        elif self.integral < -1:
            self.integral = -1

        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


# 函数：根据检测到的关键点计算人体在图像中的像素大小
def calculate_pixel_size(landmarks):
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    if left_shoulder.visibility and right_shoulder.visibility:
        pixel_distance = math.sqrt((right_shoulder.x - left_shoulder.x) ** 2 + (right_shoulder.y - left_shoulder.y) ** 2)
        return pixel_distance
    else:
        return None

# 函数：根据像素大小和摄像头参数估计距离
def estimate_distance(focal_length, pixel_size, pixel_distance, assumed_body_width):
    estimated_distance = (focal_length * assumed_body_width) / (pixel_size * pixel_distance)
    return estimated_distance

# 函数：根据目标距离调整小车移动
def adjust_chassis_movement(distance):
    # PID 控制器的参数
    Kp = -0.5  # 比例系数
    Ki = 0  # 积分系数
    Kd = 0  # 微分系数
    setpoint = 5  # 目标距离
    # 初始化PID控制器的变量
    previous_error = 0
    integral = 0
    # 假设 distance 是当前小车与目标之间的距离

    # 计算误差
    error = setpoint - distance
    # 更新积分项
    integral += error
    # 计算微分项（这里使用误差的变化）
    derivative = error - previous_error
        # 更新之前的误差值，用于下一次迭代
    previous_error = error

    # 使用PID公式计算输出
    output = Kp * error + Ki * integral + Kd * derivative

    # 限制输出速度在合理范围内
    max_speed = 8
    min_speed = -8
    if output > max_speed:
        chassis_speed = max_speed
    elif output < min_speed:
        chassis_speed = min_speed
    else:
        chassis_speed = output
    # 使用计算出的速度来驱动小车
    # 假设 ep_chassis 是一个可以接收速度指令的对象
    ep_chassis.drive_speed(x=chassis_speed, y=0, z=0)


def adjust_position_pid(center_x, center_y, width, height):
    Kp_position = -800  # 比例系数
    Ki_position = 0.01  # 积分系数
    Kd_position = 0.5 # 微分系数
    setpoint = width / 2  # 目标距离

    # 初始化PID控制器的变量
    position_previous_error = 0
    position_integral = 0

    # 计算误差
    position_error = setpoint - center_x

    # 更新积分项
    position_integral += position_error

    # 计算微分项（这里使用误差的变化）
    position_derivative = position_error - position_previous_error

    # 更新之前的误差值，用于下一次迭代
    position_previous_error = position_error

    # 使用PID公式计算输出
    output = Kp_position * position_error + Ki_position * position_integral + Kd_position * position_derivative

    # 限制输出速度在合理范围内
    max_speed = 15
    min_speed = -15
    if output > max_speed:
        position_speed = max_speed
    elif output < min_speed:
        position_speed = min_speed
    else:
        position_speed = output
        # 使用计算出的速度来驱动小车
    # 假设 ep_chassis 是一个可以接收速度指令的对象
    ep_chassis.drive_speed(x=0, y=0, z=position_speed)




# 函数：处理每一帧图像
def process_frame(frame, ep_camera):
    height, width, c = frame.shape
    # 将图像从BGR转换为RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 进行姿态检测
    results = pose.process(rgb_frame)
    #检测保存人体关键点坐标
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        landmarks = results.pose_landmarks.landmark
        landmark_points = [(lm.x * width, lm.y * height) for lm in landmarks]


        # 计算人体中心点的坐标，即所有关键点的平均位置
        center_x = sum([pt[0] for pt in landmark_points]) / len(landmark_points)
        center_y = sum([pt[1] for pt in landmark_points]) / len(landmark_points)
        # adjust_position_pid(center_x, center_y, width, height)
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)


        pixel_size_in_pixel = calculate_pixel_size(results.pose_landmarks)
        if pixel_size_in_pixel is not None:
            estimated_distance = estimate_distance(focal_length, pixel_size, pixel_size_in_pixel, assumed_body_width)
            print(f"Estimated distance from camera: {estimated_distance:.2f} meters")
            if(center_x> 340 or center_x < 300):
                adjust_position_pid(center_x, center_y, width, height)
            else:
                adjust_chassis_movement(estimated_distance)

    else:
        ep_chassis.drive_speed(x=0, y=0, z=0)


# 函数：显示图像并处理
def display_images(ep_camera):
    global PTime
    while True:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if frame is None:
            continue
        process_frame(frame, ep_camera)

        # 查看FPS
        cTime = time.time()  # 处理完一帧图像的时间
        fps = 1 / (cTime - PTime)
        PTime = cTime  # 重置起始时间
        # 在视频上显示fps信息
        cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Real-time Pose Detection and Distance Estimation', frame)
        # 等待按键，如果按下'q'键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ep_camera.stop_video_stream()

# 主程序
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    # 开启视频流
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    try:
        # 创建并启动显示图像的线程
        display_thread = threading.Thread(target=display_images, args=(ep_camera,))
        display_thread.start()
        while True:
            time.sleep(1)  # 降低主线程的执行频率以减轻负载
    except KeyboardInterrupt:
        # 当用户按下Ctrl+C时退出循环
        print("Program is shutting down...")
    finally:
        # 关闭摄像头
        ep_camera.stop_video_stream()
        # 释放摄像头资源并关闭窗口
        cv2.destroyAllWindows()
        # 关闭机器人连接
        ep_robot.close()
