import cv2
import numpy as np
import argparse
import onnxruntime as ort
import threading
import time
from robomaster import robot, camera, sensor
from queue import Queue

PTime = 0
distance_queue = Queue()
target_queue = Queue()
exit_event = threading.Event()


class PicoDet():
    def __init__(self, model_pb_path, label_path, prob_threshold=0.4, iou_threshold=0.3):
        self.classes = list(map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])

    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=False):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        origin_shape = srcimg.shape[:2]
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')

        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # type: ignore
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=0)  # type: ignore
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)

        return img, scale_factor

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        return color_map

    def detect(self, srcimg):
        img, scale_factor = self.resize_image(srcimg)
        img = self._normalize(img)

        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob, self.net.get_inputs()[1].name: scale_factor})

        outs = np.array(outs[0])
        expect_boxes = (outs[:, 1] > 0.5) & (outs[:, 0] > -1)
        np_boxes = outs[expect_boxes, :]

        # 存储目标框坐标信息的列表
        boxes_info = []

        for i in range(np_boxes.shape[0]):
            classid, conf = int(np_boxes[i, 0]), np_boxes[i, 1]
            xmin, ymin, xmax, ymax = int(np_boxes[i, 2]), int(np_boxes[i, 3]), int(np_boxes[i, 4]), int(
                np_boxes[i, 5])

            # 将目标框的坐标信息添加到列表中
            boxes_info.append({
                'class_id': classid,
                'confidence': conf,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

            # 在图像上绘制检测框和标签
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)
            label = f"{self.classes[classid]}: {conf:.2f}"
            cv2.putText(srcimg, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=2)

        return srcimg, boxes_info


def display_images(ep_camera, first_stage):
    global PTime
    class_id = 0
    while not exit_event.is_set():
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if frame is None:
            continue

        frame, boxes_info = net.detect(frame)

        for box_info in boxes_info:
            class_id = box_info['class_id']
            confidence = box_info['confidence']
            xmin, ymin = box_info['xmin'], box_info['ymin']
            xmax, ymax = box_info['xmax'], box_info['ymax']

            # if class_id == 39 and not first_stage:
            #     center_x = (box_info['xmin'] + box_info['xmax']) / 2
            #     center_y = (box_info['ymin'] + box_info['ymax']) / 2
            #     if center_x>=300 and center_x<=310:
            #         ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            #         first_stage=True
            #         print("center，stage", center_x,first_stage)

            if class_id == 39 and not first_stage:
                center_x = (box_info['xmin'] + box_info['xmax']) / 2
                center_y = (box_info['ymin'] + box_info['ymax']) / 2
                if center_x >= 300 and center_x <= 320:
                    ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
                    first_stage = True
                    print("center，stage", center_x, first_stage)

                # adjust_position_pid(center_x, center_y)
            elif class_id != 39 and not first_stage:
                ep_chassis.drive_speed(x=0, y=0, z=-20)

        target_queue.put((first_stage, class_id))
        cTime = time.time()
        fps = 1 / (cTime - PTime)
        PTime = cTime
        cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Real-time Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ep_camera.stop_video_stream()


def sub_data_handler(sub_info):
    distance_queue.put(sub_info)


def adjust_position_pid(center_x, center_y):
    Kp_position = -40  # 比例系数
    Ki_position = 0  # 积分系数
    Kd_position = 0  # 微分系数
    setpoint = 320  # 目标距离

    # 初始化PID控制器的变量
    position_previous_error = 0
    position_integral = 0

    # 计算误差
    position_error = setpoint - center_x
    print(position_error)

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


def adjust_chassis_movement(distance):
    # PID 控制器的参数
    Kp = -0.5  # 比例系数
    Ki = 0  # 积分系数
    Kd = 0  # 微分系数
    setpoint = 250  # 目标距离
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


def action_grab():
    ep_arm.moveto(0, 50).wait_for_completed(timeout=2)

    ep_arm.moveto(120, 50).wait_for_completed(timeout=2)
    #
    # ep_arm.moveto(150, 0).wait_for_completed(timeout=2)
    #
    # ep_arm.moveto(180, -98).wait_for_completed(timeout=2)

    ep_gripper.close(power=50)
    time.sleep(1)
    ep_gripper.pause()

    ep_arm.moveto(120, 100).wait_for_completed(timeout=2)

    ep_chassis.move(x=-0.5, y=0, z=0, xy_speed=0.7).wait_for_completed()


def action_laydown():
    ep_arm.moveto(120, 50).wait_for_completed(timeout=2)

    ep_gripper.open(power=100)
    time.sleep(2)
    ep_gripper.pause()

    ep_arm.moveto(120, 70).wait_for_completed(timeout=2)

    ep_arm.moveto(0, 50).wait_for_completed(timeout=2)


def grab_target():
    global rotate_flag
    second_stage = False
    third_stage = False
    four_stage = False
    while not exit_event.is_set():
        # center_x,center_y,class_id,width = target_queue.get()
        first_stage, class_id = target_queue.get()
        distance = distance_queue.get()

        if first_stage and not second_stage:
            print(distance)
            # adjust_chassis_movement(distance[0])
            if distance[0] >= 50 and distance[0] <= 70:
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
                ep_gripper.open(power=100)
                time.sleep(2)
                ep_gripper.pause()
                second_stage = True

            elif distance[0] < 50:
                ep_chassis.drive_speed(x=-0.05, y=0, z=0)
            elif distance[0] > 70:
                ep_chassis.drive_speed(x=0.1, y=0, z=0)

        if second_stage and not third_stage:
            # ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            action_grab()
            third_stage = True

        if third_stage and not four_stage:
            # ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            action_laydown()
            four_stage = True

        # print("class_id", center_x,center_y,class_id,width)
        # ep_gripper.open(power=50)
        # ep_gripper.close(power=50)


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_sensor = ep_robot.sensor
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper
    ep_chassis = ep_robot.chassis

    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    ep_sensor.sub_distance(freq=100, callback=sub_data_handler)

    ep_arm.moveto(0, 50).wait_for_completed(timeout=2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='picodet_l_320_lcnet_postprocessed.onnx', help="onnx filepath")
    parser.add_argument('--classfile', type=str, default='coco_label.txt', help="classname filepath")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.6, type=float, help='nms iou thresh')
    args = parser.parse_args()

    net = PicoDet(args.modelpath, args.classfile, prob_threshold=args.confThreshold, iou_threshold=args.nmsThreshold)

    try:
        # distance_thread = threading.Thread(target=process_distance)
        # distance_thread.start()
        display_thread = threading.Thread(target=display_images, args=(ep_camera, False))
        display_thread.start()
        grab_target_thread = threading.Thread(target=grab_target)
        grab_target_thread.start()

        while not exit_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program is shutting down...")
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        ep_gripper.open(power=100)
        time.sleep(1)
        ep_gripper.pause()
        ep_arm.moveto(0, 50).wait_for_completed(timeout=2)
    finally:
        exit_event.set()
        ep_camera.stop_video_stream()
        ep_sensor.unsub_distance()
        distance_queue.join()
        target_queue.join()
        cv2.destroyAllWindows()
        ep_robot.close()
