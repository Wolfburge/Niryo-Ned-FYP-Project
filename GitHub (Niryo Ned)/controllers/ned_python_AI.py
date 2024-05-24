# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ned_Controller controller in python.

# Webots controller for the Niryo Ned robot.
# With this controller, you can see the 6 different axis of the robot moving
# You can also control the robots with your keyboard and launch a Pick and Pack

import json 
from controller import Robot
from controller import Keyboard
from controller import Camera
from controller import Supervisor
from controller import PositionSensor
from pathlib import Path 
from PIL import Image
import numpy as np 
import tensorflow as tf
from tensorflow import keras
import cv2
import time
from imageai.Detection.Custom import CustomObjectDetection


Movement_Model_Path = Path("C:/Users/MOHAMED RIYAZ/Desktop/FYP PROJECT/controllers/saved_model (79).h5")
Object_Detectection_Model_Path = Path("C:/Users/MOHAMED RIYAZ/Desktop/yolov3_FINAL_DRAFT_last (7).pt")
Json_File_Path = Path("C:/Users/MOHAMED RIYAZ/Desktop/FINAL_DRAFT_yolov3_detection_config.json")


detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("C:\\Users\\MOHAMED RIYAZ\\Desktop\\FYP PROJECT\\controllers\\yolov3_FINAL_DRAFT_last (7).pt")
detector.setJsonPath("C:\\Users\\MOHAMED RIYAZ\\Desktop\\FYP PROJECT\\controllers\\FINAL_DRAFT_yolov3_detection_config.json")
detector.loadModel()

Movement_Model = keras.models.load_model(Movement_Model_Path)

s1 = 0.1
s2 = 0.1
s3= 0.1
s4= 0.1
s5 = 0.1
s6 = 0.1

IMAGE_FOLDER = Path("C:/Users/MOHAMED RIYAZ/Desktop/JSON FOLDER/images")
POSITION_FOLDER = Path("C:/Users/MOHAMED RIYAZ/Desktop/JSON FOLDER/json flies")
s = 111

robot = Supervisor() 
can = robot.getFromDef("NedBox")
can_pos = can.getField("translation")
can_ros = can.getField("rotation")
origin_can = can_pos.getSFVec3f()
origin_ros = can_ros.getSFRotation()

camera = robot.getDevice("cam")
Camera.enable(camera,64)  # Enable the camera with a certain time ste

robot_name = robot.getName()
print('Name of the robot: ' + robot_name + '\n')

# Init the motors - the Ned robot is a 6-axis robot arm
# You can find the name of the rotationalMotors is the device parameters of each HingeJoints
m1 = robot.getDevice('joint_1')
m2 = robot.getDevice('joint_2')
m3 = robot.getDevice('joint_3')
m4 = robot.getDevice('joint_4')
m5 = robot.getDevice('joint_5')
m6 = robot.getDevice('joint_6')
m7 = robot.getDevice('gripper::left')
m8 = robot.getDevice('gripper::right')
p1 = m1.getPositionSensor()
p1.enable(64)
p2 = m2.getPositionSensor()
p2.enable(64)
p3 = m3.getPositionSensor()
p3.enable(64)
p4 = m4.getPositionSensor()
p4.enable(64)
p5 = m5.getPositionSensor()
p5.enable(64)
p6 = m6.getPositionSensor()
p6.enable(64)
p7 = m7.getPositionSensor()
p7.enable(64)
p8 = m8.getPositionSensor()
p8.enable(64)
# Set the motor velocity
# First we make sure that every joints are at their initial positions
# m1.setPosition(0)
# m2.setPosition(0)
# m3.setPosition(0)
# m4.setPosition(0)
# m5.setPosition(0)
# m6.setPosition(0)
# m7.setPosition(0)

# Set the motors speed. Here we set it to 1 radian/second
m1.setVelocity(0.3)
m2.setVelocity(0.3)
m3.setVelocity(0.3)
m4.setVelocity(0.3)
m5.setVelocity(0.3)
m6.setVelocity(0.3)
m7.setVelocity(2)
m8.setVelocity(2)

# ----Function to realize a demo of the Ned robot moving----
def demo():
    m1.setVelocity(1)
    m2.setVelocity(1)
    m3.setVelocity(1)

    if robot.step(0) == -1:
        return
    m1.setPosition(1.6)
    m7.setPosition(0.01)

    if robot.step(1500) == -1:
        return
    m1.setPosition(0)

    if robot.step(1500) == -1:
        return
    m2.setPosition(0.5)

    if robot.step(700) == -1:
        return
    m2.setPosition(0)

    if robot.step(700) == -1:
        return
    m1.setPosition(-0.5)
    m4.setPosition(1.4)

    if robot.step(1500) == -1:
        return
    m4.setPosition(0)

    if robot.step(1500) == -1:
        return
    m5.setPosition(-1)

    if robot.step(700) == -1:
        return
    m5.setPosition(0)

    if robot.step(1000) == -1:
        return
    m3.setPosition(0)
    m1.setPosition(0)

    if robot.step(500) == -1:
        return
    m6.setPosition(1.5)

    if robot.step(1000) == -1:
        return
    m6.setPosition(0)

    if robot.step(1000) == -1:
        return
    m7.setPosition(0)

    if robot.step(500) == -1:
        return
    m7.setPosition(0.01)

def get_image(camera):
    image = camera.getImage()
    image = np.frombuffer(image, np.uint8).reshape(
    (camera.getHeight(), camera.getWidth(), 4)
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    im = Image.fromarray(image)
    image_path = "C:/Users/MOHAMED RIYAZ/Desktop/Test_Image_Bounding/image1.png"
    im.save(image_path)
    return image

def pick_place(camera):
    print("predicting")
    image = get_image(camera)
    boxes = detector.detectObjectsFromImage(nms_treshold=0.1,objectness_treshold=0.1,input_image= image, output_image_path= "C:/Users/MOHAMED RIYAZ/Desktop/Test_Image_Bounding/image.png")
    if len(boxes)>0:
        block = boxes[0]
        if block["name"] == "block":
            cord = block["box_points"]
            cord = np.asarray(cord)
            cord = cord.reshape(1, cord.shape[0])
            print(cord)
            pos_hand = np.array(Movement_Model(cord,training=False))
            print(pos_hand)
        m1.setPosition(pos_hand[0][0])
        m2.setPosition(pos_hand[0][1])
        m3.setPosition(pos_hand[0][2])
        m4.setPosition(pos_hand[0][3])
        m5.setPosition(pos_hand[0][4])
        m6.setPosition(pos_hand[0][5]) 
        ##m7.setPosition(pos_hand[0][6])
        ##m8.setPosition(pos_hand[0][7])
        return([pos_hand[0][6],pos_hand[0][7]])

def save_position(m1,m2,m3,m4,m5,m6,m7,m8,s):
    command = {"m1":m1,"m2":m2,"m3":m3,"m4":m4,"m5":m5,"m6":m6,"m7":m7,"m8":m8}
    print(command)
    print("ugfdewyv")
    with open(POSITION_FOLDER/f"pos_{s}.json","w") as f:
        json.dump(command,f)
flag = f=False
while True:

    print("------------COMMANDS--------------")
    print("Launch demo --> SHIFT+D")
    print("Move joint_1 --> SHIFT+A or SHIFT+Z")
    print("Move joint_2 --> SHIFT+Q or SHIFT+S")
    print("Move joint_3 --> SHIFT+W or SHIFT+X")
    print("Move joint_4 --> SHIFT+Y or SHIFT+U")
    print("Move joint_5 --> SHIFT+H or SHIFT+J")
    print("Move joint_6 --> SHIFT+B or SHIFT+N")
    print("Open/Close Gripper --> SHIFT+L or SHIFT+M")
    print("Launch Pick and Place --> SHIFT+P")
    print("Move Block back to original position --> SHIFT+C")
    print("----------------------------------")
    

    timestep = int(robot.getBasicTimeStep())
    keyboard = Keyboard()
    keyboard.enable(timestep)

    timer = 0
    flag = False
    flag2 = False
    flag3 = False
    while robot.step(timestep) != -1:
        key = keyboard.getKey()
        if key == Keyboard.SHIFT + ord('A'):
            print("Move --> joint_1 left")
            r1 = p1.getValue() -s1
            m1.setPosition(max([r1,-2.5]))

        elif key == Keyboard.SHIFT + ord('Z'):
            print("Move --> joint_1 right")
            r1 = p1.getValue() +s1
            m1.setPosition(min([r1,2.5]))

        elif key == Keyboard.SHIFT + ord('Q'):
            print("Move --> joint_2 left")
            r2 = p2.getValue() +s2
            m2.setPosition(min([r2,0.8]))

        elif key == Keyboard.SHIFT + ord('S'):
            print("Move --> joint_2 right")
            r2 = p2.getValue() -s2
            m2.setPosition(max([r2,-0.8]))

        elif key == Keyboard.SHIFT + ord('W'):
            print("Move --> joint_3 left")
            r3 = p3.getValue() +s3
            m3.setPosition(min([r3,1.3]))

        elif key == Keyboard.SHIFT + ord('X'):
            print("Move --> joint_3 right")
            r3 = p3.getValue() -s3
            m3.setPosition(max([r3,-1.5]))

        elif key == Keyboard.SHIFT + ord('Y'):
            print("Move --> joint_4 left")
            r4 = p4.getValue() +s4
            m4.setPosition(min([r4,2]))

        elif key == Keyboard.SHIFT + ord('U'):
            print("Move --> joint_4 right")
            r4 = p4.getValue() -s4
            m4.setPosition(max([r4,-2]))

        elif key == Keyboard.SHIFT + ord('H'):
            print("Move --> joint_5 left")
            r5 = p5.getValue() +s5
            m5.setPosition(min([r5,1.5]))

        elif key == Keyboard.SHIFT + ord('J'):
            print("Move --> joint_5 right")
            r5 = p5.getValue() -s5
            m5.setPosition(max([r5,-1.5]))

        elif key == Keyboard.SHIFT + ord('B'):
            print("Move --> joint_6 left")
            r6 = p6.getValue() +s6
            m6.setPosition(min([r6,2.5]))

        elif key == Keyboard.SHIFT + ord('N'):
            print("Move --> joint_6 right")
            r6 = p6.getValue() -s6
            m6.setPosition(max([r6,-2.5]))

        elif key == Keyboard.SHIFT + ord('L'):
            print("Move --> Open Gripper")
            m7.setPosition(0.01)
            m8.setPosition(-0.01)

        elif key == Keyboard.SHIFT + ord('M'):
            print("Move --> Close Gripper")
            m7.setPosition(-0.01)
            m8.setPosition(0.01)
        elif key == Keyboard.SHIFT + ord('D'):
            print("Demonstrator: Move Joints")
            demo()

        elif key == Keyboard.SHIFT + ord('P'):
            print("Demonstrator: Pick And Place")
            pick_place(camera)
            flag = True
            timer = 0
        
        elif key == Keyboard.SHIFT + ord('C'):
            print(can_pos)
            can_pos.setSFVec3f(origin_can)
            can_ros.setSFRotation(origin_ros)


        elif key == ord('C'):
            origin_can = can_pos.getSFVec3f()
            origin_ros = can_ros.getSFRotation()
            print(origin_ros)

        elif key == ord('R') and flag:
        # Capture image from the camera
            image = camera.getImage()
            image = np.frombuffer(image, np.uint8).reshape(
            (camera.getHeight(), camera.getWidth(), 4)
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            im = Image.fromarray(image)
            image_path = IMAGE_FOLDER / f"image_{s}.png"
            im.save(image_path)
            s+=1
            flag = False

        elif key == ord('P') and flag == False:
            save_position(p1.getValue(),p2.getValue(),p3.getValue(),p4.getValue(),p5.getValue(),p6.getValue(),p7.getValue(),p8.getValue(),s)
            flag=True

        elif key == ord("I"):
            flag = False

        elif key == ord("T"):
            m1.setPosition(1.9)
            m2.setPosition(0)
            m3.setPosition(0)
            m4.setPosition(0)
            m5.setPosition(0)
            m6.setPosition(0)
            m7.setPosition(0.01)
            m8.setPosition(0.01)

        elif key == Keyboard.SHIFT + ord('O'):
            m1.setPosition(0.375)
            m2.setPosition( 0)
            m3.setPosition(0)
            m4.setPosition(0)
            m5.setPosition(0)
            m6.setPosition(0)
            flag2 = True
            timer = 0
           
        if timer > 200 and flag:
            m7.setPosition(-0.008)
            m8.setPosition(-0.008)
            flag = False
            print("closing gripper")

        if  timer > 300 and flag2:
            m1.setPosition(0.375)
            m2.setPosition( 0.373)
            m3.setPosition(0.890)
            m4.setPosition(4.950e-09)
            m5.setPosition(-4.094e-09)
            m6.setPosition(4.148e-11)
            flag3 = True
            flag2 = False
            timer = 0

        if timer > 300 and flag3:
            m7.setPosition(0.01)
            m8.setPosition(0.01)
            flag3 = False
            print("opening gripper")

        

        timer += 1

            


            


