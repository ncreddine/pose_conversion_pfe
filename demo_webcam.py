#!/usr/bin/env python3

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
import copy

from data.utils import skeleton_connections

import mediapipe as mp
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def length(A):
    return np.linalg.norm(A)

def draw_poses(ax, ar,  pose, default_skeleton, color, label):
    data = pose.reshape(52,3)
    artist = ax.scatter(*data.T, s = 10, color = color, label = label)
    ar.append(artist)
    for origin, end in default_skeleton :
        x, y , z   = data[end]
        xx, yy, zz = data[origin]
        line, = ax.plot([x, xx],[y, yy],[z,zz], color = color,  linewidth = 2)
        ar.append(line)
    return ax, ar

class Demo:
    def __init__(self, path_to_model, num_threads):
        self.interpreter = tf.lite.Interpreter(
                model_path = path_to_model,
                num_threads = num_threads )
        self.interpreter.allocate_tensors()
        
        # I/O details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # set mediapipe hollistics
        self.holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity = 2, min_detection_confidence = 0.9)

        # plot utils
        self.fig = plt.figure(figsize=(14,6))
        self.axis = [self.fig.add_subplot(121),
                     self.fig.add_subplot(122, projection='3d')]
        self.fig.tight_layout()

        # start video capture
        self.cap = cv.VideoCapture(0)

        # interractive mode
        plt.ion()
        plt.show()
        self.clear_axis(self.axis[1])
        self.axis[1].view_init(90,90)

    def clear_axis(self, axis):
        axis.cla()
        axis.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])

    def start(self):
        self.anim = FuncAnimation(self.fig, self.animate,interval = 1, blit = False)
        plt.pause(0)

    def animate(self,i):
        self.axis[0].clear()
        self.clear_axis(self.axis[1])

        ret, image = self.cap.read()
        # 
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        # 
        image.flags.writeable = True

        self.axis[0].imshow(image)
        _pose3d, _handR, _handL = self.get_coord(results)
        if isinstance(_pose3d, np.ndarray) and isinstance(_handR, np.ndarray) and isinstance(_handL, np.ndarray) :
            # Convert skeleton
            pose_image  = self.adapt_skeleton(_pose3d, _handR, _handL, image.shape)
            data  = self.normalize_pose(pose_image)
            x  = data[:,:2].reshape(1, 52, 2)
            predicted = self.infer_model(x)

            self.axis[1], _ = draw_poses(self.axis[1], [], predicted, skeleton_connections(), color = "red", label = "ours")
            self.axis[1], _ = draw_poses(self.axis[1], [], data.flatten(), skeleton_connections(), color = "blue", label = "GT using mediapipe")
            self.axis[1].legend()

    def infer_model(self, data) :
        # Convert to float32
        input_data = np.float32(data).reshape(1,-1)

        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data

    def adapt_skeleton(self, _pose3d, _handR, _handL, shape):
        height, width = shape[:2]
        ## Substract hand origin
        _handR  -= [1, 1, _handR[0,2]]
        _handL  -= [1, 1, _handL[0,2]]
        ## Scale the hand
        f = 4
        _handR  *= [1, 1, f]
        _handL  *= [1, 1, f]
        ## Add wrist distance
        _handR  += [1, 1, _pose3d[16,2]]
        _handL  += [1, 1, _pose3d[15,2]]
        #  create new list
        new_pose = []
        #  add 'Neck'
        new_pose.append((_pose3d[11] + _pose3d[12])/2)
        #  add 'RShoulder', 'RElbow', 'RWrist', 
        new_pose.append(_pose3d[12]); new_pose.append(_pose3d[14]); new_pose.append(_pose3d[16]);
        #  add 'LShoulder', 'LElbow', 'LWrist',
        new_pose.append(_pose3d[11]); new_pose.append(_pose3d[13]); new_pose.append(_pose3d[15]);
        #  add 'Nose', 'REye', 'LEye'
        new_pose.append(_pose3d[ 0]); new_pose.append(_pose3d[ 5]); new_pose.append(_pose3d[ 2]);
        #  add 'LHand'
        for landmark in _handL :
            new_pose.append(landmark)
        #  add 'RHand'
        for landmark in _handR :
            new_pose.append(landmark)

        new_pose = np.array(new_pose)
        return new_pose 
#
    def normalize_pose(self,pose):
        scale =  length(pose[4] - pose[1])
        return (pose - pose[0]) / scale

    def get_coord(self, results):
        # Get pose, rhand , lhand cordinates
        pose = results.pose_landmarks
        right_hand = results.right_hand_landmarks
        left_hand  = results.left_hand_landmarks
        
        #
        if not right_hand == None :
            right_hand = np.array([(p.x, p.y, p.z) for p in results.right_hand_landmarks.landmark])
        #
        if not left_hand == None :
            left_hand  = np.array([(p.x, p.y, p.z) for p in results.left_hand_landmarks.landmark])
        #
        if not  pose == None :
            pose  =      np.array([(p.x, p.y, p.z) for p in results.pose_landmarks.landmark])
        return pose, right_hand, left_hand
        
if __name__ == '__main__':
    demo = Demo('./models/linear_model.tflite', num_threads = 6)
    demo.start()
