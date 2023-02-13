#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argcomplete, argparse, json
from argcomplete.completers import ChoicesCompleter

def argcomplete_speaker(prefix, parsed_args, **kwargs) :
    """ Argument Completer for speaker """
    file_ = open(parsed_args.from_json, 'r')
    data  = json.load(file_)[parsed_args.set]
    return (spk for spk in data if spk.startswith(prefix))

def argcomplete_video(prefix, parsed_args, **kwargs) :
    """ Argument Completer for video """
    file_ = open(parsed_args.from_json, 'r')
    data  = json.load(file_)[parsed_args.set][parsed_args.speaker]
    return (vid.split('||')[1] for vid in data if vid.split('||')[1].startswith(prefix))

def argcomplete_interval(prefix, parsed_args, **kwargs) :
    """ Argument Completer for intervals """
    file_ = open(parsed_args.from_json, 'r')
    data  = json.load(file_)[parsed_args.set][parsed_args.speaker]
    key   = list(filter(lambda x : x.split('||')[1] == parsed_args.video, list(data.keys())))[0]
    data = data[key]
    return ( itrv for itrv in data if itrv.startswith(prefix))


def download_video(url, filename):
    """Download youtube video. Ensure it has a unique hour and minute timestamp , and rsync to the youtube library folder."""
    ydl_opts = {
        "no_warnings": True,
        "quiet" : 1,
        'logtostderr' : True,
        "noprogress": False,
        "outtmpl" : filename
    }
    # try:
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)

            # return True
    # except Exception as e: 
        # print(e)

def get_timedelta(isoformat):
    try :
        s, microsecond = isoformat.split(".")
    except :
        s = isoformat
        microsecond = 0
    microsecond = min( int(microsecond) , timedelta.max.microseconds)
    hour, minute , second =  list(map(int, s.split(":")))
    return timedelta(minutes = minute, seconds = second, microseconds = microsecond)

def draw_poses(ax, ar,  pose, default_skeleton, color):
    data = pose.reshape(52,3)
    x_, y_, z_ = data.T
    artist = ax.scatter(x_, z_, y_,  s = 10, color = color)
    ar.append(artist)
    for origin, end in default_skeleton :
        x, y , z   = data[end]
        xx, yy, zz = data[origin]
        line, = ax.plot([x, xx],[z, zz],[y,yy], color = color,  linewidth = 2)
        ar.append(line)
    return ax, ar

class Demo :
    """ Demo script that can be used with wether the webcam or on pats PATS """
    def __init__(self, args):
        self.args = args
        # Ground Truth pose
        self.holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity = 2, min_detection_confidence = 0.85)
        # Inference module
        self.init_interpreter(args.model, args.num_threads)
        # Figure plot
        self.init_fig()

        if args.webcam :
            self.cap = cv.VideoCapture(0)
        else :
            # Downloads folder
            self._downloads = pathlib.Path('./downloads')
            if not self._downloads.exists() : 
                self._downloads.mkdir()
            # Read JSON file
            self.data  = json.load(open(args.from_json, 'r'))
            speaker_dict = self.data[args.set][args.speaker]
            key   = list(filter(lambda x : x.split('||')[1] == args.video, list(speaker_dict.keys())))[0]
            timecodes = self.data[args.set][args.speaker][key][args.interval]
            # Download the video
            url, _ = key.split('||')
            path_to_video  =  self._downloads / args.video
            download_video(url, path_to_video.as_posix())
            # find filename with real extension 
            filename = list(self._downloads.glob(args.video +'*'))[0]
            # Read video & clip
            frame_reader = VideoClipReader(filename,  get_timedelta(timecodes['start_time']), get_timedelta(timecodes['end_time'])) # src : https://stonesoup.readthedocs.io/en/v0.1b5/auto_demos/Video_Processing.html
            self.frame_reader_ = list(frame_reader)
            video_frames_  = len(self.frame_reader_)


    def start(self):
        """ Begin the demo using FuncAnimation from matplotlib """
        if self.args.webcam :
            self.anim = FuncAnimation(self.fig, self.animate,interval = 1, blit = False)
            plt.pause(0)
        elif self.args.from_json :
            self.anim = FuncAnimation(self.fig, self.animate, frames=len(self.frame_reader_), interval=1, blit = False, repeat = False)
            plt.pause(0)
            if self.args.save :
                writer = animation.FFMpegWriter(fps=25, metadata=dict(artist='Me'), bitrate=1800)
                self.anim.save('./mydemo.mp4', writer = writer)
            plt.show()

    def animate(self, i):
        """ animation on each frame """
        self.axis[0].clear()
        self.axis[0].set_xticks([])
        self.axis[0].set_yticks([])
        self.clear_axis(self.axis[1])
        self.clear_axis(self.axis[2])

        if args.webcam :
            ret, image = self.cap.read()
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else :
            _, image = self.frame_reader_[i]
            image = np.array(image.pixels)
        results = self.holistic.process(image)
        self.axis[0].imshow(image)
        _pose3d, _handR, _handL = self.get_coord(results)
        if isinstance(_pose3d, np.ndarray) and isinstance(_handR, np.ndarray) and isinstance(_handL, np.ndarray) :
            # Convert skeleton
            pose_image  = self.adapt_skeleton(_pose3d, _handR, _handL, image.shape)
            gt  = self.normalize_pose(pose_image)
            gt_2d  = gt[:,:2].reshape(1, 52, 2)
            predicted = self.infer_model(gt_2d)

            self.axis[1], _ = draw_poses(self.axis[1], [], gt.flatten(), skeleton_connections(), color = "blue")
            self.axis[2], _ = draw_poses(self.axis[2], [], predicted, skeleton_connections(), color = "red")
            self.axis[1].set_title("mediapipe")
            self.axis[2].set_title("2D to 3D conversion")
        
    def infer_model(self, data) :
        # Convert to float32
        input_data = np.float32(data).reshape(1,-1)

        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data

    def adapt_skeleton(self, _pose3d, _handR, _handL, shape):
        ## Tanslation to z = 0
        _handR  -= [1, 1, _handR[0,2]]
        _handL  -= [1, 1, _handL[0,2]]
        ## Scale the hand
        f = 4
        _handR  *= [1, 1, f]
        _handL  *= [1, 1, f]
        ## Add wrist - hand origin distance
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
        """ Normalization 
            @param : pose skeleton
        """
        scale =  np.linalg.norm(pose[4] - pose[1])
        return (pose - pose[0]) / scale

    def get_coord(self, results):
        """ Read landmarks from mediapipe results.
            Manage the case where neither a hand or a pose are detected and convert them to numpy """
        pose = results.pose_landmarks
        right_hand = results.right_hand_landmarks
        left_hand  = results.left_hand_landmarks
        if results.right_hand_landmarks is not None :
            right_hand = np.array([(p.x, p.y, p.z) for p in results.right_hand_landmarks.landmark])
        if results.left_hand_landmarks is not None :
            left_hand  = np.array([(p.x, p.y, p.z) for p in results.left_hand_landmarks.landmark])
        if results.pose_landmarks is not  None :
            pose  =      np.array([(p.x, p.y, p.z) for p in results.pose_landmarks.landmark])
        return pose, right_hand, left_hand


    def init_interpreter(self, model, num_threads) :
        """ Initialize model interpreter for the tflite file
            @param  : model path to `.tflite` model
                    : num_threads parameters to the model
        """
        self.interpreter = tf.lite.Interpreter(
                model_path = args.model,
                num_threads = args.num_threads )
        self.interpreter.allocate_tensors()
        # Model i/o details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def init_fig(self):
        """ Figure params"""
        self.fig = plt.figure(figsize=(10,6))
        self.axis = [self.fig.add_subplot(131),
                     self.fig.add_subplot(132, projection='3d'),
                     self.fig.add_subplot(133, projection='3d')]
        self.fig.tight_layout()
        self.count_frames = 0
        plt.ion() # live plot
        plt.show()
        self.clear_axis(self.axis[1])
        self.axis[1].view_init(180, -60)
        self.axis[2].view_init(180, -60)

    def clear_axis(self, axis):
        """ Clear axis between frames """
        axis.cla()
        axis.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])


def parse_args() :
    """ Argument parser """
    parser = argparse.ArgumentParser(prog='PROG')

    parser.add_argument('--model', type = str,  help = 'Choose the tflite model')
    parser.add_argument('--num_threads', type = int, help = 'Inference parameter')

    parser.add_argument('--webcam', action = 'store_true', help = "Choose to read from webcam")

    parser.add_argument('--from_json', type=str,  help = "Path to `sorted.json`")
    parser.add_argument('--set', type=str,  help = "Select 'train', 'dev', 'test'").completer = ChoicesCompleter(('train', 'test', 'dev'))
    parser.add_argument('--speaker', type=str,  help = 'Select speaker name').completer = argcomplete_speaker
    parser.add_argument('--video', type=str,  help = 'Choose video name').completer = argcomplete_video
    parser.add_argument('--interval', type=str,  help = 'Choose interval_id').completer = argcomplete_interval
    parser.add_argument('--save',action = 'store_true', help = 'Save demo as GIF')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.webcam and (args.from_json or args.set or args.speaker or args.video or args.interval or args.save ) :
        print('Invalid flags: --webcam and --from_json cannot be set at the same time')
        exit()

    if not (args.webcam) and not ( args.from_json and args.set and args.speaker and args.video and args.interval) :
        print('Missing args: respect order from_json, set, speaker, video, interval')
        exit()

    
    return args

if __name__ == '__main__':
    args = parse_args()
    # imports
    from matplotlib.animation import FuncAnimation
    from matplotlib import animation
    import matplotlib.pyplot as plt
    import silence_tensorflow.auto
    import tensorflow as tf
    import numpy as np
    import copy, pathlib
    import mediapipe as mp
    from yt_dlp import YoutubeDL
    from stonesoup.reader.video import VideoClipReader
    from datetime import timedelta
    import cv2 as cv
    from data.utils import skeleton_connections
    # init and launch
    demo = Demo(args)
    demo.start()
