import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# import tensorflow as tf
import h5py
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm

# Video clipping
from yt_dlp import YoutubeDL
from stonesoup.reader.video import VideoClipReader
from datetime import timedelta

from copy import copy
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import animation
import os
import pathlib

PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 7, 6, 10, 11, 12, 13, 10, 15, 16, 17, 10, 19, 20, 21, 10, 23, 24, 25, 10, 27, 28, 29, 3, 31, 32, 33, 34, 31, 36, 37, 38, 31, 40, 41, 42, 31, 44, 45, 46, 31, 48, 49, 50] # colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
default_skeleton = list(zip(range(len(PARENTS)), PARENTS))[1:]

def draw_poses(ax, ar,  pose, default_skeleton):
    data = pose.T
    #---
    # off = data[0]
    # off = np.flip(np.tile(off, (52,1)), axis = 1)
    # data[0] = [0,0]
    # data = data + off
    #---
    # data[0] = [0,0]
    #---
    artist = ax.scatter(*data.T, s = 10, color = 'blue')
    ar.append(artist)
    for origin, end in default_skeleton :
        x, y   = data[end] 
        xx, yy = data[origin]
        x = x  ; y = y 
        xx = xx; yy = yy
        line, = ax.plot([x, xx],[y, yy], color = 'blue',  linewidth = 2)
        ar.append(line)
    return ax, ar

def download_video(url, filename):
    """Download youtube video. Ensure it has a unique hour and minute timestamp , and rsync to the youtube library folder."""
    ydl_opts = {
        "no_warnings": True,
        "quiet" : 1,
        'logtostderr' : True,
        "noprogress": False,
        "outtmpl" : filename,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download(url)
    except Exception as e: 
        pass
        # if isinstance(e, DownloadError):
        #     pass
        # elif hasattr(e, 'message'):
        #     if "Command returned error code 23" in e.message:
        #         pass
        #     else:
        #         raise(e)
        # else:
        #     raise(e)   

def get_timedelta(isoformat):
    try :
        s, microsecond = isoformat.split(".")
    except :
        s = isoformat
        microsecond = 0
    microsecond = min( int(microsecond) , timedelta.max.microseconds)
    hour, minute , second =  list(map(int, s.split(":")))
    return timedelta(minutes = minute, seconds = second, microseconds = microsecond)

exluded_speakers = []
class Dataset :
    def __init__(self, filename, path_to_data, data_storage) :
        # Mediapipe pose
        self.holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity = 2, min_detection_confidence = 0.85)

        self.path_to_data = pathlib.Path(path_to_data)
        self.file_ = open(filename, 'r')
        self.data_storage = data_storage
        self.data  = json.load(self.file_)

        # Create a dumping download file
        self._downloads = pathlib.Path('./_downloads')
        if not self._downloads.exists() : 
            self._downloads.mkdir()

        # Build the dataset
        self.build_dataset()

    def build_dataset(self):
        # Create dataset
        dataset = pathlib.Path(self.data_storage ) / 'dataset'
        if not dataset.exists() : dataset.mkdir()

        # Build train
        trainset = dataset / 'train'  ;
        if not trainset.exists() : trainset.mkdir()
        self.build_subset(trainset, 'train')
        # Build dev
        devset   = dataset / 'dev' 
        if not devset.exists() : devset.mkdir()
        self.build_subset(devset, 'dev')
        # Build test
        testset  = dataset / 'test';
        if not testset.exists() : testset.mkdir()
        self.build_subset(testset, 'test')

    def build_subset(self, subset, name_subset) :
        # Each speaker
        num_speakers = len(self.data[name_subset])
        with tqdm(total=num_speakers, bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = False)  as pbar :
            for speaker in self.data[name_subset] :
                if not speaker in exluded_speakers :
                    pbar.set_description(speaker)
                    # self.for_speaker("seth", subset, name_subset)
                    self.for_speaker(speaker, subset, name_subset)
            pbar.close()

    def for_speaker(self, speaker, subset, name_subset):
        # Create a speaker folder and store intervals in it
        speaker_ = subset / speaker 
        if not speaker_.exists() :  speaker_.mkdir()
        # Youtube links of each speaker
        link_set = self.data[name_subset][speaker]
        # Loop through each links
        for ind, link in enumerate(tqdm(link_set, desc = 'Links', bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = False)) :
            # Download Youtube video
            filename = f'{subset.name}_{speaker}_link{ind}'
            VIDEO_PATH = os.path.join(self._downloads, filename )
            download_video(link, VIDEO_PATH)
            VIDEO_PATH = list(self._downloads.glob(filename +'.*'))[0]
            # Get intervals
            intervals = link_set[link]
            # 
            interval_dict = {}
            for  interval_id in tqdm(intervals, desc = 'Intervals' ,  bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = False) :
                # Timecodes
                timecodes = intervals[interval_id]
                # Extract poses
                status, pose2d, pose3d, missing_intervals = self.get_pose_interval(speaker, interval_id, timecodes, VIDEO_PATH)
                if status :
                    h5f = h5py.File( speaker_ / f'{interval_id}.hdf5', 'w')
                    h5f.create_dataset('missing_intervals', data = np.int16(missing_intervals))
                    h5f.create_dataset('pose2d', data = pose2d)
                    h5f.create_dataset('pose3d', data = pose3d)
                    h5f.close()

        
    def get_pose_interval(self, speaker, interval_id, timecodes, VIDEO_PATH):
        """ Read from the annotated 2d poses and extract 3d pose from image"""
        # Start & End time 
        start_time = timecodes['start_time']
        end_time   = timecodes['end_time']

        # Read video & clip
        frame_reader = VideoClipReader(VIDEO_PATH,  get_timedelta(start_time), get_timedelta(end_time)) # src : https://stonesoup.readthedocs.io/en/v0.1b5/auto_demos/Video_Processing.html
        list_frame_reader = list(frame_reader)
        video_frames_  = len(list_frame_reader)

        # Read 2d pose from .h5 files
        pose_2d = self.get_2dpose(speaker, interval_id)
        annotated_frames = len(pose_2d)
        
        missing_frames = []
        p2d = []
        p3d = []

        # Read the video framen
        for ind in tqdm(range(annotated_frames), desc =f'Frames', bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = False):
        # for ind in range(annotated_frames) :
            # Read the equivalent frame from the set
            _ , frame = list_frame_reader[round( ind * video_frames_ / annotated_frames)]
            pixels = copy(frame.pixels)
            image = Image.fromarray(pixels)
            image = np.array(image)
            shape = image.shape
            # 2D pose
            data = pose_2d[ind].reshape(2, 52).T
            data[0] = [0,0]
            # 3D pose
            results = self.holistic.process(image)
            # Preprocess the image
            _pose3d, _handR, _handL = self.get_coord(results)
            if isinstance(_pose3d, np.ndarray) and isinstance(_handR, np.ndarray) and isinstance(_handL, np.ndarray) :
                # Convert skeleton
                _3d_all = self.adapt_skeleton(_pose3d, _handR, _handL, image.shape)
                _3d_all = self.normalize_pose(_3d_all)
                data    = self.normalize_pose(data)
                p2d.append(data)
                p3d.append(_3d_all)
            else : 
                missing_frames.append(ind)
        if len(p2d) == 0  or len(p3d) == 0 :
            return False,  p2d, p3d , missing_frames
        else : 
            p2d, p3d , missing_frames = np.vstack(p2d), np.vstack(p3d), missing_frames
            return True, p2d, p3d , missing_frames
        return p2d, p3d , missing_frames

    def adapt_skeleton(self, _pose3d, _handR, _handL, shape):
        height, width = shape[:2]
        _pose3d *= [width, height, width]
        _handR  *= [width, height, width]
        _handL  *= [width, height, width]
        #  create new list
        new_pose = []
        #  add neck
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

    def normalize_pose(self,pose):
        scale =  length(pose[4] - pose[1])
        return (pose - pose[0]) / scale

    def get_coord(self, results):
        # Get pose, rhand , lhand cordinates
        pose = results.pose_landmarks
        right_hand = results.right_hand_landmarks
        left_hand  = results.left_hand_landmarks
        
        if not right_hand == None :
            right_hand = np.array([(p.x, p.y, p.z) for p in results.right_hand_landmarks.landmark])
        #
        if not left_hand == None :
            left_hand  = np.array([(p.x, p.y, p.z) for p in results.left_hand_landmarks.landmark])
        #
        if not  pose == None :
            pose  =      np.array([(p.x, p.y, p.z) for p in results.pose_landmarks.landmark])
        return pose, right_hand, left_hand

    def get_2dpose(self, speaker, interval_id) :
        path_pose_2d = self.path_to_data / f'{speaker}/pats/data/processed/{speaker}' / f'{interval_id}.h5'
        pose_2d = h5py.File(path_pose_2d)
        data = pose_2d['pose']['data']        # data = data + data[0]
        return data


def length(A):
    return np.linalg.norm(A)

dataset = Dataset("./metadata.json", "/media/holdee/MyPassport/PATS", "/media/holdee/MyPassport/")
# dataset = Dataset("./metadata.json", "/home/holdee/Documents/M2/PFE/conversion/", ".
