import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

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

def download_video(url, filename):
    """Download youtube video. Ensure it has a unique hour and minute timestamp , and rsync to the youtube library folder."""
    ydl_opts = {
        "no_warnings": True,
        "overwrites": False,
        "restrictfilenames": True,
        "noprogress": True,
        "prefer_ffmpeg": True,
        "outtmpl" : filename,
        # "postprocessors": [ 
        #     {
        #         'key':'FFmpegMetadata',
        #         'add_metadata': True,
        #     },
        # ]
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download(url)
    except Exception as e: 
        if isinstance(e, DownloadError):
            pass
        elif hasattr(e, 'message'):
            if "Command returned error code 23" in e.message:
                pass
            else:
                raise(e)
        else:
            raise(e)   

def get_timedelta(isoformat):
    try :
        s, microsecond = isoformat.split(".")
    except :
        s = isoformat
        microsecond = 0
    microsecond = min( int(microsecond) , timedelta.max.microseconds)
    hour, minute , second =  list(map(int, s.split(":")))
    return timedelta(minutes = minute, seconds = second, microseconds = microsecond)

class Dataset :
    def __init__(self, filename, path_to_data) :
        self.path_to_data = pathlib.Path(path_to_data)
        self.file_ = open(filename, 'r')
        self.data  = json.load(self.file_)

        # Create a dumping download file
        self._downloads = pathlib.Path('./_downloads')
        if not self._downloads.exists() : 
            self._downloads.mkdir()

        # Mediapipe pose
        self.pose =  mp_pose.Pose( static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.3) 

        # Build the dataset
        self.build_dataset()
        


    def build_dataset(self):
        # Create dataset
        dataset = pathlib.Path('dataset')
        if not dataset.exists() :
            dataset.mkdir()

        # Build train
        trainset = dataset / 'train'
        if not trainset.exists() :
            trainset.mkdir()
        self.build_subset(trainset, 'train')

        # Build dev
        devset   = dataset / 'dev'
        if not devset.exists() :
            devset.mkdir()
        self.build_subset(devset, 'dev')

        # Build test
        testset  = dataset / 'test'
        if not testset.exists() :
            testset.mkdir()
        self.build_subset(testset, 'test')

    def for_speaker(self, speaker, subset, name_subset):
        link_set = self.data[name_subset][speaker]
        # Each link
        for ind, link in enumerate(link_set) :
            # Video Filename
            filename = f'{subset.name}_{speaker}_link{ind}'
            VIDEO_PATH = os.path.join(self._downloads, filename + '.webm')
            download_video(link, VIDEO_PATH)

            # ---- Get the intervals of the video ---- # 
            intervals = link_set[link]

            for interval_id in intervals :
                # interval ID
                timecodes = intervals[interval_id]
                # start of the window
                start_time = timecodes['start_time']
                # end of the window
                end_time = timecodes['end_time']
                # clip the window
                frame_reader = VideoClipReader(VIDEO_PATH,  get_timedelta(start_time), get_timedelta(end_time)) # src : https://stonesoup.readthedocs.io/en/v0.1b5/auto_demos/Video_Processing.html
                NUM_FRAMES = len(list(frame_reader.clip.iter_frames()))
                # ---- t the intervals of the video ---- # 
                fig, ax = plt.subplots(num="VideoClipReader output")
                pose = []
                # for timestamp, frame in frame_reader:
                with tqdm(range(NUM_FRAMES), ncols = 100, desc ="Extraction\t") as pbar :
                    for timestamp, frame in frame_reader :
                        # Read the frame pixels
                        pixels = copy(frame.pixels)
                        # Plot output
                        image = cv.resize(np.array(pixels), (640, 480))
                        results = self.pose.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                        pbar.update(1)
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        # Plot pose world landmarks.
                        # mp_drawing.plot_landmarks(
                        #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                        cv.imshow('MediaPipe Face Detection', cv.flip(image, 1))
                        if cv.waitKey(5) & 0xFF == 27:
                          break
                break
            break

        
    def build_subset(self, subset, name_subset) :
        # create subsets
        pose_2d = subset / "pose_2d"
        pose_3d = subset / "pose_3d"
        # Each speaker
        self.for_speaker("conan", subset, name_subset)

dataset = Dataset("./metadata.json", ".")
