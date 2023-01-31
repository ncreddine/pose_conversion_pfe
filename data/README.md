## Dataset

The [Pose, Audio, Transcript, Style (PATS)](https://chahuja.com/pats/) was initially intended for style transfer between different speakers, this database relates different verbal and non-verbal behavior that characterizes a person, such as __speech__, __voice tone__ and __pose__ [1]. You can fill the following [form](https://chahuja.com/pats/download.html) to download it (130 Go).


This dataset is constituted of 22 speakers, with more than 251 hours of video and about 84 000 annotated intervals in total. Since only 2D poses are included in the dataset we need to extract 3d pose from the detaset videos. The dataset doesn't directly include videos, but they are referenced as  links to download.

The data preparation is divided into to steps :
 - Create a `metadata.json` file that regroups all the intevals for each link.
 - Download each video through the provided link and clip it.
 - Apply `mediapipe` hollistic extractor for each video interval to get the 3D pose of the speaker
 
 
 ## metadata
 
 The file `build_json.py` converts the provided `cmu_intervals_df.csv` file in PATS, to  a`metadata.json`  file. This file is sorted like so :
 
 
 ```
 set (train, test, dev) {
      speaker (22 speakers) {
          link {
              interval_id {
                    "start_time" : 
                    "end_time"   : 
                    "delta_time" : 
              },
          },
      },
 },
 ```
 
---
```
[1] Chaitanya Ahuja, Dong Won Lee, Yukiko I. Nakano, and Louis-Philippe Morency. "Style Transfer for Co-Speech 
    Gesture Animation: A Multi-Speaker Conditional-Mixture Approach." ECCV 2020
```
