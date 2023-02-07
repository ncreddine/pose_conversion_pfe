## Dataset

The [Pose, Audio, Transcript, Style (PATS)](https://chahuja.com/pats/) was initially intended for style transfer between different speakers, this database relates different verbal and non-verbal behavior that characterizes a person, such as __speech__, __voice tone__ and __pose__ [1]. You can fill the following [form](https://chahuja.com/pats/download.html) to download it (130 Go).


This dataset is constituted of 22 speakers, with more than 251 hours of video and about 84 000 annotated intervals in total. Since only 2D poses are included in the dataset we needed to extract 3d pose from the detaset videos. The dataset doesn't directly include videos, but they are referenced as  links to download.

The data preparation is divided into two steps :
 - Create a compact `metadata.json` file that regroups all the intevals for each link.
 - Download each video through the provided link and clip it.
 - Apply `mediapipe` hollistic extractor for each video interval to get the 3D pose of the speaker
 
 
 ## Convert to JSON
The first step is to execute the `csv_to_JSON.py` file. This converts the provided `cmu_intervals_df.csv`, where links to each speaker and their `interval_ids` are stored, to a json file more readable and easier to execute for our next step. The following command does what we described above :
```
./csv_to_JSON.py    --intervals  path/to/cmu_intervals_df.csv \
                    --missing    path/to/missing_intervals.h5 \
                    --output     where/to/store/sorted.json
```

## Extract 3D poses

The data we will train our model on are the 2D poses that are included in the dataset, and this to predict the 3D poses from them. Since only the 2D poses are annotated in the dataset, we need to extract 3D poses from the PATS videos. We choose to use mediapipe's [holistic](https://google.github.io/mediapipe/solutions/holistic.html) to extract the 3D poses we need that matchs the PATS annotations. 

To do this you need to execute 'pose.py' in order to download the video, extract the 3D pose from the intervals, and read the correspondant 2D pose in the PATS dataset. Theses data are stored inside folders that are created for each speaker in each test, train and dev set.

## Skeleton adaptation


While performing a holistic's detection using mediapipe, the model uses the other defined modules such as pose, hand, and face, and assemble their outputs into one structure. This is shown by the following diagram from mediapipe's site :
<img src="https://mediapipe.dev/images/mobile/holistic_pipeline_example.jpg"  width="60%" height="60%">

In fact, this step causes an offset and scale difference between hands and body. First the offset is between the hand root and wrist points, since the hand detector performs a depth estimation that is centerd at 0. This could be fixed by translating the hand origin to the wrist. The second difference is in the depth scale between the hand and the body, since the two estimations aren't related, it arent at the same scale, we rescale the hand.


---
```
[1] Chaitanya Ahuja, Dong Won Lee, Yukiko I. Nakano, and Louis-Philippe Morency. "Style Transfer for Co-Speech 
    Gesture Animation: A Multi-Speaker Conditional-Mixture Approach." ECCV 2020
```
