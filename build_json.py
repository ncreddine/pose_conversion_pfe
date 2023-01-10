import numpy as np
import pandas as pd
from datetime import time
from datetime import timedelta
import json
import tqdm

#i convert date time into time object
def get_time(isoformat):
    try :
        s, microsecond = isoformat.split(".")
    except :
        s = isoformat
        microsecond = 0
    microsecond = min( int(microsecond) , time.max.microsecond)
    hour, minute , second =  list(map(int, s.split(":")))
    return time(hour, minute, second, microsecond).isoformat()

def get_deltatime(isoformat):
    isoformat = str(isoformat)
    try :
        seconds, microseconds = isoformat.split(".")
        seconds = int(seconds)
    except :
        seconds = int(isoformat)
        microsecond = 0
    microseconds = min( int(microseconds) , time.max.microsecond)
    minutes = seconds // 60 ; seconds = seconds % 60
    return time(minute = minutes, second = seconds, microsecond = microseconds).isoformat()
    

## Read csv file 

# ind_dev   = df.index[df['dataset'] == 'dev'  ].to_list()
# ind_test  = df.index[df['dataset'] == 'test' ].to_list()

SPEAKERS_  = ["almaram", "angelica", "chemistry", "conan", "ellen", "jon", "oliver", "rock", "seth", "shelly", "maher", "huckabee", "fallon", "lec_cosmic", "colbert", "corden", "lec_evol", "minhaj", "bee", "lec_law", "ytch_dating", "lec_hist", "ytch_charisma", "ytch_prof", "ferguson", "noah"]

def sort_to_json(df):
    DATASET = {}
    for _set in ['train', 'test', 'dev'] :
        cond_set = df['dataset'] == _set
        # 
        speaker_dict = {}
        for speaker in SPEAKERS_ :
            # 
            sub_df = df[cond_set & (df['speaker'] == speaker)]
            youtube_links = np.unique(sub_df['video_link'])
            # get video links
            links_dict = {}
            for link in youtube_links :
                subsub_df = sub_df[sub_df['video_link'] == link]
                #
                start_time = subsub_df['start_time'].apply(lambda x : get_time(x.split(' ')[1]))
                end_time   = subsub_df['end_time'].apply(lambda x :   get_time(x.split(' ')[1]))
                delta_time = subsub_df['delta_time'].apply(lambda x : x)
                interval_id = subsub_df['interval_id']
                # timecodes  = list(zip( interval_id , zip(start_time, end_time,delta_time)))
                timecodes  = { id_ : { 'start_time' : st_ , 'end_time' : ed_ , 'delta_time' : dt_ } for id_, st_, ed_, dt_ in zip(interval_id, start_time, end_time,delta_time) }
                links_dict.update({link : timecodes})
                # 
            speaker_dict.update({ speaker : links_dict })
        DATASET.update({ _set : speaker_dict })
    my_data_json = json.dumps(DATASET, indent  = 4)
    with open('data.json', 'w') as f :
        f.write(my_data_json)
        

class Dataset: 
    def __init__(self, filename) :
        intervals = pd.read_csv("./cmu_intervals_df.csv")

        # varnames = intervals.head(0)
        # for x in varnames :
        #     exec("self." + x  + " = []")
        # self.delta_time
        # self.end_time
        # self.interval_id
        # self.speaker
        # self.start_time
        # self.video_fn
        # self.video_link

        sort_to_json(intervals)



dataset = Dataset("./cmu_intervals_df.csv")
