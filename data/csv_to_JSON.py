from utils import get_time, subset
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import json
import h5py

def filter_timecodes(df, link, missing) :
    """ Sets all the intervals for one link (video) in the speaker dataframe
        @param  : speaker dataframe
                : link to search
                : missing intervals to check
        ---------
        @return : dict of interval 

            {interval_id : <str>} : {
                      'start_time' : <datatime.time object>,
                       'end_time'  : <datatime.time object>,
                      'delta_time' : <float>
                      },
    """
    sub_df   = subset(df,'video_link',link)
    start_time  = sub_df['start_time'].apply(lambda x : get_time(x.split(' ')[1]))
    end_time    = sub_df[ 'end_time' ].apply(lambda x : get_time(x.split(' ')[1]))
    delta_time  = sub_df['delta_time'].apply(lambda x : x)
    interval_id = sub_df['interval_id']
    # -- timecodes
    timecodes  = { id_ : { 'start_time' : st_ , 'end_time' : ed_ , 'delta_time' : dt_ } 
            for id_, st_, ed_, dt_ in zip(interval_id, start_time, end_time,delta_time) if id_ not in missing }
    return timecodes

def sort_to_json(path_to_intervals, path_to_missing_intervals, output_file ):
    """ Compacts the the dataframe into a more readable file
        @param  : path to `cmu_intervals_df.csv` file
                : path to `missing_intervals.h5` file
                : output file, default {WORKING_DIR}/metadata.json
        ------------
        @return : None
    """
    df = pd.read_csv(path_to_intervals) # dataframe
    missing = list(h5py.File(path_to_missing_intervals)['intervals']) # missing intervals

    DATASET = {}
    # --> set
    with tqdm(total = 3,  bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = True) as pbar :
        for dataset_ in ['train', 'test', 'dev'] :
            pbar.set_description(dataset_)
            dataset_df = subset(df, 'dataset', dataset_)
            speakers = np.unique(dataset_df['speaker'])
            speaker_dict = {}
            # --> speaker
            with tqdm(total = len(speakers),  bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = False) as pbar2 :
                for _speaker in speakers :
                    pbar2.set_description(_speaker)
                    speaker_df = subset(dataset_df ,'speaker', _speaker)
                    links_df = np.unique(dataset_df['video_link']) ; 
                    links_dict = {}
                    # -->  links
                    with tqdm(total = len(links_df),  bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = False, desc = 'links' ) as pbar3 :
                        for video_link in links_df :
                            timecodes = filter_timecodes(speaker_df, video_link, missing)
                            if len(timecodes) != 0 : links_dict.update({video_link : timecodes})
                            pbar3.update(1)
                    if len(links_dict) != 0 : speaker_dict.update({ _speaker : links_dict })
                    pbar2.update(1)
            if len(speaker_dict) != 0 : DATASET.update({ dataset_ : speaker_dict })
            pbar.update(1)
        # Save as JSON
        my_data_json = json.dumps(DATASET, indent  = 4)
        with open(output_file, 'w') as f :
            f.write(my_data_json)
        

def parse_args():
    parser = argparse.ArgumentParser(description = 'sort the data')

    parser.add_argument('-i', '--intervals', default = './cmu_intervals_df.csv', type=str, metavar='NAME', help = 'path to `cmu_intervals_df.csv` file')
    parser.add_argument('-m', '--missing', default = './missing_intervals.h5', type=str, metavar='NAME', help = 'path to `missing_intervals.h5` file')
    parser.add_argument('-o', '--output', default = './sorted.json', type=str, metavar='NAME', help = 'extraction path')

    args = parser.parse_args() 

    return args

def main(args):
    sort_to_json(args.intervals, args.missing, args.output)

if __name__ == '__main__':
    main(parse_args())

