# prefect
# from prefect import flow, task, get_run_logger
# from prefect_dask.task_runners import DaskTaskRunner

# python common
import os
import argparse
import logging

# local libraries
import sys
sys.path.append("../../")

from lib.util import active_BALD
from lib.util_dashboard import write_audio_for_plot, write_video_for_dash
from lib.mids_pytorch_model import Model, get_wav_for_path_pipeline, plot_mids_MI
# python universe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.cli import tqdm
# nn
import torch
import torch.nn as nn

def get_run_logger():
    return logging.getLogger()


#@task(name="Get wav from path")
def _get_wav_for_path_pipeline(path, sr=8000):
    return get_wav_for_path_pipeline(path, sr)



# @task(name="write audio")
def _write_audio_for_plot(text_output_filename, signal, output_filename, root_out, sr):
    return write_audio_for_plot(text_output_filename, signal, output_filename, root_out, sr)

# @task(name="write video")
def _write_video_for_dash(plot_filename, audio_output_filename, audio_length, root_out, output_filename):
    write_video_for_dash(plot_filename, audio_output_filename, audio_length, root_out, output_filename)


# @flow(name="MED Inference",
    #   task_runner=DaskTaskRunner())
def write_output(rootFolderPath, csv_filename, dir_out=None, det_threshold=0.5, feat_type='stft',
                 n_fft=1024, n_feat=128, win_size=30, step_size=30, n_hop=128, sr=8000,
                 norm_per_sample=True, debug=False, to_dash=False, batch_size=16):
    '''dir_out = None if we want to save files in the same folder that we read from.
       det_threshold=0.5 determines the threshold above which an event is classified as positive. See detect_timestamps for
       a more nuanced discussion on thresholding and what we wish to save upon running the algorithm.'''

    model_name = 'mids_v4'

    logger = get_run_logger()

    files_df = pd.read_csv(csv_filename, low_memory=False)

    files_df['done'] = 0

    while files_df['done'].sum() < len(files_df):
        for row_index, file_row in tqdm(files_df.iterrows(), total=len(files_df)):
            if file_row['done'] == 0:
                filename = file_row['filename']
                root = os.path.join(rootFolderPath, file_row['path'])

                # file names and output directories
                if dir_out:
                    root_out = os.path.join(dir_out, file_row['path'])
                else:
                    root_out = os.path.join(rootFolderPath, file_row['path'])
                output_filename = os.path.splitext(filename)[0]

                file_suffix = f'_win_{win_size}_step_{step_size}_{model_name}_{det_threshold}.txt'
                text_output_filename = os.path.join(root_out, output_filename) + file_suffix
                audio_output_filename = os.path.join(root_out, output_filename) + '_mozz_pred.wav'
                plot_filename = os.path.join(root_out, output_filename) + '.png'
                video_output_filename = os.path.join(root_out, output_filename) + '_mozz_pred.mp4'

                if os.path.exists(text_output_filename) and os.path.exists(audio_output_filename)\
                         and os.path.exists(plot_filename) and not os.path.exists(video_output_filename):

                    _, signal_length = _get_wav_for_path_pipeline([audio_output_filename], sr=sr)

                    if signal_length < (n_hop * win_size) / sr:
                        logger.info(f"{filename} too short. {signal_length} < {(n_hop * win_size) / sr}")
                        files_df.loc[row_index, 'done'] = 1
                        continue
                    else:
                        logger.info(f"Read {filename}.  Signal Length: {signal_length}")

                    audio_length = signal_length / sr
                    _write_video_for_dash(plot_filename, audio_output_filename, audio_length, root_out, output_filename)
                
                    files_df.loc[row_index, 'done'] = 1
                    

if __name__ == "__main__":
    # print(my_flow('./sample_files/r2022-07-26_00.00.00.410__v6.aac_u2022-07-26_06.00.24.521642.aac'))
    # print(my_flow('./sample_files'))
    parser = argparse.ArgumentParser(description="""
    This function writes the predictions of the model.
    """)
    parser.add_argument(
        "rootFolderPath", help="Source destination of audio files. Can be a parent directory.")
    parser.add_argument(
        "csv_filename", help="Location of CSV file with paths")
    parser.add_argument(
        "--dir_out", help="Output directory. If not specified, predictions are output to the same folder as source.")
    parser.add_argument("--to_dash", default=False, type=bool,
                        help="Save predicted audio, video, and corresponding labels to same directory as dictated by dir_out.")
    parser.add_argument("--norm", default=True,
                        help="Normalise feature windows with respect to themsleves.")
    parser.add_argument("--win_size", default=30,
                        type=int, help="Window size.")
    parser.add_argument("--step_size", default=30, type=int, help="Step size.")
    parser.add_argument("--threshold", default=0.5, type=float,
                        help="Detection threshold above which samples classified positive.")

    # dir_out=None, det_threshold=0.5, n_samples=10, feat_type='log-mel',n_feat=128, win_size=40, step_size=40,
    #              n_hop=512, sr=8000, norm=False, debug=False, to_filter=False

    args = parser.parse_args()

    rootFolderPath = args.rootFolderPath
    csv_filename = args.csv_filename
    dir_out = args.dir_out
    to_dash = args.to_dash
    win_size = args.win_size
    step_size = args.step_size
    norm_per_sample = args.norm
    det_threshold = args.threshold

    write_output(rootFolderPath, csv_filename, dir_out=dir_out, norm_per_sample=norm_per_sample,
                 win_size=win_size, step_size=step_size, to_dash=to_dash, det_threshold=det_threshold)

# python med.py --dir_out /data/output --win_size=360 --step_size=120 --to_dash True /data .aac