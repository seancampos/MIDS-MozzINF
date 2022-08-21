# prefect
from prefect import flow, task, get_run_logger
from prefect_dask.task_runners import DaskTaskRunner

# python common
import os
import argparse

# local libraries
import sys
sys.path.append("../../")

from lib.util import active_BALD
from lib.util_dashboard import write_audio_for_plot, write_video_for_dash
from lib.mids_pytorch_model import Model, get_wav_for_path_pipeline, plot_mids_MI
# python universe
import numpy as np
import matplotlib.pyplot as plt
# nn
import torch
import torch.nn as nn


#@task(name="Get wav from path")
def _get_wav_for_path_pipeline(path, sr=8000):
    return get_wav_for_path_pipeline(path, sr)


@task(name="predict on frames")
def _predict_on_frames(signal, model, device, step_size, n_hop, batch_size):
    # padding is the difference between the win size and step size
    # The first window is silence prepended to the step size to fill the WindowsError
    # then the window slides by the step amount until the last frame is the step followed by
    # silence to fill the window
    pad_amt = (win_size - step_size) * n_hop
    pad_l = torch.zeros(1, pad_amt) + (0.1**0.5) * torch.randn(1, pad_amt)
    pad_r = torch.zeros(1, pad_amt) + (0.1**0.5) * torch.randn(1, pad_amt)
    padded_stepped_signal = torch.cat([pad_l, signal, pad_r], dim=1).unfold(
        1, win_size * n_hop, step_size * n_hop).transpose(0, 1).to(device)  # b, 1, s

    softmax = nn.Softmax(dim=1)

    prediction_list = []
    spectrogram_list = []
    with torch.no_grad():
        for batch_signals in torch.split(padded_stepped_signal, batch_size, 0):
            predictions = model(batch_signals)
            predction_probabilities = softmax(
                predictions['prediction']).cpu().detach()
            prediction_list.append(predction_probabilities)
            spectrogram_list.append(
                predictions['spectrogram'].cpu().detach().numpy())

    prediction_tensor = torch.cat(prediction_list)

    # align the predictions according to the sliding window so that
    # each step has all of it's prediciton windows stacked
    stacked_predictions = []
    prediction_length = signal.unfold(
        1, win_size * n_hop, step_size * n_hop).shape[1]
    for i in range(win_size // step_size):
        stacked_predictions.append(prediction_tensor[i:i + prediction_length])
    
    #get spectrograms w/o padding
    list_offset = (len(spectrogram_list)-prediction_length) // 2
    spectrograms = np.concatenate(spectrogram_list[list_offset:list_offset + prediction_length])

    return torch.stack(stacked_predictions, dim=-3).numpy(), spectrograms


#@task(name="build timestamp list")
def _build_timestmap_list(mean_predictions, G_X, U_X, time_to_sample, det_threshold):
    """Use the predictions to build an array of contiguous timestamps where the
    probability of detection is above threshold"""
    
    # find where the average 2nd element (positive score) is > threshold
    condition = mean_predictions[:, 1] > det_threshold
    preds_list = []
    for start, stop in _contiguous_regions(condition):
        # start and stop are frame indexes
        # so multiply by n_hop and step_size samples
        # then div by sample rate to get seconds
        preds_list.append([str(start * time_to_sample), str(stop * time_to_sample),
                           "{:.4f}".format(
                               np.mean(mean_predictions[start:stop][:, 1]))
                           + " PE: " +
                           "{:.4f}".format(np.mean(G_X[start:stop]))
                           + " MI: " + "{:.4f}".format(np.mean(U_X[start:stop]))])

    return preds_list

def _iterate_audiofiles(rootFolderPath, audio_format):
    """Generator that yields the path and filename for each audiofile matching
    the file type"""
    for root, dirs, files in os.walk(rootFolderPath):
        for filename in files:
            if filename.endswith(audio_format):
                yield root, filename


def _contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx

@task(name="write audio")
def _write_audio_for_plot(text_output_filename, signal, output_filename, root_out, sr):
    return write_audio_for_plot(text_output_filename, signal, output_filename, root_out, sr)

@task(name="write video")
def _write_video_for_dash(plot_filename, audio_output_filename, audio_length, root_out, output_filename):
    write_video_for_dash(plot_filename, audio_output_filename, audio_length, root_out, output_filename)

@task(name="process sample")
def _process_sample(root, filename, model, device, rootFolderPath, win_size, step_size, n_hop, sr, batch_size, det_threshold, file_suffix):
    logger = get_run_logger()
    
    signal, signal_length = _get_wav_for_path_pipeline(
        [os.path.join(root, filename)], sr=sr)
    if signal_length < (n_hop * win_size) / sr:
        logger.info(
            f"{filename} too short. {signal_length} < {(n_hop * win_size) / sr}")
        return
    else:
        logger.info(f"Read {filename}.  Signal Length: {signal_length}")

    predictions, spectrograms = _predict_on_frames(
        signal, model, device, step_size, n_hop, batch_size)

    frame_count = signal.unfold(1, win_size * n_hop, step_size * n_hop).shape[1]
    G_X, U_X, _ = active_BALD(np.log(predictions), frame_count, 2)
    mean_predictions = np.mean(predictions, axis=0)

    print(mean_predictions)
    
    timestamp_list = _build_timestmap_list(mean_predictions, G_X, U_X, (n_hop * step_size / sr), det_threshold)

    # file names and output directories
    if dir_out:
        root_out = root.replace(rootFolderPath, dir_out)
    else:
        root_out = root
    if not os.path.exists(root_out):
        os.makedirs(root_out)
    output_filename = os.path.splitext(filename)[0]

    text_output_filename = os.path.join(root_out, output_filename) + file_suffix
    #  save text output
    np.savetxt(text_output_filename, timestamp_list, fmt='%s', delimiter='\t')
    
    if to_dash:
        audio_output_filename, audio_length, has_mosquito = _write_audio_for_plot(text_output_filename, signal, output_filename, root_out, sr)
        if has_mosquito:
            plot_filename = plot_mids_MI(spectrograms, mean_predictions[:,1], U_X, det_threshold, root_out, output_filename)
            _write_video_for_dash(plot_filename, audio_output_filename, audio_length, root_out, output_filename)


@flow(name="MED Inference",
      task_runner=DaskTaskRunner())
def write_output(rootFolderPath, audio_format, dir_out=None, det_threshold=0.5, feat_type='stft',
                 n_fft=1024, n_feat=128, win_size=30, step_size=30, n_hop=128, sr=8000,
                 norm_per_sample=True, debug=False, to_dash=False, batch_size=16):
    '''dir_out = None if we want to save files in the same folder that we read from.
       det_threshold=0.5 determines the threshold above which an event is classified as positive. See detect_timestamps for
       a more nuanced discussion on thresholding and what we wish to save upon running the algorithm.'''

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else torch.device("cpu"))
    model = Model('convnext_base_384_in22ft1k',
                  image_size=384, NFFT=n_fft, n_hop=n_hop)
    checkpoint = torch.load('/models/model_e1_2022_04_07_11_52_08.pth')
        #'/models/pytorch/model_presentation_draft_2022_04_07_11_52_08.pth')

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    model_name = 'mids_v4'

    logger = get_run_logger()

    for root, filename in _iterate_audiofiles(rootFolderPath, audio_format):
        file_suffix = f'_win_{win_size}_step_{step_size}_{model_name}_{det_threshold}.txt'
        _process_sample(root, filename, model, device, rootFolderPath, 
            win_size, step_size, n_hop, sr, 
            batch_size, det_threshold, file_suffix)
        

if __name__ == "__main__":
    # print(my_flow('./sample_files/r2022-07-26_00.00.00.410__v6.aac_u2022-07-26_06.00.24.521642.aac'))
    # print(my_flow('./sample_files'))
    parser = argparse.ArgumentParser(description="""
    This function writes the predictions of the model.
    """)
    parser.add_argument(
        "rootFolderPath", help="Source destination of audio files. Can be a parent directory.")
    parser.add_argument(
        "audio_format", help="Any file format supported by librosa load.")
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
    audio_format = args.audio_format
    dir_out = args.dir_out
    to_dash = args.to_dash
    win_size = args.win_size
    step_size = args.step_size
    norm_per_sample = args.norm
    det_threshold = args.threshold

    write_output(rootFolderPath, audio_format, dir_out=dir_out, norm_per_sample=norm_per_sample,
                 win_size=win_size, step_size=step_size, to_dash=to_dash, det_threshold=det_threshold)

# python med.py --dir_out /data/output --win_size=360 --step_size=120 --to_dash True /data .aac