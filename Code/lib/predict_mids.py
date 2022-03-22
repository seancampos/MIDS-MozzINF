# +

# I had to find the right version of pytorch with the widget here https://pytorch.org/
#pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# other dependencies
#pip install timm

## nnAudio
#pip install git+https://github.com/KinWaiCheuk/nnAudio.git#subdirectory=Installation

import os
import util
import util_dashboard as util_dash
import numpy as np
import sys

from mids_pytorch_model import Model, get_wav_for_path_pipeline

import torch
import torch.nn as nn

import argparse
import matplotlib.pyplot as plt


# -

def write_output(rootFolderPath, audio_format,  dir_out=None, det_threshold=0.5, n_samples=1, feat_type='stft',n_fft=512, win_size=224, step_size=80,
                 n_hop=512//8, sr=8000, norm_per_sample=True, debug=False, to_dash=False, batch_size=16):

        '''dir_out = None if we want to save files in the same folder that we read from.
           det_threshold=0.5 determines the threshold above which an event is classified as positive. See detect_timestamps for 
           a more nuanced discussion on thresholding and what we wish to save upon running the algorithm.'''
        # rootFolderPath = 'F:\PostdocData\HumBugServer\SemiFieldDataTanzania'
        # audio_format = '.wav'
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device("cpu"))
        
        model = Model('convnext_large_in22ft1k')
        sigmoid = nn.Sigmoid()
#         https://drive.google.com/file/d/12ozdcETHX5cBMrdNMainWCWIp-bQlq6J/view?usp=sharing
        checkpoint = torch.load('../models/model_e28_2022_03_14_05_46_11.pth')
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model_name = 'mids_v1'

        

        mozz_audio_list = []
        
        print('Processing:', rootFolderPath, 'for audio format:', audio_format)

        i_signal = 0
        for root, dirs, files in os.walk(rootFolderPath):
            for filename in files:
                if audio_format in filename:
                    print(root, filename) 
                    i_signal+=1
                    try:            
                        x, x_l = get_wav_for_path_pipeline([os.path.join(root, filename)], sr=sr)
                        if debug:
                            print(filename + ' signal length', x_l)
                        if x_l < (n_hop * win_size)/sr: 
                            print('Signal length too short, skipping:', x_l, filename) 
                        else:
                            X = x[0].unfold(1,win_size*n_hop,step_size*n_hop).transpose(0,1).to(device) # b, 1, s

                            X_batches = torch.split(X,batch_size,0)

                            out = []
                            for i in range(n_samples):
                                preds_batch = []
                                for X_batch in X_batches:
                                    preds = sigmoid(model(X_batch)['prediction']).cpu().detach().numpy()
                                    preds = np.insert(preds,0,1-np.squeeze(preds),axis=1)
                                    preds_batch.append(preds)
                                out.append(np.concatenate(preds_batch))

                            G_X, U_X, _ = util.active_BALD(np.log(out), X, 2)


                            y_to_timestamp = np.repeat(np.mean(out, axis=0), step_size, axis=0)
                            G_X_to_timestamp = np.repeat(G_X, step_size, axis=0)
                            U_X_to_timestamp = np.repeat(U_X, step_size, axis=0)
                            preds_list = util.detect_timestamps_BNN(y_to_timestamp, G_X_to_timestamp, U_X_to_timestamp, 
                                                                det_threshold=det_threshold)   

                            if debug:
                                print(preds_list)
                                for times in preds_list:
                                    mozz_audio_list.append(librosa.load(os.path.join(root, filename), offset=float(times[0]),
                                                                     duration=float(times[1])-float(times[0]), sr=sr)[0])


                            if dir_out:
                                root_out = root.replace(rootFolderPath, dir_out)
                            else:
                                root_out = root
                            print('dir_out', root_out, 'filename', filename)


                            if not os.path.exists(root_out): os.makedirs(root_out)

                            if filename.endswith(audio_format):  
                                output_filename = filename[:-4]  # remove file extension for renaming to other formats.
                            else:
                                output_filename = filename # no file extension present

                            text_output_filename = os.path.join(root_out, output_filename) + '_MIDS_step_' + str(step_size) + '_samples_' + str(n_samples) + '_'+ str(model_name) + '.txt'
                            np.savetxt(text_output_filename, preds_list, fmt='%s', delimiter='\t')

                            if to_dash: 
                                mozz_audio_filename, audio_length, has_mosquito = util_dash.write_audio_for_plot(text_output_filename, root, filename, output_filename, root_out, sr)
                                if has_mosquito:
                                    plot_filename = util_dash.plot_mozz_MI(X_CNN, y_to_timestamp[:,1], U_X_to_timestamp, 0.5, root_out, output_filename)
                                    util_dash.write_video_for_dash(plot_filename, mozz_audio_filename, audio_length, root_out, output_filename)
                    except Exception as e:
                        print("[ERROR] Unable to load {}, {} ".format(os.path.join(root, filename)),e)

        print('Total files of ' + str(audio_format) + ' format processed:', i_signal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This function writes the predictions of the model.
    """)
    parser.add_argument("rootFolderPath", help="Source destination of audio files. Can be a parent directory.")
    parser.add_argument("audio_format", help="Any file format supported by librosa load.")
    parser.add_argument("--dir_out", help="Output directory. If not specified, predictions are output to the same folder as source.")
    parser.add_argument("--to_dash", default=False, type=bool, help="Save predicted audio, video, and corresponding labels to same directory as dictated by dir_out.")
    parser.add_argument("--norm", default=True, help="Normalise feature windows with respect to themsleves.")
    parser.add_argument("--win_size", default=30, type=int, help="Window size.")
    parser.add_argument("--step_size", default=30, type=int, help="Step size.")
    parser.add_argument("--BNN_samples", default=10, type=int, help="Number of MC dropout samples.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")

    # dir_out=None, det_threshold=0.5, n_samples=10, feat_type='log-mel',n_feat=128, win_size=40, step_size=40,
    #              n_hop=512, sr=8000, norm=False, debug=False, to_filter=False

    args = parser.parse_args()

    rootFolderPath = args.rootFolderPath
    audio_format = args.audio_format
    dir_out = args.dir_out
    to_dash = args.to_dash
    win_size = args.win_size
    step_size = args.step_size
    n_samples = args.BNN_samples
    norm_per_sample=args.norm
    batch_size = args.batch_size


    write_output(rootFolderPath, audio_format, dir_out=dir_out, norm_per_sample=norm_per_sample,
                 win_size=win_size, step_size=step_size, to_dash=to_dash, n_samples=n_samples, batch_size=batch_size)
