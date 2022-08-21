import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchaudio
import librosa

import timm
from nnAudio import features


def get_wav_for_path_pipeline(path_names, sr):
    signal_length = 0
    effects = [["remix", "1"]]
    if sr:
        effects.extend([
          # ["bandpass", f"400",f"1000"],
          # ["rate", f'{sr}'],
          ['gain', '-n'],
          ["highpass", "200"]
        ])
    for path in path_names:
        signal, rate = librosa.load(path, sr=sr)
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(torch.tensor(signal).expand([2, -1]), sample_rate=rate, effects=effects)
        f = waveform[0]
        mu = torch.std_mean(f)[1]
        st = torch.std_mean(f)[0]
        # clip amplitudes
        signal = torch.clamp(f, min=mu-st*3, max=mu+st*3).unsqueeze(0)
        signal_length += len(signal[0]) / sr
    return signal, signal_length


class Normalization():
    """This class is for normalizing the spectrograms batch by batch. The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected. In this paper, we found that 'imagewise' normalization works better than 'framewise'"""
    def __init__(self, mode='framewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                output = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                output[torch.isnan(output)]=0 # Making nan to 0
                return output
        elif mode == 'imagewise':
            def normalize(x):
                size = x.shape
                x_max = x.reshape(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                x_min = x.reshape(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1) # Make it broadcastable
                x_min = x_min.unsqueeze(1) # Make it broadcastable 
                return (x-x_min)/(x_max-x_min)
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def __call__(self, x):
        return self.normalize(x)



# +
def pcen(x, eps=1e-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
    frames = x.split(1, -2)
    m_frames = []
    last_state = None
    for frame in frames:
        if last_state is None:
            last_state = s * frame
            m_frames.append(last_state)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add_(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    if training:
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_


class PCENTransform(nn.Module):

    def __init__(self, eps=1e-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):
        super().__init__()
        if trainable:
            self.log_s = nn.Parameter(torch.log(torch.Tensor([s])))
            self.log_alpha = nn.Parameter(torch.log(torch.Tensor([alpha])))
            self.log_delta = nn.Parameter(torch.log(torch.Tensor([delta])))
            self.log_r = nn.Parameter(torch.log(torch.Tensor([r])))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable

    def forward(self, x):
#         x = x.permute((0,2,1)).squeeze(dim=1)
        if self.trainable:
            x = pcen(x, self.eps, torch.exp(self.log_s), torch.exp(self.log_alpha), torch.exp(self.log_delta), torch.exp(self.log_r), self.training and self.trainable)
        else:
            x = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable)
#         x = x.unsqueeze(dim=1).permute((0,1,3,2))
        return x


# -

def plot_mids_MI(X_CNN, y, MI, p_threshold, root_out, filename, out_format='.png'):
    '''Produce plot of all mosquito detected above a p_threshold. Supply Mutual Information values MI, feature inputs 
    X_CNN, and predictions y (1D array of 0/1s). Plot to be displayed on dashboard either via svg or as part of a
    video (looped png) with audio generated for this visual presentation.
    
    `out_format`: .png, or .svg
    
    '''
    pos_pred_idx = np.where(y>p_threshold)[0]

    fig, axs = plt.subplots(2, sharex=True, figsize=(10,5), gridspec_kw={
           'width_ratios': [1],
           'height_ratios': [2,1]})
    # x_lims = mdates.date2num(T)
    # date_format = mdates.DateFormatter('%M:%S')
    # axs[0].xaxis_date()
    # axs[0].xaxis.set_major_formatter(date_format)
    
    axs[0].set_ylabel('Frequency (kHz)')
    
    axs[0].imshow(np.hstack(X_CNN.squeeze()[pos_pred_idx]), aspect='auto', origin='lower',
                  extent = [0, len(pos_pred_idx), 0, 4], interpolation=None)
    axs[1].plot(y[pos_pred_idx], label='Probability of mosquito')
    axs[1].plot(MI[pos_pred_idx], '--', label='Uncertainty of prediction')
    axs[1].set_ylim([0., 1.02])
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              frameon=False, ncol=2)
    # axs[1].xaxis.set_major_formatter(date_format)
    
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[0].yaxis.set_label_position("right")
    axs[0].yaxis.tick_right()
    # axs[1].set_xlim([t[0], t[-1]])
    axs[1].grid(which='major')
    # axs[1].set_xlabel('Time (mm:ss)')
    axs[1].xaxis.get_ticklocs(minor=True)
    axs[1].yaxis.get_ticklocs(minor=True)
    axs[1].minorticks_on()
    labels = axs[1].get_xticklabels()
    # remove the first and the last labels
    labels[0] = ""
    # set these new labels
    axs[1].set_xticklabels(labels)
#     

    plt.subplots_adjust(top=0.985,
    bottom=0.1,
    left=0.0,
    right=0.945,
    hspace=0.065,
    wspace=0.2)
#     plt.show()
    output_filename = os.path.join(root_out, filename) + out_format
    plt.savefig(output_filename, transparent=False)
    plt.close(plt.gcf()) # May be better to re-write to not use plt API
# fig.autofmt_xdate()
    return output_filename


# +
# Subclass the pretrained model and make it a binary classification

class Model4(nn.Module):
    def __init__(self, model_name, image_size=224, NFFT=512, n_hop=64, sr=8000):
        super().__init__()
        # num_classes=0 removes the pretrained head
        self.backbone = timm.create_model(model_name,
                        pretrained=False, num_classes=2, in_chans=1, 
                        drop_path_rate=0.05, 
                        drop_rate=0.05)
        #####  This section is model specific
        #### It freezes some fo the layers by name
        #### you'll have to inspect the model to see the names
        self.spec_layer = features.STFT(n_fft=NFFT, freq_bins=None, hop_length=n_hop,
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                          fmin=300, fmax=1600, sr=sr, output_format="Magnitude", trainable=True,
                                       verbose=False)
        #### end layer freezing        
        self.sizer = T.Resize((image_size,image_size))
        self.norm_layer = Normalization(mode='framewise')
        self.pcen_layer = PCENTransform(eps=1e-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True)
        
    def forward(self, x):
        # first compute spectrogram
        spec = self.spec_layer(x)  # (B, F, T)
        # normalize
#         spec = spec.transpose(1,2) # (B, T, F)
        spec = self.pcen_layer(spec)
        spec = self.norm_layer(spec)
        
        # then size for CNN model
        # and create a channel
        spec = self.sizer(spec)
        x = spec.unsqueeze(1)
        # then repeat channels
        pred = self.backbone(x)
                
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output


# +
# Subclass the pretrained model and make it a binary classification

class Model3(nn.Module):
    def __init__(self, model_name, image_size=224, NFFT=512, n_hop=64, sr=8000):
        super().__init__()
        # num_classes=0 removes the pretrained head
        self.backbone = timm.create_model(model_name,
                        pretrained=True, num_classes=2, in_chans=3, 
                        drop_path_rate=0.2, global_pool='avgmax',
                        drop_rate=0.2)
        #####  This section is model specific
        #### It freezes some fo the layers by name
        #### you'll have to inspect the model to see the names
        for name, param in self.backbone.named_parameters():
            if param.requires_grad and 'head' not in name \
                and not name.startswith('norm') \
                and 'stages.3' not in name and 'layers.3' not in name \
                and 'blocks.26' not in name and 'blocks.26' not in name \
                and 'blocks.24' not in name and 'blocks.25' not in name \
                and 'blocks.22' not in name and 'blocks.23' not in name \
                and 'blocks.20' not in name and 'blocks.21' not in name \
                and 'blocks.22' not in name and 'blocks.23' not in name \
                and 'blocks.19' not in name and 'blocks.18' not in name \
                and 'blocks.17' not in name and 'blocks.5.' not in name:
                param.requires_grad = False
        #### end layer freezing
        self.spec_layer = features.STFT(n_fft=NFFT, freq_bins=None, hop_length=n_hop,
                              window='hann', freq_scale='no', center=True, pad_mode='reflect',
                          fmin=300, fmax=4000, sr=sr, output_format="Magnitude", trainable=True)
        
        self.mel_layer = features.MelSpectrogram(n_fft=NFFT, n_mels=128, hop_length=n_hop,
                                window='hann',  center=True, pad_mode='reflect',
                          sr=sr,  trainable_mel=True, trainable_STFT=True)
        self.vqt_layer = features.CQT( sr=sr, hop_length=n_hop, fmin=32.7, fmax=None, n_bins=84, bins_per_octave=12, trainable=True)
#         self.out = nn.Linear(self.backbone.num_features, 1)
        self.sizer = T.Resize((image_size,image_size))
        self.norm_layer = Normalization(mode='framewise')
        self.pcen_layer = PCENTransform(eps=1e-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True)
        
    def forward(self, x):
        # first compute spectrogram
        spec = self.spec_layer(x)  # (B, F, T)
        # normalize
#         spec = spec.transpose(1,2) # (B, T, F)
        spec = self.pcen_layer(spec)
        spec = self.norm_layer(spec)
        
#         if self.training:
#             spec = self.timeMasking(spec)
#             spec = self.freqMasking(spec)

        # then size for CNN model
        # and create a channel
        spec = self.sizer(spec)
        spec = spec.unsqueeze(1)
        
        mel = self.mel_layer(x)
        mel = self.norm_layer(mel)
        mel = self.sizer(mel)
        mel = mel.unsqueeze(1)
        
        vqt = self.vqt_layer(x)
        vqt = self.norm_layer(vqt)
        vqt = self.sizer(vqt)
        vqt = vqt.unsqueeze(1)
        
        x = torch.cat([spec,mel,vqt],dim=1)
        
        
        # then repeat channels
        pred = self.backbone(x)
        
#        pred = self.out(x)
        
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output


# +
# Subclass the pretrained model and make it a binary classification

class Model(nn.Module):
    def __init__(self, model_name, image_size=224, NFFT=512, n_hop=64, sr=8000):
        super().__init__()
        self.backbone = timm.create_model(model_name,
                        pretrained=False, num_classes=2, in_chans=1, 
                        drop_path_rate=0.1,
                        drop_rate=0.1)
        self.spec_layer = features.STFT(n_fft=NFFT, freq_bins=None, hop_length=n_hop,
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                          fmin=300, fmax=3000, sr=sr, output_format="Magnitude", trainable=True)
#         self.spec_layer = features.MelSpectrogram(n_fft=config.NFFT, n_mels=128, hop_length=config.n_hop,
#                                 window='hann',  center=True, pad_mode='reflect',
#                           sr=config.rate,  trainable_mel=True, trainable_STFT=True)
#         self.out = nn.Linear(self.backbone.num_features, 1)
        self.sizer = T.Resize((image_size,image_size))
        self.norm_layer = Normalization(mode='framewise')
        self.pcen_layer = PCENTransform(eps=1e-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True)
        
    def forward(self, x):
        # first compute spectrogram
        spec = self.spec_layer(x)  # (B, F, T)
        # normalize
#         spec = spec.transpose(1,2) # (B, T, F)
        spec = self.pcen_layer(spec)
        spec = self.norm_layer(spec)
        
        # then size for CNN model
        # and create a channel
        spec = self.sizer(spec)
        x = spec.unsqueeze(1)
        # then repeat channels
        pred = self.backbone(x)
                
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output


# +
# Subclass the pretrained model and make it a binary classification

class Model1(nn.Module):
    def __init__(self, model_name, image_size=224, NFFT=512, n_hop=64, sr=8000):
        super().__init__()
        self.backbone = timm.create_model(model_name,
                        pretrained=False, num_classes=2, in_chans=1, 
                        drop_path_rate=0.1,
                        drop_rate=0.1)
        self.spec_layer = features.MelSpectrogram(n_fft=NFFT, n_mels=128, hop_length=n_hop,
                                window='hann',  center=True, pad_mode='reflect',
                          sr=sr,  trainable_mel=True, trainable_STFT=True)
#         self.spec_layer = features.MelSpectrogram(n_fft=config.NFFT, n_mels=128, hop_length=config.n_hop,
#                                 window='hann',  center=True, pad_mode='reflect',
#                           sr=config.rate,  trainable_mel=True, trainable_STFT=True)
#         self.out = nn.Linear(self.backbone.num_features, 1)
        self.sizer = T.Resize((image_size,image_size))
        self.norm_layer = Normalization(mode='framewise')
        self.pcen_layer = PCENTransform(eps=1e-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True)
        
    def forward(self, x):
        # first compute spectrogram
        spec = self.spec_layer(x)  # (B, F, T)
        # normalize
#         spec = spec.transpose(1,2) # (B, T, F)
        spec = self.pcen_layer(spec)
        spec = self.norm_layer(spec)
        
        # then size for CNN model
        # and create a channel
        spec = self.sizer(spec)
        x = spec.unsqueeze(1)
        # then repeat channels
        pred = self.backbone(x)
                
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output


# +
# Subclass the pretrained model and make it a binary classification

class Model0(nn.Module):
    def __init__(self, model_name, image_size=224, NFFT=512, n_hop=64, sr=8000):
        super().__init__()
        # num_classes=0 removes the pretrained head
        self.backbone = timm.create_model(model_name,
                        pretrained=False, num_classes=2, in_chans=3, 
                        drop_path_rate=0.2, global_pool='avgmax',
                        drop_rate=0.2)

        self.spec_layer = features.STFT(n_fft=NFFT, freq_bins=None, hop_length=n_hop,
                              window='hann', freq_scale='no', center=True, pad_mode='reflect',
                          fmin=300, fmax=3000, sr=sr, output_format="Magnitude", trainable=True)
        
#         self.mel_layer = features.STFT(n_fft=config.NFFT, freq_bins=None, hop_length=config.n_hop,
#                               window='hann', freq_scale='log', center=True, pad_mode='reflect',
#                           fmin=300, fmax=3000, sr=config.rate, output_format="Magnitude", trainable=True)
        self.mel_layer = features.MelSpectrogram(n_fft=NFFT, n_mels=128, hop_length=n_hop,
                                window='hann',  center=True, pad_mode='reflect',
                          sr=sr,  trainable_mel=True, trainable_STFT=True)
        self.vqt_layer = features.STFT(n_fft=NFFT, freq_bins=None, hop_length=n_hop,
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                          fmin=300, fmax=1800, sr=sr, output_format="Magnitude", trainable=True)
#         self.out = nn.Linear(self.backbone.num_features, 1)
        self.sizer = T.Resize((image_size,image_size))
        self.norm_layer = Normalization(mode='framewise')
        self.pcen_layer = PCENTransform(eps=1e-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True)
        self.pcen_layer2 = PCENTransform(eps=1e-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True)
        
    def forward(self, x):
        # first compute spectrogram
        spec = self.spec_layer(x)  # (B, F, T)
        # normalize
#         spec = spec.transpose(1,2) # (B, T, F)
        spec = self.pcen_layer(spec)
        spec = self.norm_layer(spec)
        
#         if self.training:
#             spec = self.timeMasking(spec)
#             spec = self.freqMasking(spec)

        # then size for CNN model
        # and create a channel
        spec = self.sizer(spec)
        spec = spec.unsqueeze(1)
        
        mel = self.mel_layer(x)
#         mel = self.pcen_layer(mel)
        mel = self.norm_layer(mel)
        mel = self.sizer(mel)
        mel = mel.unsqueeze(1)
        
        vqt = self.vqt_layer(x)
        vqt = self.pcen_layer2(vqt)
        vqt = self.norm_layer(vqt)
        vqt = self.sizer(vqt)
        vqt = vqt.unsqueeze(1)
        
        x = torch.cat([spec,mel,vqt],dim=1)
        
        
        # then repeat channels
        pred = self.backbone(x)
        
#        pred = self.out(x)
        
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output
# -


