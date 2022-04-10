# MIDS Model Inference

Requirements:
Here are the package install commands I used:
```
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install timm
pip install git+https://github.com/KinWaiCheuk/nnAudio.git#subdirectory=Installation
```

Not required, but recommended, I used the Nvidia PyTorch container.  Here's a sample command.  Update the volume location to your file system appropriately.

```
docker run --gpus all -p 8888:8888 -v /YOUR_DIRECTORY:/my_data --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:21.12-py3
```

The model checkpoint is available here:
```
https://drive.google.com/file/d/1fAzdxz_faoDylgjb1ipireVrrRAdMFo8/view?usp=sharing
```

It's currently a large model.  Working to optimize and reduce parameter count in the coming weeks.

I've created a modified copy of the inference with the hyperparamters used during training as default parameters for the `write_output` command.

Example of running the script from the `Code/lib` directory.

`python predict_mids.py ../data/ .wav`

Inference will run on the GPU (but should fall back to the CPU gracefully).  Batch size is currently set to 32, which is pretty conservative, and is only important for long clips.  It will automatically run multiple batches if the wav file is too long and collate the results.
