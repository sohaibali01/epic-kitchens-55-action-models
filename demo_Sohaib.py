#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from model_loader import load_checkpoint, make_model
from PIL import Image

import torch
import subprocess as sp
import numpy as np

FFMPEG_BIN = "ffmpeg.exe"
videoPath = "fp1.mp4"
videoWidth = 640
videoHeight = 360
width = 224
height = 224
skipTime = 0.5 # in sec
frameRate = 30  # appx from 29.97
skipLength = round(skipTime*frameRate)
frameIndex = 0
batch_size = 1
segment_count = 8

if __name__ == "__main__":
    
    command = [FFMPEG_BIN,
            '-i', videoPath,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec','rawvideo', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**6) # 1GB
    tsm = load_checkpoint('TSM_arch=resnet50_modality=RGB_segments=8-cfc93918.pth.tar')
    tsn = load_checkpoint('TSN_arch=resnet50_modality=RGB_segments=8-3ecf904f.pth.tar')
  
    out = None
    numSegments = 0
    while(True):
        raw_image = pipe.stdout.read(videoWidth*videoHeight*3)
        if ((frameIndex % skipLength) == 0):
            # transform the byte read into a numpy array
            image =  np.frombuffer(raw_image, dtype='uint8')
            image = image.reshape((videoHeight,videoWidth,3))

            new_im = Image.fromarray(image)
            #new_im.show()
            minDim = min(videoHeight,videoWidth)

            # Crop the center of the image
            left = (videoWidth - minDim)/2
            top = (videoHeight - minDim)/2
            right = (videoWidth + minDim)/2
            bottom = (videoHeight + minDim)/2
            new_im = new_im.crop((left, top, right, bottom))
            new_im = new_im.resize((height,width))
            new_im.save(str(numSegments)+".jpg")
            npImage = np.array(new_im)
            npImage = npImage.astype(np.float32) / 255.0
            outnp = np.rollaxis(npImage, 2, 0)
            if out is None:
                out = outnp[None, None, None,:,:,:]
            else:
                out = np.concatenate((out, outnp[None, None, None,:,:,:]), 1)

            numSegments = numSegments + 1
            #new_im.show()
        
        # throw away the data in the pipe's buffer.
        pipe.stdout.flush()
        frameIndex=frameIndex+1

        if (numSegments>=segment_count):
            frames = torch.from_numpy(out)
            inputs = frames.reshape((batch_size, -1, height, width))
            for model in [tsn, tsm]:
                # You can get features out of the models
                #features = model.features(inputs)
                # and then classify those features
                #verb_logits, noun_logits = model.logits(features)
                #print(verb_logits.numpy(), noun_logits.numpy())

                # or just call the object to classify inputs in a single forward pass
                verb_logits, noun_logits = model(inputs)
                verbs = verb_logits.detach().numpy()
                nouns =  noun_logits.detach().numpy()
                print(np.argmax(verbs), np.argmax(nouns))
            out = None
            numSegments = 0
 