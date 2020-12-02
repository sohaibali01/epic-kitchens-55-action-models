#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from model_loader import load_checkpoint, make_model

if __name__ == "__main__":
    batch_size = 1
    segment_count = 8
    snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
    snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
    height, width = 224, 224

    tsm = load_checkpoint('TSM_arch=resnet50_modality=RGB_segments=8-cfc93918.pth.tar')
    tsn = load_checkpoint('TSN_arch=resnet50_modality=RGB_segments=8-3ecf904f.pth.tar')
    
    inputs = torch.randn(
        [batch_size, segment_count, snippet_length, snippet_channels, height, width]
    )
    # The segment and snippet length and channel dimensions are collapsed into the channel
    # dimension
    # Input shape: N x TC x H x W
    inputs = inputs.reshape((batch_size, -1, height, width))
    for model in [tsn, tsm]:
        # You can get features out of the models
        features = model.features(inputs)
        # and then classify those features
        verb_logits, noun_logits = model.logits(features)
        print(inputs.shape, features.shape, verb_logits.shape, noun_logits.shape)

        # or just call the object to classify inputs in a single forward pass
        #verb_logits, noun_logits = model(inputs)
        #print(verb_logits.shape, noun_logits.shape)
