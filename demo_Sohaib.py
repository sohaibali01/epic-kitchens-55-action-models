#!/usr/bin/env python3
import argparse
import logging
import sys, os
from pathlib import Path
from typing import Any, Dict

from model_loader import load_checkpoint, make_model
from PIL import Image

import torch
import subprocess as sp
import numpy as np

import csv
from collections import defaultdict
import datetime
import time

FFMPEG_BIN = "./data/ffmpeg.exe"
videoPath = "./data/fp3.mp4"
nounCSV = "./data/EPIC_noun_classes.csv"
verbCSV = "./data/EPIC_verb_classes.csv"
GT_Output_CSV = "./data/GT-fp3.csv"
outputCSV = "./data/output.csv"
tsm_model = "./data/TSM_arch=resnet50_modality=RGB_segments=8-cfc93918.pth.tar"
tsn_model = "./data/TSN_arch=resnet50_modality=RGB_segments=8-3ecf904f.pth.tar"
videoWidth = 640
videoHeight = 360
width = 224
height = 224
skipTime = 2/8.0 # in sec
frameRate = 25  # appx 
videoLength = 60 * 11 # in sec
skipLength = round(skipTime*frameRate)
totalFrames = videoLength * frameRate
frameIndex = 0
batch_size = 1
segment_count = 8

def readCsv(fileName):
    columns = defaultdict(list) # each value in each column is appended to a list
    with open(fileName) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value 
                columns[k].append(v) # append the value into the appropriate list
    f.close()                                # based on column name k
    return columns

def getTop5Accuracy():
    cols = readCsv(GT_Output_CSV)
    names = ["TSN( )-Verb", "TSN( )-Noun","TSM( )-Verb", "TSM( )-Noun"]
    numSamples = len(cols["Time"])
    for name in names:
        boolArr = np.zeros(numSamples,dtype=bool)
        for col in [cols[name[0:4]+str(1)+name[5:]],
                    cols[name[0:4]+str(2)+name[5:]],
                    cols[name[0:4]+str(3)+name[5:]],
                    cols[name[0:4]+str(4)+name[5:]],
                    cols[name[0:4]+str(5)+name[5:]] ]:

            currentArr=np.zeros(numSamples,dtype=bool)
            i=0
            for a, b in zip(col, cols["GT-"+str(name[7:])]):
                currentArr[i] = (a==b)
                i=i+1 
            boolArr =  np.logical_or(boolArr, currentArr)
        acc = np.sum(boolArr)/numSamples
        print(name, acc)

                   

if __name__ == "__main__":
    
    getTop5Accuracy()

    nounLabels = readCsv(nounCSV)
    verbLabels = readCsv(verbCSV)

    if os.path.exists(outputCSV):
        os.remove(outputCSV)

    outFile = open(outputCSV, 'w', newline='') 
    writer = csv.writer(outFile)
    writer.writerow(["Time", "TSN(1)-Verb", "TSN(2)-Verb","TSN(3)-Verb","TSN(4)-Verb","TSN(5)-Verb",
                             "TSM(1)-Verb", "TSM(2)-Verb","TSM(3)-Verb","TSM(4)-Verb","TSM(5)-Verb",
                             "TSN(1)-Noun","TSN(2)-Noun","TSN(3)-Noun","TSN(4)-Noun","TSN(5)-Noun",
                             "TSM(1)-Noun","TSM(2)-Noun","TSM(3)-Noun","TSM(4)-Noun","TSM(5)-Noun"])

    command = [FFMPEG_BIN,
            '-i', videoPath,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec','rawvideo', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**6) # 1GB
    
    tsm = load_checkpoint(tsm_model)
    tsn = load_checkpoint(tsn_model)
  
    out = None
    numSegments = 0
    tsnTime=[]
    tsmTime=[]
    while(frameIndex < totalFrames):
        raw_image = pipe.stdout.read(videoWidth*videoHeight*3)
        if ((frameIndex % skipLength) == 0):
            # transform the byte read into a numpy array
            image =  np.frombuffer(raw_image, dtype='uint8')
            image = image.reshape((videoHeight,videoWidth,3))

            new_im = Image.fromarray(image)
            #new_im.show()
            
            #minDim = min(videoHeight,videoWidth)
            ## Crop the center of the image
            #left = (videoWidth - minDim)/2
            #top = (videoHeight - minDim)/2
            #right = (videoWidth + minDim)/2
            #bottom = (videoHeight + minDim)/2
            #new_im = new_im.crop((left, top, right, bottom))
            new_im = new_im.resize((height,width))
            #new_im.save(str(numSegments)+".jpg")
          
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
            outVerbNames = []
            outNounNames = []
            for model in [tsn, tsm]:
                s = time.time()
                verb_logits, noun_logits = model(inputs)
                curr_time = time.time() - s 
                if model==tsn:
                    tsnTime.append(curr_time)
                else:
                    tsmTime.append(curr_time)
                verbs = verb_logits.detach().numpy()
                nouns =  noun_logits.detach().numpy()
                verbLabel = verbs[0].argsort()[-5:][::-1]
                nounLabel = nouns[0].argsort()[-5:][::-1]
                ind=0
                while ind<=4:
                    outVerbNames.append(verbLabels['class_key'][verbLabel[ind]] )
                    outNounNames.append(nounLabels['class_key'][nounLabel[ind]])
                    ind = ind + 1

            ss = frameIndex / float(frameRate)
            totNames = [str(datetime.timedelta(seconds=ss))]
            for vNames in outVerbNames:
                totNames.append(vNames)
            for vNames in outNounNames:
                totNames.append(vNames)
            writer.writerow(totNames)
            #print(str(datetime.timedelta(seconds=ss)), outVerbNames[0], outNounNames[0])
            
            out = None
            numSegments = 0

    outFile.close()
    print("tsm - Inference Time: ", sum(tsmTime)/len(tsmTime))
    print("tsn - Inference Time: ", sum(tsnTime)/len(tsnTime))
 