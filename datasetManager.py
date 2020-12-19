
import os
from PIL import Image

import torch
import numpy as np
import csv

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image

def readCsv(fileName):
    columns = defaultdict(list) # each value in each column is appended to a list
    with open(fileName) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value 
                columns[k].append(v) # append the value into the appropriate list
    f.close()                                # based on column name k
    return columns

class youtubeDataset(Dataset):

    def __init__(self, root_dir, nounCSV, verbCSV, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.noun_list = readCsv(nounCSV)['class_key']
        self.verb_list = readCsv(verbCSV)['class_key']

        dirs = getListOfFiles(root_dir)
        folderNames = []
        for dir in dirs:
            folderNames.append(os.path.split(dir)[0])
        self.dirNames = list(set(folderNames))
        f=6

    def __len__(self):
        return len(self.dirNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        verb_noun = os.path.split(self.dirNames[idx])[1]
        verbName = verb_noun.split(' ')[0]
        nounName = verb_noun.split(' ')[1].replace("-",":")

        i=0
        out = None
        while i<8:
            imName = self.dirNames[idx] + "/" + str(i) + ".jpg"
            new_im = Image.open(imName)
            npImage = np.array(new_im)
            npImage = npImage.astype(np.float32) / 255.0
            outnp = np.rollaxis(npImage, 2, 0) # move channels from last to 1st dimension
            if out is None:
                out = outnp[:,:,:]
            else:
                out = np.concatenate((out, outnp[:,:,:]), 0)
            i=i+1

        i=0
        verbVector=[i]
        while i < len(self.verb_list):
            if self.verb_list[i] == verbName:
                verbVector[0]=i
            i=i+1

        i=0
        nounVector=[i]
        while i < len(self.noun_list):
            if self.noun_list[i] == nounName:
                nounVector[0]=i
            i=i+1

        sample = ( torch.from_numpy(out), torch.tensor(verbVector), torch.tensor(nounVector))

        if self.transform:
            sample = self.transform(sample)

        return sample

def getListOfFiles(dirName):
        # create a list of file and sub directories 
        # names in the given directory 
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
                    
        return allFiles

