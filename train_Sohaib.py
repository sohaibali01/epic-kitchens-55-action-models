from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt
import time
import os
import copy

import datasetManager
from model_loader import load_checkpoint

batch_size = 6
num_epochs = 25

feature_extract = True
tsm_model = "./data/TSM_arch=resnet50_modality=RGB_segments=8-cfc93918.pth.tar"
savePath1 = "./data/customModel1.pth"
savePath2 = "./data/customModel2.pth"
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects_verbs = 0
            running_corrects_nouns = 0
            
            # Iterate over data.
            for inputs, verb_labels, noun_labels in dataloaders[phase]:
                #inputs = inputs.permute(3,1,2,0)
                #noun_labels = noun_labels.permute(1,0)
                #verb_labels = verb_labels.permute(1,0)

                inputs = inputs.to(device)
                verb_labels = verb_labels.to(device)
                noun_labels = noun_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    verb_logits, noun_logits = model(inputs)
                    
                    verb_labels = verb_labels.flatten()
                    noun_labels = noun_labels.flatten()

                    loss_Verb = criterion(verb_logits, verb_labels)
                    loss_Noun = criterion(noun_logits, noun_labels)
                    loss = loss_Verb + loss_Noun
                    #loss_nouns = criterion(noun_logits, inputs['verb'])
                    _, preds_Verb = torch.max(verb_logits, 1)   
                    _, preds_Noun = torch.max(noun_logits, 1)
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects_verbs += torch.sum(preds_Verb == verb_labels.data)
                running_corrects_nouns += torch.sum(preds_Noun == noun_labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc_verb = running_corrects_verbs.double() / len(dataloaders[phase].dataset)
            epoch_acc_noun = running_corrects_nouns.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: Verb {:.4f} Acc: Noun {:.4f}'.format(phase, epoch_loss, epoch_acc_verb, epoch_acc_noun))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_ft, feature_extract):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    input_size = 0
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.fc_verb = nn.Linear(model_ft.fc_verb.in_features, model_ft.fc_verb.out_features)
    model_ft.fc_noun = nn.Linear(model_ft.fc_noun.in_features, model_ft.fc_noun.out_features)
    input_size = 224

    return model_ft, input_size


if __name__ == "__main__":
   
    tsm_Model = load_checkpoint(tsm_model)
    print(tsm_Model)
   # Initialize the model for this run
    model_ft, input_size = initialize_model(tsm_Model, feature_extract)

    # Print the model we just instantiated
    print(model_ft)

    dt = datasetManager.youtubeDataset(root_dir="D:/EPIC-KITCHEN/data/train/",
                        nounCSV = "./data/EPIC_noun_classes.csv",
                        verbCSV = "./data/EPIC_verb_classes.csv")

    dataloader = DataLoader(dt, batch_size=batch_size,shuffle=True, num_workers=2,drop_last=True)

    dataloaders_dict = {'train': dataloader}
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    torch.save(model_ft.state_dict(), savePath1)
    torch.save(model_ft, savePath2)

    #for i in range(len(dt)):
    #    sample = dt[i]

#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.eval()