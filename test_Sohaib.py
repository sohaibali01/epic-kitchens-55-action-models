
import datasetManager
from model_loader import load_checkpoint
import torch

orig_model = "./data/TSM_arch=resnet50_modality=RGB_segments=8-cfc93918.pth.tar"
tsm_model = "./data/customModel2.pth"

if __name__ == "__main__":

    tsm_Model = load_checkpoint(orig_model)
    ckpt = torch.load(tsm_model)
    tsm_Model.load_state_dict(ckpt.state_dict())

    tsm_Model.eval()
    dt = datasetManager.youtubeDataset(root_dir="D:/EPIC-KITCHEN/data/test/",
                        nounCSV = "./data/EPIC_noun_classes.csv",
                        verbCSV = "./data/EPIC_verb_classes.csv")
    
    running_corrects_verbs = 0
    running_corrects_nouns = 0

    running_corrects_verbs_k = 0
    running_corrects_nouns_k = 0

    for inputs, verb_labels, noun_labels in dt:
        verb_logits, noun_logits = tsm_Model(inputs)
       
        # Top 3
        k = 3 # return top-3
        _, sorted_verb_indices = torch.topk(verb_logits, k, dim=1, largest=True, sorted=True)
        _, sorted_noun_indices = torch.topk(noun_logits, k, dim=1, largest=True, sorted=True)
        running_corrects_verbs_k += torch.sum(sorted_verb_indices == verb_labels.data)
        running_corrects_nouns_k += torch.sum(sorted_noun_indices == noun_labels.data)

        # Top 1
        _, preds_Verb = torch.max(verb_logits, 1) 
        _, preds_Noun = torch.max(noun_logits, 1)
        running_corrects_verbs += torch.sum(preds_Verb == verb_labels.data)
        running_corrects_nouns += torch.sum(preds_Noun == noun_labels.data)

    acc_verb_k = running_corrects_verbs_k.double() / len(dt)
    acc_noun_k = running_corrects_nouns_k.double() / len(dt)

    acc_verb = running_corrects_verbs.double() / len(dt)
    acc_noun = running_corrects_nouns.double() / len(dt)

    print("Top-1 Accuracy: Verb ", acc_verb.data)
    print("Top-1 Accuracy: Noun ", acc_noun.data)

    print("Top-3 Accuracy: Verb ", acc_verb_k.data)
    print("Top-3 Accuracy: Noun ", acc_noun_k.data)

    
   