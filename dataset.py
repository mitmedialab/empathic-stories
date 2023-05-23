import sys
sys.path.append("./")

import torch

class EmpathicStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, task, data_pairs, data_stories, labels):
        self.data_pairs = data_pairs
        self.data_stories = data_stories
        self.task = task
        self.labels = labels

    def __getitem__(self, idx):
        if self.task == "similarity":
            i = self.data_pairs.iloc[idx]
            s1 = i["story_A"].replace("\n", "")
            s2 = i["story_B"].replace("\n", "")
            score = i["similarity_empathy_human_AGG"]
            return [s1, s2, score]
        
        elif self.task == "summary":
            story = self.data_stories.iloc[idx]["story"].replace("\n", "")
            event = self.data_stories.iloc[idx]["Main Event"]
            emotion = self.data_stories.iloc[idx]["Emotion Description"]
            moral = self.data_stories.iloc[idx]["Moral"]
            combined = ""
            if "EVE" in self.labels:
                combined += "[EVE]" + event 
            if "EMO" in self.labels:
                combined += "[EMO]" + emotion 
            if "MOR" in self.labels:
                combined += "[MOR]" + moral
            return story, combined
            
    def __len__(self):
        if self.task == "summary":
            return len(self.stories)
        else:
            return len(self.story_pairs)
        