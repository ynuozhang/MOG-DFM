import torch
from functools import partial
from torch.utils.data import DataLoader
from torch import nn

def collate_fn(batch):
  input_ids = torch.tensor(batch[0]['input_ids'])
  attention_mask = torch.tensor(batch[0]['attention_mask'])
  return {
    'input_ids': input_ids,
    'attention_mask': attention_mask
  }

class CustomDataModule(nn.Module):
    def __init__(self, train_dataset, val_dataset, test_dataset, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          collate_fn=partial(self.collate_fn),
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          collate_fn=partial(self.collate_fn),
                          num_workers=8,
                          pin_memory=True,
                          shuffle=False)
  
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          collate_fn=partial(self.collate_fn),
                          num_workers=8,
                          pin_memory=True,
                          shuffle=False)