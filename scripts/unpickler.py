import pytorch_lightning as pl
import torch

state_dict = torch.load('models/17-11-2021-15-35-04.ckpt')["state_dict"]

# save the pytorch model
torch.save(state_dict, 'models/state_dict_only.pt')