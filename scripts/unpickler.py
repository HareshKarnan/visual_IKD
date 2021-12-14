import pytorch_lightning as pl
import torch
from scripts.train import IKDModel

model = IKDModel.load_from_checkpoint('models/17-11-2021-15-35-04.ckpt')
torch.save(model.ikd_model.state_dict(), 'models/torchmodel.pt')

# state_dict = torch.load('models/17-11-2021-15-35-04.ckpt')["state_dict"]


# save the pytorch model
# torch.save(state_dict, 'models/state_dict_only.pt')