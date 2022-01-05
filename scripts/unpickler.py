import pytorch_lightning as pl
import torch
from scripts.train import IKDModel

model = IKDModel.load_from_checkpoint('models/30-12-2021-20-23-24.ckpt')
# torch.save(model.ikd_model.state_dict(), 'models/torchmodel.pt')
torch.save(model.ikd_model.state_dict(), 'models/visiontorchmodel.pt')

# state_dict = torch.load('models/17-11-2021-15-35-04.ckpt')["state_dict"]


# save the pytorch model
# torch.save(state_dict, 'models/state_dict_only.pt')