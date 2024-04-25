import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np

class TesselationDataset(Dataset):
    def __init__(self, params):
        self.tess_dir = params["tess_dir"]
        self.names = params.get("names", os.listdir(self.tess_dir))
        self.dims = params.get("dims", 3)

    def __getitem__(self, ind):
        name = self.names[ind]
        path = f"{self.tess_dir}/{name}"
        tess = self.read_tesr(path)
        tess_tensor = torch.from_numpy(tess)[None,:,:,:].type("torch.FloatTensor")
        if self.dims == 2:
          tess_tensor = tess_tensor[:,0,:,:].squeeze(1)
        noise = noise = torch.randn_like(tess_tensor)

        batch = {
            "train":{
                "gt_sample":tess_tensor,
            },
            "val":{
                "noise":noise
            },
            "labels":{
                "gt_sample":tess_tensor,
            }
        }

        return batch

    def __len__(self) -> int:
        return len(self.names)
    def show_samples(self, indexes):
      fig = plt.figure()
      for ind in indexes:
        name = self.names[ind]
        path = f"{self.tess_dir}/{name}"
        sample = self.read_tesr(path)
        sample = (sample*255).astype("uint8")

        if self.dims == 3:
          ax = fig.add_subplot(projection='3d')
          colors = sample[:,:,:,np.newaxis].repeat(3, axis=3)/255
          colors = np.concatenate([colors, np.ones((*sample.shape, 1))], axis=3)
          ax.voxels(sample, facecolors=colors)
        else:
          sample = sample[0]
          plt.imshow(sample, cmap="gray")
      plt.show()
    def read_tesr(self, path):
      voxels = []
      n_cells = 0
      size = 0
      with open(path, "r") as f:
          lines = f.readlines()
      read_mode = "base"
      for s in lines:
          if s.find("**cell") != -1:
              read_mode = "cell"
              continue
          elif s.find("**data") != -1:
              read_mode = "pre_data"
              continue
          elif s.find("***end") != -1:
              break
          elif s.find("**general") != -1:
              read_mode = "pre_general"
              continue
          if read_mode == "pre_data":
              read_mode = "data"
              continue
          elif read_mode == "pre_general":
              read_mode = "general"
              continue
          elif read_mode == "cell":
              n_cells = int(s.strip())
              read_mode = "base"
          elif read_mode == "data":
              new_voxels = [int(i) for i in s.strip().split(" ")]
              voxels.extend(new_voxels)
          elif read_mode == "general":
              size = [int(i) for i in s.strip().split(" ")][0]
              read_mode = "base"
      tess = np.zeros((size, size, size))
      counter = 0
      for i in range(size):
        for j in range(size):
          for k in range(size):
            tess[k, j, i] = voxels[counter]/n_cells
            counter += 1
      return tess



def collate_fn(batch, dataset):
  if dataset.dims == 3:
    samples = torch.cat([b["train"]["gt_sample"][None,:,:,:,:] for b in batch], dim=0)
    noises = torch.cat([b["val"]["noise"][None,:,:,:,:] for b in batch], dim=0)
  else:
    samples = torch.cat([b["train"]["gt_sample"][None,:,:,:] for b in batch], dim=0)
    noises = torch.cat([b["val"]["noise"][None,:,:,:] for b in batch], dim=0)
  batch = {
      "train":{
          "gt_sample":samples,
      },
      "val":{
          "noise":noises
      },
      "labels":{
          "gt_sample":samples,
      }
  }
  return batch