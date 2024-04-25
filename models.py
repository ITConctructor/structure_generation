import torch
from torch import nn 
from diffusers import UNet3DConditionModel, DDPMScheduler
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np

class DiffusersDDPM3D(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.device = params["device"]
    self.sample_channels = params.get("sample_channels", 1)
    self.sample_shape = params.get("sample_shape", (32, 32, 32))

    self.unet_channels = params.get("unet_channels", 64)
    self.unet_n_blocks = params.get("unet_n_blocks", 2)
    self.unet_channels = params.get("channels", (320, 640, 1280, 1280))
    self.unet_channels_mults = params.get("unet_channels_mults", (1,2,2,4))
    self.unet_attention_indicators = params.get("unet_attention_indicators", (False, False, True, True))

    self.cross_attention_dim = 8
    self.eps_model = UNet3DConditionModel(sample_size = self.sample_shape, in_channels = self.sample_channels, out_channels = self.sample_channels,
                                 layers_per_block = self.unet_n_blocks, cross_attention_dim=self.cross_attention_dim, block_out_channels=self.unet_channels)
    self.noise_scheduler = params.get("noise_scheduler", DDPMScheduler())
    self.loss = nn.MSELoss()

  def forward(self, batch):
    x0 = batch["gt_sample"]
    batch_size = x0.shape[0]
    t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0).to(x0.device)

    dummy_cond = torch.zeros(batch_size, x0.shape[1], self.cross_attention_dim).to(self.device)
    xt = self.noise_scheduler.add_noise(x0, noise, t)
    eps_theta = self.eps_model(xt, t, dummy_cond)[0]
    loss = self.loss(eps_theta, noise)
    return loss
  def predict(self, batch):
    with torch.no_grad():
      x = batch["noise"]
      dummy_cond = torch.zeros(x.shape[0], x.shape[2], self.cross_attention_dim).to(self.device)
      for t in tqdm(self.noise_scheduler.timesteps):
        #x = self.noise_scheduler.scale_model_input(x, timestep=t)
        time = x.new_full((x.shape[0],), t, dtype=torch.long)
        xt = self.eps_model(x, time, dummy_cond)[0]
        x = self.noise_scheduler.step(xt, t, x).prev_sample
      output = {
          "gen_sample":x
      }
    return output
  def show_outputs(self, batch):
      x = self.predict(batch)["gen_sample"][:,0,:,:,:].squeeze(1)
      x = abs(x + abs(x.min()))
      x = (x/x.max()*255).type("torch.LongTensor")

      sample = x[0].detach().cpu().numpy()
      fig = plt.figure()
      for i in range(x.shape[0]):
        tess = x[i].detach().cpu().numpy()[:,:,:,np.newaxis].repeat(3, axis=3)/255
        color = np.concatenate([tess, np.ones((*self.sample_shape, 1))], axis=3)
        ax = fig.add_subplot(projection='3d')
        ax.voxels(sample, facecolors=color)
      plt.show()
      return ""