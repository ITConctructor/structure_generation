import torch
from accelerate import Accelerator
from tqdm.notebook import tqdm
from torch import nn 
import random
import matplotlib.pyplot as plt

class Trainer():
  def __init__(self, params):
    torch.manual_seed(42)
    self.train_dataloader = params["train_dataloader"]
    self.val_dataloader = params["val_dataloader"]
    self.score = params["score"]
    self.train_score_computing_frequency = params.get("train_score_computing_frequency", 1)
    self.optimizer = params["optimizer"]
    self.optimizer_args = params["optimizer_args"]
    self.scheduler = params.get("scheduler", None)
    self.scheduler_args = params.get("scheduler_args", None)
    self.freezer = params.get("freezer", None)
    self.freezer_args = params.get("freezer_args", None)
    self.training_controller = params.get("training_controller", None)
    self.show_outputs = params.get("show_outputs", False)
    self.show_outputs_every = params.get("show_outputs_every", 1)
    self.n_outputs = params.get("n_outputs", 2)
    self.n_epochs = params["n_epochs"]
    self.device = params["device"]
    self.backup_path = params.get("backup_path", None)
    self.is_saving_history = params.get("is_saving_history", True)
    self.history = {
        "train_loss":[],
        "train_score":[],
        "val_loss":[],
        "val_score":[]
    }
  def train(self, model, verbose=True):
    model = model.to(self.device)
    optimizer = self.optimizer(model.parameters(), **self.optimizer_args)
    scheduler = None
    if self.scheduler != None and self.scheduler_args != None:
      scheduler = self.scheduler(optimizer, **self.scheduler_args)
    freezer = None
    if self.freezer != None and self.freezer_args != None:
      freezer = self.freezer(model, **self.freezer_args)

    accelerator = Accelerator(
        mixed_precision="fp16",
    )
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, self.train_dataloader, scheduler
    )
    self.train_dataloader = train_dataloader

    for epoch in range(self.n_epochs):
      #Training
      model.train()
      if freezer != None:
        freezer.step()
      train_loss = 0
      train_score = 0
      steps = 0
      train_pbar = tqdm(self.train_dataloader, desc="Training")

      local_losses = []
      local_scores = []
      for batch in train_pbar:
        with accelerator.accumulate(model):
          train_batch = self.device_dict(batch["train"])
          loss = model(train_batch)
          accelerator.backward(loss)
          accelerator.clip_grad_norm_(model.parameters(), 1.0)
          local_losses.append(loss.item())

          val_batch = self.device_dict(batch["val"])
          labels = self.device_dict(batch["labels"])

          if self.score != None:
            preds = model.predict(val_batch)
            if steps % self.train_score_computing_frequency == 0:
              score = self.score(preds, labels)
              local_scores.append(score)

          if self.training_controller != None:
            self.training_controller.update_batch(model=model, optimizer=optimizer,
                                                  scheduler=scheduler, freezer=freezer,
                                                  losses=local_losses, scores=local_scores)
          optimizer.step()
          optimizer.zero_grad()
          if scheduler != None:
            scheduler.step()

        display_loss = round(loss.item(), 5)
        train_pbar.set_description("Training. Loss: %s" % display_loss)
        steps += 1



      train_loss = sum(local_losses)/len(local_losses)
      if len(local_scores) > 0:
        train_score = sum(local_scores)/len(local_scores)



      #Validation
      val_model = accelerator.unwrap_model(model)
      val_model.eval()
      val_loss = 0
      val_score = 0
      for batch in tqdm(self.val_dataloader, desc="Validation"):
        with torch.no_grad():
          train_batch = self.device_dict(batch["train"])

          optimizer.zero_grad()
          loss = val_model(train_batch)
          val_loss += loss.item()/len(self.val_dataloader)

          val_batch = train_batch = self.device_dict(batch["val"])
          labels = train_batch = self.device_dict(batch["labels"])

          if self.score != None:
            preds = val_model.predict(val_batch)
            score = self.score(preds, labels)
            val_score += score/(len(self.val_dataloader))

      self.history["train_loss"].append(train_loss)
      self.history["train_score"].append(train_score)
      self.history["val_loss"].append(val_loss)
      self.history["val_score"].append(val_score)

      if self.training_controller != None:
          self.training_controller.update_epoch(model=val_model, optimizer=optimizer,
                                                scheduler=scheduler, freezer=freezer,
                                                losses=self.history["train_loss"], scores=self.history["train_score"])

      if verbose:
        print(f"Epoch {epoch+1}. Train loss: {train_loss}, val loss: {val_loss}, train score: {train_score}, val score: {val_score}")

      if self.show_outputs == True and (epoch + 1) % self.show_outputs_every == 0:
        self.show_outputs_fn(val_model, self.val_dataloader)

      if self.backup_path != None:
        torch.save(val_model, self.backup_path)
      if self.is_saving_history:
        self.save_history()

  def save_history(self):
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(range(len(self.history["train_loss"])), self.history["train_loss"], label="Train")
    ax.plot(range(len(self.history["val_loss"])), self.history["val_loss"], label="Val")
    fig.savefig("Losses.pdf")

    fig, ax = plt.subplots()
    ax.plot(range(len(self.history["train_score"])), self.history["train_score"], label="Train")
    ax.plot(range(len(self.history["val_score"])), self.history["val_score"], label="Val")
    fig.savefig("Scores.pdf")

  def plot_lr_schedule(self):
    steps = [self.optimizer_args["lr"]]
    if self.scheduler == None or self.scheduler_args == None:
      steps = [self.optimizer_args["lr"] for i in range(self.n_epochs)]
    else:
      dummyModel = nn.Linear(10, 10)
      dummyOptimizer = self.optimizer(dummyModel.parameters(), **self.optimizer_args)
      scheduler = scheduler = self.scheduler(dummyOptimizer, **self.scheduler_args)
      for i in range(1, self.n_epochs):
        dummyOptimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        steps.append(lr)
    fig, ax = plt.subplots()
    ax.plot(range(len(steps)), steps)
    plt.show()
  def device_dict(self, dictionary):
    for k in dictionary.keys():
      if type(dictionary[k]) == type({}):
        dictionary[k] = self.device_dict(dictionary[k])
      else:
        dictionary[k] = dictionary[k].to(self.device)
    return dictionary
  def show_outputs_fn(self, model, dataloader):
    dataset = dataloader.dataset
    collate_fn = dataloader.collate_fn
    indexes = random.sample(range(len(dataset)), self.n_outputs)
    inputs = [dataset[i] for i in indexes]
    processed = collate_fn(inputs)
    batch = self.device_dict(processed)
    print("Output samples:\n")
    model.show_outputs(batch["val"])
    dataset.show_samples(indexes)