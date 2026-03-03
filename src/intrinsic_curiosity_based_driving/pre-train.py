import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

from ensemble import StateAutoEncoder
from src.utils.datasets import VideoFramesDataset



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 dev_loader,
                 test_loader,
                 batch_size,
                 device,
                 num_epochs,
                 learning_rate,
                 earlystopping,
                 optimizer,
                 save_path = None
                 ):
        self.model = model
        self.earlystopping = earlystopping
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.device = device
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.save_path = save_path
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss()

    def train(self):
        for epoch in self.num_epochs:
            self.model.train()
            step = 0
            for batch in self.train_loader:
                batch = torch.Tensor(batch, dtype=torch.float32).to(self.device)

                self.optimizer.zero_grad()

                reconstruction, _ = self.model(batch)

                loss = self.criterion(reconstruction.flatten(), batch.flatten())
                loss.backward()
                self.optimizer.step()
                print(f"Epoch {epoch+1}/{self.num_epochs}, Step: {step}, Loss: {loss.item()}")
                step += 1
            eval_loss = self.eval()

            if self.save_path and self.earlystopping.best_loss > eval_loss:
                torch.save(self.model.state_dict(), self.save_path)

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Eval Loss: {eval_loss}")

            self.earlystopping(eval_loss)
            if self.earlystopping.should_stop:
                print("Early stopping triggered")
                break

    def eval(self, test = False):
        self.model.eval()
        losses = []

        dataloader = self.dev_loader
        if test:
            dataloader = self.test_loader

        with torch.no_grad():
            for batch in dataloader:
                batch = torch.Tensor(batch, dtype = torch.float32).to(self.device)

                reconstruction, _ = self.model(batch)

                loss = self.criterion(reconstruction.flatten(), batch.flatten())
                losses.append(loss.item())

        return sum(losses)/len(losses)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    """
    GET THE DATASET
    """

    lmdb_path = "/home/sergio/Documentos/tfg/gym-liftoff-connector/data/training_data/video_images.lmdb"

    dataset = VideoFramesDataset(lmdb_path=lmdb_path, transform=transform)
    num_frames = len(dataset)
    all_indices = list(range(num_frames))

    train_idx, resto_idx = train_test_split(all_indices, test_size=0.3, random_state=42)
    dev_idx, test_idx = train_test_split(resto_idx, test_size=0.5, random_state=42)

    train_dataset = VideoFramesDataset(lmdb_path, indices=train_idx, transform=transform)
    dev_dataset = VideoFramesDataset(lmdb_path, indices=dev_idx, transform=transform)
    test_dataset = VideoFramesDataset(lmdb_path, indices=test_idx, transform=transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    """
    PREPARE THE TRAINING
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StateAutoEncoder(latent_dim=256)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30

    earlystopping = EarlyStopping(patience=10, min_delta=0.01)

    save_path = "/home/sergio/Documentos/tfg/gym-liftoff-connector/src/intrinsic_curiosity_based_driving/state_encoder.pth"

    trainer = Trainer(model, train_loader, dev_loader, test_loader, batch_size, device, num_epochs, learning_rate, earlystopping, optimizer)
    trainer.train()

    test_loss = trainer.eval(test=True)

    print(f"FINAL TEST LOSS: {test_loss}")



