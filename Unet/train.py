from model.unet_model import UNet
from model.dataset import ISBI_Loader
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim

def train_net(net, device, data_path, epochs=10, batch_size=1, lr=0.00001):
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                              )
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            net.train()
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_1.pth')
                loss.backward()
                optimizer.step()
                pbar.update(1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    data_path = r"D:\project\datasets"
    train_net(net, device, data_path, epochs=10, batch_size=1)
