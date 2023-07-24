import os
import sys
from LBGNN import LBGNN
from SRCNN import LCGNN
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

input_root = './DataSet/train/'
input_val_root = './DataSet/val/'

class MyDataset(Dataset):
    def __init__(self, input_root, transform=None):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.transforms = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):

        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = Image.open(input_img_path).convert('RGB')
        input_img = self.transforms(input_img)
        return input_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.Resize([360, 640]),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize([360, 640]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}

train_dataset = MyDataset(input_root, transform=data_transform["train"])
train_num = len(train_dataset)
validate_dataset = MyDataset(input_val_root, transform=data_transform["val"])
val_num = len(validate_dataset)

batch_size = 16

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=100, shuffle=False, num_workers=0)
print("using {} images for training, {} images for validation.".format(train_num, val_num))

BL_net = LBGNN()
BL_net.to(device)
BL_optimizer = optim.Adam(BL_net.parameters(), lr=0.0002)

LC_net = LCGNN()
LC_net.to(device)
LC_optimizer = optim.Adam(LC_net.parameters(), lr=0.0002)

loss_function = nn.MSELoss()

epochs = 600
BL_save_path = './LBGNN.pth'
LC_save_path = './SRCNN.pth'
max_loss = 10
train_steps = len(train_loader)
train_loss_data = []
valid_loss_data = []

for epoch in range(epochs):
    BL_net.train()
    LC_net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, images in enumerate(train_bar):

        BL_optimizer.zero_grad()
        LC_optimizer.zero_grad()
        Backlight = BL_net(images.to(device))
        LCsignal = LC_net(images.to(device))

        Backlight_inv = Backlight / 2 + 0.5
        LCsignal_inv = LCsignal / 2 + 0.5

        Backlight1 = Backlight_inv[:, 0:3, :, :]
        Backlight2 = Backlight_inv[:, 3:6, :, :]
        LC1 = LCsignal_inv[:, 0:1, :, :]
        LC2 = LCsignal_inv[:, 1:2, :, :]

        field1 = Backlight1*LC1
        field2 = Backlight2*LC2
        display = field1+field2

        display[display < 0] = 0
        display[display > 1] = 1

        images_inv = images / 2 + 0.5

        loss = loss_function(display, images_inv.to(device)) + 0.01 * loss_function(field1, images_inv.to(device))
        loss.backward()

        BL_optimizer.step()
        LC_optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1, epochs, loss)

    BL_net.eval()
    LC_net.eval()
    valid_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_images in val_bar:
            Backlight = BL_net(val_images.to(device))
            LCsignal = LC_net(val_images.to(device))

            Backlight_inv = Backlight / 2 + 0.5
            LCsignal_inv = LCsignal / 2 + 0.5

            Backlight1 = Backlight_inv[:, 0:3, :, :]
            Backlight2 = Backlight_inv[:, 3:6, :, :]
            LC1 = LCsignal_inv[:, 0:1, :, :]
            LC2 = LCsignal_inv[:, 1:2, :, :]

            field1 = Backlight1 * LC1
            field2 = Backlight2 * LC2
            display = field1 + field2

            display[display < 0] = 0
            display[display > 1] = 1

            val_images_inv = val_images / 2 + 0.5

            loss = loss_function(display, val_images_inv.to(device)) + 0.01 * loss_function(field1, val_images_inv.to(device))
            valid_loss += loss.item()

    print('[epoch %d] train_loss: %.5f  val_loss: %.5f' % (epoch + 1, running_loss / train_steps, valid_loss))

print('Finished Training')