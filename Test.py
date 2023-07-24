import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from LBGNN import LBGNN
from SRCNN import LCGNN
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tran(img):
    img = img.numpy()
    img = np.squeeze(img, 0)
    img = img / 2 + 0.5  # Unnormalize
    img = np.transpose(img, (1, 2, 0))
    return img
data_transform = transforms.Compose([transforms.Resize([360, 640]),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
BL_model = LBGNN().to(device)
BL_weights_path = "./LBGNN.pth"
BL_model.load_state_dict(torch.load(BL_weights_path))
LC_model = LCGNN().to(device)
LC_weights_path = "./SRCNN.pth"
LC_model.load_state_dict(torch.load(LC_weights_path))

i = 836  # 0836.png, 0872.png and 0886.png are from DIV2K dataset
s = str(i).zfill(4)
img_path = s + ".png"
Ori_img = Image.open(img_path).convert('RGB')

img = data_transform(Ori_img)
img = torch.unsqueeze(img, dim=0)

BL_model.eval()
LC_model.eval()
with torch.no_grad():
    output = BL_model(img.to(device)).cpu()
    LCsignal = LC_model(img.to(device)).cpu()

Backlight = tran(output)
Backlight[Backlight > 1] = 1
Backlight[Backlight < 0] = 0
LCsignal = tran(LCsignal)
LCsignal[LCsignal > 1] = 1
LCsignal[LCsignal < 0] = 0
Backlight1 = Backlight[:, :, 0:3]
Backlight2 = Backlight[:, :, 3:6]
LC1 = LCsignal[:, :, 0:1]
LC2 = LCsignal[:, :, 1:2]


field1 = Backlight1 * LC1
field2 = Backlight2 * LC2

field2_cbu = np.concatenate((np.zeros((360, 20, 3)), field2[:, 0:620, :]), axis=1)

display = field1 + field2
display[display > 1] = 1

colorbreakup = field1 + field2_cbu
colorbreakup[colorbreakup > 1] = 1

plt.subplot(3, 3, 1)
plt.imshow(Backlight1)
plt.title('1st real backlight')
plt.axis('off')
plt.subplot(3, 3, 2)
plt.imshow(LC1, cmap="gray")
plt.title('1st LC signal')
plt.axis('off')
plt.subplot(3, 3, 3)
plt.imshow(field1)
plt.title('1st front screen image')
plt.axis('off')
plt.subplot(3, 3, 4)
plt.imshow(Backlight2)
plt.title('2nd real backlight')
plt.axis('off')
plt.subplot(3, 3, 5)
plt.imshow(LC2, cmap="gray")
plt.title('2nd LC signal')
plt.axis('off')
plt.subplot(3, 3, 6)
plt.imshow(field2)
plt.title('2nd front screen image')
plt.axis('off')
plt.subplot(3, 3, 7)
plt.imshow(Ori_img)
plt.title('Original image')
plt.axis('off')
plt.subplot(3, 3, 8)
plt.imshow(display)
plt.title('Display image')
plt.axis('off')
plt.subplot(3, 3, 9)
plt.imshow(colorbreakup)
plt.title('Color breakup image')
plt.axis('off')
plt.show()
