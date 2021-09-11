import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.utils import save_image


num_steps = 20000
learning_rate = 1e-3
alpha = 1
beta = 0.1
IMG_SIZE = 512 if torch.cuda.is_available() else 256


def load_img_as_tensor(img_name):
    img = cv2.imread(img_name)
    assert isinstance(img, np.ndarray)  # To check if image exist

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = torch.from_numpy(img / 255)  # np.array -> torch.tensor
    img = img.unsqueeze(0)  # img.shape (IMG_SIZE,IMG_SIZE,3)->(1,IMG_SIZE,IMG_SIZE,3)

    # img - Batch,Hight,Width,Channels ->Batch,Channels,Hight,Width
    img = torch.einsum("bhwc->bchw", img)
    return img.float()


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_layers = [0, 5, 10, 19, 28]

        self.model = models.vgg19(pretrained=True).eval()

        self.model = self.model.features[:29]

    def forward(self, x):
        features = []

        for idx, layer in enumerate(self.model):
            x = layer(x)

            if idx in self.features_layers:
                features.append(x)

        return features


def gram_matrix(x):
    batch, channels, hight, width = x.shape
    features = x.view(batch * channels, hight * width)
    G = torch.mm(features, features.t())
    return G


class StyleLoss(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = [out.detach() for out in features]

    def forward(self, x):
        loss = 0
        for target, pred in zip(self.features, x):
            G = gram_matrix(pred)
            A = gram_matrix(target)
            loss += F.mse_loss(A, G)
        return loss


class ContentLoss(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = [out.detach() for out in features]

    def forward(self, x):
        loss = 0
        for target, pred in zip(self.features, x):
            loss += F.mse_loss(target, pred)
        return loss


def transfer(model, img, style, content):

    img = img.detach().clone().to(device).requires_grad_()

    optimizer = optim.Adam([img])

    style_feature = model(style)
    content_feature = model(content)

    content_loss = ContentLoss(content_feature)
    style_loss = StyleLoss(style_feature)

    for idx in tqdm(range(num_steps)):
        img_feature = model(img)

        c_loss = content_loss(img_feature)
        s_loss = style_loss(img_feature)

        loss = alpha * s_loss + beta * c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 200 == 0:
            save_image(img.clone().detach(), f"out/out_{(idx//200):04d}.jpg")

    return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGG().to(device).eval()

style_img = load_img_as_tensor("content/style3.jpg").to(device)
content_img = load_img_as_tensor("style/kitten2.jpg").to(device)


img = content_img.clone().to(device)

transfer(model, img, style_img, content_img)
