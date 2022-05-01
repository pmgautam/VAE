import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2, stride=1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(32, 128, 5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.conv4 = nn.Conv2d(256, 256, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)
        self.fc1 = nn.Linear(4*4*256, 512)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.9)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv4(x)))
        x = x.view(-1, 256*4*4)
        x1 = x
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x, x1
