# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch import nn
from torch.nn import functional as F
from urllib3 import encode_multipart_formdata

from feature_extraction_vat_cnn import FCN_Encoder

params = {"dropout": 0.1, "input_channels": 3}


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class Encoder(nn.Module):
    def __init__(self, enc, zsize):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(1024 * 2 * 2, zsize)
        self.fc2 = nn.Linear(1024 * 2 * 2, zsize)
        self.enc = enc

    def forward(self, x):
        x = self.enc(x)

        h1 = self.fc1(x)
        h2 = self.fc2(x)

        return h1, h2


class PrintShape(nn.Module):
    def __init__(self, layer_name=""):
        super(PrintShape, self).__init__()
        self.layer_name = layer_name

    def forward(self, x):
        # print(f"{self.layer_name} shape: {x.shape}")
        return x


class VAE(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3):
        super(VAE, self).__init__()

        d = 128
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" %
                    (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(1024 * 2 * 2, zsize)
        self.fc2 = nn.Linear(1024 * 2 * 2, zsize)

        self.d1 = nn.Linear(zsize, inputs * 2 * 2)

        mul = inputs // d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" %
                    (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1),
                nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

        enc_modules = []

        for i in range(self.layer_count):
            enc_modules.append(getattr(self, "conv%d" % (i + 1)))
            enc_modules.append(getattr(self, "conv%d_bn" % (i + 1)))
            enc_modules.append(nn.ReLU())

        enc_modules.append(PrintShape("decode1"))
        enc_modules.append(Reshape((1024 * 2 * 2,)))
        enc_modules.append(PrintShape("decode2"))

        encoder = nn.Sequential(*enc_modules)
        self.encoder = Encoder(encoder, zsize)

        dec_modules = [Reshape((self.zsize,)),
                       nn.Linear(zsize, self.d_max * 2 * 2),
                       Reshape((self.d_max, 2, 2)),
                       nn.LeakyReLU(0.2)]

        for i in range(1, self.layer_count):
            dec_modules.append(getattr(self, "deconv%d" % (i + 1)))
            dec_modules.append(getattr(self, "deconv%d_bn" % (i + 1)))
            dec_modules.append(nn.LeakyReLU(0.2))

        dec_modules.append(getattr(self, "deconv%d" % (self.layer_count + 1)))
        dec_modules.append(nn.Tanh())

        self.decoder = nn.Sequential(*dec_modules)

    def encode(self, x):
        # for i in range(self.layer_count):
        #     x = F.relu(getattr(self, "conv%d_bn" % (i + 1))
        #                (getattr(self, "conv%d" % (i + 1))(x)))

        # x = self.encoder(x)
        # x = x.view(x.shape[0], 256 * 2 * 2)
        # h1 = self.fc1(x)
        # h2 = self.fc2(x)
        h1, h2 = self.encoder(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, 2, 2)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))
                             (getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        z = z.view(-1, self.zsize, 1, 1)

        return self.decode(z), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
