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

from __future__ import print_function

import os
import time

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torchvision.utils import save_image

from discriminator import Discriminator
from net import *

torch.manual_seed(42)

im_size = 32
gamma = 15

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
discrim = Discriminator().to(device)

criterion = torch.nn.BCELoss()


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x)**2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar -
                            mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


rec_dir = "/content/drive/MyDrive/vae_vertical_attention/results_rec"
gen_dir = "/content/drive/MyDrive/vae_vertical_attention/results_gen"
models_dir = "/content/drive/MyDrive/vae_vertical_attention/models"

os.makedirs(rec_dir, exist_ok=True)
os.makedirs(gen_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


def main():
    z_size = 512
    vae = VAE(zsize=z_size, layer_count=4)
    vae.cuda()
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    lr = 0.0005

    enc_optimizer = optim.Adam(
        vae.encoder.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    discrim_optimizer = optim.Adam(
        discrim.parameters(), lr=lr)
    dec_optimizer = optim.Adam(
        vae.decoder.parameters(), lr=lr)

    train_epoch = 100

    sample1 = torch.randn(batch_size, z_size).view(-1, z_size, 1, 1).cuda()

    for epoch in range(train_epoch):
        vae.train()

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0:
            enc_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for i, data in enumerate(trainloader):
            x = data[0].cuda()
            real_label = torch.ones(x.shape[0], 1).to(device)
            fake_label = torch.zeros(x.shape[0], 1).to(device)

            vae.train()
            discrim.train()

            # vae loss
            rec, mu, logvar = vae(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            vae_loss = loss_re + loss_kl

            enc_optimizer.zero_grad()
            vae_loss.backward(retain_graph=True)
            enc_optimizer.step()

            z_p = torch.randn(batch_size, z_size).to(device)

            # decoder loss
            # classify all real batch
            rec1, mu1, logvar1 = vae(x)

            output_real = discrim(x)[0]
            loss_gan_real_dec = criterion(output_real, real_label)

            # classify all fake batch
            output_fake = discrim(rec1)[0]
            loss_gan_fake_dec = criterion(output_fake, fake_label)

            gen = vae.decoder(z_p)
            gen = gen * 0.5 + 0.5
            output_gen = discrim(gen)[0]
            loss_x_p = criterion(output_gen, fake_label)
            gan_loss_dec = loss_gan_real_dec + loss_gan_fake_dec + loss_x_p

            rec1, mu1, logvar1 = vae(x)
            rec_loss_dec = torch.mean((rec1 - x)**2)
            dec_loss = gamma * rec_loss_dec - gan_loss_dec

            dec_optimizer.zero_grad()
            dec_loss.backward(retain_graph=True)
            dec_optimizer.step()

            # GAN loss
            # classify all real batch
            rec2, mu2, logvar2 = vae(x)

            output_real = discrim(x)[0]
            loss_gan_real = criterion(output_real, real_label)

            # classify all fake batch
            output_fake = discrim(rec2)[0]
            loss_gan_fake = criterion(output_fake, fake_label)

            gen = vae.decoder(z_p)
            gen = gen * 0.5 + 0.5
            output_gen = discrim(gen)[0]
            loss_x_p = criterion(output_gen, fake_label)
            gan_loss = loss_gan_real + loss_gan_fake + loss_x_p

            discrim_optimizer.zero_grad()
            gan_loss.backward(retain_graph=True)
            discrim_optimizer.step()

            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()
            #############################################

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            i += 1
            if i % m == 0:
                rec_loss /= m
                kl_loss /= m
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    x_rec, _, _ = vae(x)
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample  # .cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               f'{rec_dir}/sample_' + str(epoch) + "_" + str(i) + '.png')
                    x_rec = vae.decode(sample1)
                    resultsample = x_rec * 0.5 + 0.5
                    resultsample = resultsample  # .cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               f'{gen_dir}/sample_' + str(epoch) + "_" + str(i) + '.png')

        if (epoch + 1) % 5 == 0:
            print(f"saving model")
            torch.save(vae.state_dict(
            ), f"{models_dir}/VAEmodel_cifar_{epoch}.pkl")

    print("Training finish!... save training results")


if __name__ == '__main__':
    main()
