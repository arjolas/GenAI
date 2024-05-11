# Advanced GAN
# https://medium.com/@ideami articles on AI; work with diff architectures CNN

import torch, torchvision, os, PIL, pdb
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)
def show(tensor, num=25):
    data = tensor.detach().cpu()
    grid = make_grid(data[:num], nrow=5).permute(1,2,0)

    plt.imshow(grid.clip(0,1))
    plt.show()

### hyper parameters and general parameters
n_epochs = 10000
batch_size = 128
lr = 1e-4
z_dim = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cur_step=0
crit_cycles=5
gen_losses=[]
crit_losses=[]
show_step=35
save_step=35

# generator model
class Generator(nn.Module):
    def __init__(self, z_dim=64, d_dim=16):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.d_dim = d_dim

        # powerful to detect image features equations:
        # nn.Conv2d: (n+2*pad-ks)//stride + 1
        # nn.ConvTranspose2d: (n-1)*stride -2*padding+ks
        # n: the old width or height of the image
        # ks: kernel size: the size of the kernel, eg 3 for 3x3
        # stride: the amount of positions you move everytime sliding the kernel
        # pad: the amount of padding to add to the image

        self.gen = nn.Sequential(
            #nn.ConvTranspose2d
            nn.ConvTranspose2d(z_dim, d_dim*32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d_dim*32),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim*32, d_dim*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim*16),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim*16, d_dim*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 8, d_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 4, d_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 2, d_dim * 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1)
        return self.gen(x)

def gen_noise(num, z_dim, device='cuda'):
    return torch.randn(num, z_dim, device=device)


class Critic(nn.Module):
    def __init__(self, z_dim=64, d_dim=16):
        super(Critic, self).__init__()
        self.z_dim = z_dim
        self.d_dim = d_dim
        self.gen = nn.Sequential(
            nn.Conv2d(3, d_dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(d_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim, d_dim*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(d_dim*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim*2, d_dim*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(d_dim*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim*4, d_dim*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(d_dim*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim*8,d_dim*16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(d_dim*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim*16,1, 4, 1, 0, bias=False),

        )

        def forward(self, image):
            # image: 128x3x128x128
            crit_pred= self.crit(image)
            return crit_pred.view(len(crit_pred), -1)


def init_weights(m):
    # init weights in different weights
    if isinstance(m, m.Conv2d) or isinstance(m, m.ConvTranspose2d):
        torch.nm.init.normal_(m.weight, 0.0, 0.02)
        torch.nm.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.BatchNorm2d):
        torch.nm.init.normal_(m.weight, 0.0, 0.02)
        torch.nm.init.constant_(m.bias, 0.0)

# eg: gen = gen.apply(init_weights)



### managing dataset
class Dataset(Dataset):
    def __init__(self, path, size=128, lim=10000):
        self.sizes=[size,size]
        items, labels=[], []

        for data in os.listdir(path)[:lim]:
            #path = './data/celeba/img_align_celeba'
            #data = 1111.jpg
            item = os.path.join(path, data)
            items.append(item)
            labels.append(data) # as gen ai models we dont need lables

        self.items = items
        self.labels = labels

    def __len__(self):
        return len(self)

    def __getitem__(self, idx):
        data = PIL.Image.open(self.items[idx]).convert('RGB')
        # return a data structure having the size of image 128x128
        # resize 128x128x3channels
        data = np.array(torchvision.transforms.Resize(self.sizes)(data))
        # dimension of 3 channels x 128 x 128
        # values of 0 to 255
        data = np.transpose(data, (2,0,1)).astype(np.float32, copy=False)
        # we need tensors with values from 0 to 1
        data = torch.from_numpy(data).div(255)
        return data, self.labels[idx]

# gradient penalty calculation

def get_gp(real, fake, crit, alpha, gamma=10):
    mix_images = real * alpha + fake * (1-alpha) # 128x3x128x128
    mix_scores = crit(mix_images) # 128x1

    gradient = torch.autograd.grad(
        inputs = mix_images,
        outputs = mix_scores,
        grad_outputs = torch.ones_like(mix_scores),
        retain_graph = True,
        create_graph = True
    ) # 128x128x128

    gradient = gradient.view(len(gradient), -1)# 128x49152
    gradient_norm = gradient.norm(2)
    gp = gamma * ((gradient_norm - 1)**2).mean()
    return gp

#save and load checkpoints
root_path = './data/'
def save_checkpoints(name):
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': gen.state_dict(),
        'optimizer_state_dict': gen_opt.state_dict()
    }, f'{root_path}G-{name}.pkl'

    )
    #critic
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': crit.state_dict(),
        'optimizer_state_dict': crit_opt.state_dict()
    }, f'{root_path}C-{name}.pkl'
    )

# load checkpoints
def load_checkpoints(name):
    checkpoint=torch.load(f'{root_path}G-{name}.pkl')
    gen.load_state_dict(checkpoint['model_state_dict'])
    gen_opt.load_state_dict(checkpoint['optimizer_state_dict'])

    #critic
    checkpoint = torch.load(f'{root_path}C-{name}.pkl')
    crit.load_state_dict(checkpoint['model_state_dict'])
    crit_opt.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    # download and load dataset
    import gdown, zipfile

    url = 'https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ'
    path = 'data/celeba'
    #download_path = f'{path}/img_align_celeba_png.7z'

    if not os.path.exists(path):
        os.makedirs(path)

    #after first download+unzip comment to not do it multiple times
    #gdown.download(url, download_path, quiet=False)

    #with zipfile.ZipFile(download_path, 'r') as ziphandler:
    #    ziphandler.extractall(path)

    # dataset creation
    data_path ='./data/celeba/img_align_celeba'
    ds = Dataset(data_path, size=128, lim=10000)

    # data loader
    dataloader = DataLoader(ds, batch_size= batch_size, shuffle=True)

    # models
    gen = Generator(z_dim).to(device)
    crit = Critic().to(device)

    # optimizer
    gen_opt = torch.optim.Adam(gen.parameters(), lr= lr, betas=(0.5, 0.9))
    crit_opt = torch.optim.Adam(crit.parameters(), lr= lr, betas=(0.5, 0.9))

    x,y = next(iter(dataloader))
    show(x)

    #save and load checkpoints
    #save_checkpoints('test')
    #load_checkpoints('test')

    #training loop
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_bs = len(real) #128
            real=real.to(device)

            ### critic
            mean_crit_loss=0
            for _ in range(crit_cycles):
                crit_opt.zero_grad()
                noise=gen_noise(cur_bs, z_dim)
                fake = gen(noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                alpha = torch.rand(len(real),1,1,1,device=device, requires_grad=True) #128x1x1x1
                gp= get_gp(real, fake.detach(), crit, alpha)

                crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp

                mean_crit_loss=crit_loss.item()/crit_cycles

                crit_loss.backward(retain_graph=True)
                crit_opt.step()

            crit_losses+=[mean_crit_loss]

            ### GEN
            gen_opt.zero_grad()
            noise=gen_noise(cur_bs, z_dim)
            fake = gen(noise)
            crit_fake_pred = crit(fake)

            gen_loss = crit_fake_pred.mean()
            gen_loss.backward()
            gen_opt.step()
            gen_losses.append(gen_loss.item())

            if cur_step % save_step and cur_step >0:
                print("saving checkpoint", cur_step, save_checkpoints)
                save_checkpoints('latest')

            ### stats
            if (cur_step % show_step and cur_step>0):
                show(fake, name='fake')
                show(real, name='real')

                gen_mean=sum(-gen_losses[show_step:])/len(gen_losses)
                crit_mean=sum(-crit_losses[show_step:])/len(crit_losses)
                print(f"Epoch: {epoch}: Step {cur_step}, gen loss: {gen_mean}, critic loss:{crit_mean}")

                plt.plot(
                    range(len(gen_losses)),
                    torch.Tensor(gen_losses).to(device),
                    label='gen_loss',
                )

                plt.plot(
                    range(len(crit_losses)),
                    torch.Tensor(crit_losses).to(device),
                    label='crit_loss',
                )

                plt.ylim(-1000,1000)
                plt.legend()
                plt.show()

            cur_step+=1