#Clone clip architecture from openai on github
#clone taming transformers from openai on github
#source https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf
# import libraries
import numpy as np
import torch, os, imageio, pdb, math
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import PIL
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf
from CLIP import clip
#import warnings
#warnings.filterwarnings('ignore')

##helper functions
def show_from_tensor(tensor):
    img = tensor.clone()
    img = img.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10,7))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def norm_data(data):
    return (data.clip(-1,1) + 1)/2 ### range moves between 0 and 1

###Parameters
learning_rate = .5
batch_size = 1
wd =.1 # regularizer parameter, used by the optimizer helping limit the size of the wight
noise_factor = .1 # crops of images and resize them, noise factore used to improve

total_iter = 100
im_shape = [225, 400, 3] # height, width and channel
size1, size2, channels = im_shape

### Clip model ###
clipmodel, _ = clip.load('ViT-B/32', jit=False)
clipmodel.eval() # inference
print(clip.available_models())
print('Clip model visual resolution', clipmodel.visual.input_resolution)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

### Taming transformer ###
# download the transformers
from taming-transformers.taming.models.vqgan import VQModel
def load_config(config_path, display=False):
    config_data = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config_data)))

def load_vqgan(config, chkpath=None):
    model = VQModel(**config.model.params)
    if chkpath is not None:
        state_dict = torch.load(chkpath, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return model.eval() # we dont want to train, only inference

def generator(x):
    x = taming_model.post_quant_conv(x)
    x = taming_model.decoder(x)
    return x

taming_config = load_config('./models/vqgan_imagenet_f16_16384/configs/model.yaml', display=True)
taming_model = load_vqgan(taming_config, chkpath='./models/vqgan_imagenet_f16_16384/checkpoints/last.ckpt').to(device)

### Set up the parameters to optimize ###
class Parameters(torch.nn.Module):
    def __init__(self):
        super(Parameters, self).__init__()
        self.data = .5*torch.randn(batch_size, 256, size1//16, size2//16).cuda()# 1x256x14x15
        self.data = torch.nn.Parameter(torch.sin(self.data))


    def forward(self, x):
        return self.data


def init_params():
    parameters = Parameters().cuda()
    optimizer = torch.optim.AdamW(parameters.data, lr=learning_rate)
    return parameters, optimizer


### encoding prompts
normalize = torchvision.transforms.normalize((0.48, 0.45, 0.408), (0.26, 0.26, 0.27))

def encodeText(text):
    t=clip.tokenize(text).cuda()
    t=clipmodel.encode_text(t).detach().clone()
    return t

def createEncodings(include, exclude, extras):
    # include specifing what to include as feature
    # exclude specifing features
    # extras features
    include_enc=[]
    for text in include:
        include_enc.append(encodeText(text))

    exclude_enc=encodeText(exclude) if exclude != '' else 0
    extras_enc=encodeText(extras) if extras != '' else 0

    return include_enc, exclude_enc, extras_enc

augTransform = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomAffine(30, (.2,.2), fill=0)
).cuda()

Params, optimizer = init_params()

with torch.no_grad():
    print(Params().shape)
    img=norm_data(generator(Params()).cpu())#1x3x224x400
    print(img.shape)
    show_from_tensor(img[0])


### Create crops, generating imgs
def create_crops(img, num_crops=30):
    p=size1//2
    img=torch.nn.functional.pad(img,(p,p,p,p), mode='constant', value=0)
    img=augTransform(img)

    crop_set=[]
    for ch in range(num_crops):
        gap1=int(torch.normal(1,.5,()).clip(.2, 1.5)*size1)
        gap2=int(torch.normal(1,.5,()).clip(.2, 1.5)*size1)
        offsetx=torch.randint(0, int(size1*2-gap1),())
        offsety=torch.randint(0, int(size1*2-gap1),())

        crop=img[:,:,offsetx:offsetx+gap2,offsety:offsety+gap2]
        crop=torch.nn.functional.interpolate(crop,(224,224), mode='bilinear', align_corners=True)
        crop_set.append(crop)

    img_crops=torch.cat(crop_set) #30x3x224x224
    img_crops=img_crops+noise_factor*torch.rand_like(img_crops,requires_grad=False)
    return img_crops

### show the generation
def showme(Params, show_crop):
    with torch.no_grad():
        generated=generator(Params())

        if show_crop:
            print('Augmented crop example')
            aug_gen=generated.float() #1x3#224x400
            aug_gen=create_crops(aug_gen, num_crops=1)
            aug_gen_norm=norm_data(aug_gen[0])
            show_from_tensor(aug_gen_norm)

        print('generation')
        latest_gen=norm_data(generated.cpu())
        show_from_tensor(latest_gen[0])

    return latest_gen[0]

def optimize_result(Params, prompt):
    alpha=1 #importance of including encodings
    beta=.5 #importance of exluding encodings

    # img encoding
    out=generator(Params())
    out=norm_data(out)
    out=create_crops(out)
    out=normalize(out)
    imange_enc=clipmodel.encode_image(out)

    # txt encoding
    final_enc=w1*prompt+w2*extras_enc
    final_text_include_enc=final_enc/final_enc.norm(dim=1, keepdim=True)
    final_text_exclude_enc=exclude_enc

    # loss
    main_loss=torch.cosine_similarity(final_text_include_enc, imange_enc, -1)
    penalize_loss=torch.cosine_similarity(final_text_exclude_enc, imange_enc, -1)

    final_loss=-alpha*main_loss+beta*penalize_loss
    return final_loss

# optimization process
def optimize(Params, optimizer, prompt):
    loss=optimize_result(Params, prompt).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

### Training loop ###
def training_loop(Params, optimizer, show_crop=False):
    res_img=[]
    res_z=[]

    for prompt in include_enc:
        iteration=0
        Params, optimizer=init_params()

        for it in range(total_iter):
            loss=optimize(Params, optimizer, prompt)

            if iteration>0 and iteration%(total_iter-1)==0:
                new_img=showme(Params, show_crop)
                res_img.append(new_img)
                res_z.append(Params())
                print(loss.item())

            iteration+=1
        #clean cache
        torch.cuda.empty_cache()
    return res_img, res_z

### Creating a video of the interpolation ###
torch.cuda.empty_cache()
include=['sketch of a lady', 'sketch of a man on a horse']
exclude=['watermark', 'cropped']
extras='watercolor paper texture'
w1=1
w2=1
include_enc, exclude_enc, extras_enc= createEncodings(include, exclude, extras)
res_img, res_z=training_loop(Params, optimizer, show_crop=True)

# interpolation to create e video
def interpolate(res_z_list, duration_list):
    gen_img_list=[]
    fps=25

    for idx, (z, duration) in enumerate(zip(res_z_list, duration_list)):
        num_steps=int(duration*fps)
        z1=z
        z2=res_z_list[(idx+1)%len(res_z_list)]

        for step in range(num_steps):
            alpha=math.sin(1.5*step/num_steps)**6
            z_new=alpha*z2+(1-alpha)*z1

            new_gen=norm_data(generator(z_new).cpu())[0]
            new_img=T.ToPILImage(mode='RGB')(new_gen)
            gen_img_list.append(new_img)

    return gen_img_list

durations=[3,3,3,3,3,3,3]
interp_result_img_list=interpolate(res_z, durations)

### creating the video
out_video_path=f"../res2.mp4"
writer=imageio.get_writer(out_video_path, fps=25)
for pil_img in interp_result_img_list:
    img=np.array(pil_img, dtype=np.uint8)
    writer.append_data(img)

writer.close() # write and save in the path

#display the video
from IPython.display import HTML
from base64 import b64encode
mp4=open("../res2.mp4","rb").read()
data="data:video/mp4;base64,"+b64encode(mp4).decode()
HTML(data)


