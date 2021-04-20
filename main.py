import io
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import clip
import PIL
from dall_e import map_pixels, unmap_pixels, load_model

output_path = './generations'
prompt = 'A penguin made of flowers'
lr = 3e-1
img_save_freq = 5

output_dir = os.path.join(output_path, f'"{prompt}"')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

target_img_size = 256
final_img_size = 512

ref_img = PIL.Image.open("img/pingu.png")
ref_img = ref_img.resize((224,224))
ref_img = np.asarray(ref_img, np.float32) / 255.
ref_img = np.moveaxis(ref_img, 2, 0)
ref_img = torch.tensor(ref_img).to(device)

vgg_model = torchvision.models.vgg16(pretrained=True).to(device).eval()
vgg_layers = vgg_model.features
vgg_layer_name_mapping = {
    '3': "relu1_2",
    '8': "relu2_2",
    '15': "relu3_3",
    '22': "relu4_3",
}

def preprocess(img):
    min_img_dim = min(img.size)

    if min_img_dim < target_img_size:
        raise ValueError(f'min dim for img {min_img_dim} < {target_img_size}')

    img_ratio = target_img_size / min_img_dim
    min_img_dim = (round(img_ratio * img.size[1]),
                   round(img_ratio * img.size[0]))
    img = TF.resize(img, min_img_dim, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_img_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


def compute_clip_loss(img, text, ref_img=None):
    img = clip_transform(img)
    img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    img_logits = clip_model.encode_image(img)

    tokenized_text = clip.tokenize([text]).to(device).detach().clone()
    text_logits = clip_model.encode_text(tokenized_text)

    loss = 10 * -torch.cosine_similarity(text_logits, img_logits).mean()

    if ref_img is not None:
        ref_img = clip_transform(ref_img)
        ref_img = torch.nn.functional.upsample_bilinear(ref_img, (224, 224))
        ref_img_logits = clip_model.encode_image(ref_img)

        loss += 10 * -torch.cosine_similarity(ref_img_logits, img_logits).mean()
        loss /= 2

    return loss

def compute_visual_loss(img_batch, ref_img):
    loss = 0

    for img in img_batch:
        loss += (img - ref_img)**2

    return 10 * torch.sum(loss) / (224*224*img_batch.shape[0])

def compute_perceptual_loss(img_batch, ref_img):
    loss = 0

    img_visual_feats = img_batch
    ref_visual_feats = ref_img

    for name, module in vgg_layers._modules.items():
        img_visual_feats = module(img_visual_feats)
        ref_visual_feats = module(ref_visual_feats)

        if name in vgg_layer_name_mapping:
            loss += 10 * -torch.cosine_similarity(img_visual_feats, ref_visual_feats).mean()
        
    loss /= len(vgg_layer_name_mapping)
    return loss

def get_stacked_random_crops(img, num_random_crops=64):
    img_size = [img.shape[2], img.shape[3]]

    crop_list = []
    for _ in range(num_random_crops):
        crop_size_y = int(img_size[0] * torch.zeros(1, ).uniform_(.75, .95))
        crop_size_x = int(img_size[1] * torch.zeros(1, ).uniform_(.75, .95))

        y_offset = torch.randint(0, img_size[0] - crop_size_y, ())
        x_offset = torch.randint(0, img_size[1] - crop_size_x, ())

        crop = img[:, :, y_offset:y_offset + crop_size_y,
                   x_offset:x_offset + crop_size_x]

        crop = torch.nn.functional.upsample_bilinear(crop, (224, 224))

        crop_list.append(crop)

    img = torch.cat(crop_list, axis=0)

    return img

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

clip_transform = torchvision.transforms.Compose([
    # clip_preprocess.transforms[2],
    clip_preprocess.transforms[4],
])

dec = load_model("https://cdn.openai.com/dall-e/decoder.pkl", device)
dec.eval()

z_logits = torch.rand((1, 8192, 64, 64)).cuda()

z_logits = torch.nn.Parameter(z_logits, requires_grad=True)

optimizer = torch.optim.Adam(
    params=[z_logits],
    lr=lr,
    betas=(0.9, 0.999),
)

counter = 0
while True:
    z = torch.nn.functional.gumbel_softmax(
        z_logits.permute(0, 2, 3, 1).reshape(1, 64**2, 8192),
        hard=False,
        dim=1,
    ).view(1, 8192, 64, 64)

    x_stats = dec(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

    x_rec_stacked = get_stacked_random_crops(
        x_rec,
        num_random_crops=32,
    )

    loss_list = []

    # clip_loss = compute_clip_loss(x_rec_stacked, prompt, ref_img.unsqueeze(0))
    # loss_list.append(clip_loss)
    # print(f"CLIP {clip_loss}")

    # visual_loss = compute_visual_loss(x_rec_stacked, ref_img)
    # visual_loss *= 0.1
    # loss_list.append(visual_loss)
    # print(f"VISUAL {visual_loss}")

    visual_loss = compute_perceptual_loss(x_rec_stacked, ref_img.unsqueeze(0))
    loss_list.append(visual_loss)
    print(f"PERCEPTUAL {visual_loss}")

    loss = sum(loss_list) / len(loss_list)

    print(f"LOSS {loss}")
    print("\n")



    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    counter += 1
    if counter % img_save_freq == 0:
        x_rec_vis = T.ToPILImage(mode='RGB')(x_rec[0])
        x_rec_vis.save(f"{output_dir}/{counter}.png")
        print(f"SAVED {counter}")
        print("\n")