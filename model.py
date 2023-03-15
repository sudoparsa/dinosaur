import torch
import math
import sys
from torch import nn
import einops
from torch.nn import functional as F
import utils
from slot_attention import SlotAttention
import vision_transformer as vits
from torchvision import models as torchvision_models


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class MLP_Decoder(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.modules = [
            nn.Conv2d(hid_dim, hid_dim, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(hid_dim, hid_dim, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(hid_dim, hid_dim, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(hid_dim, out_dim + 1, kernel_size=1),
        ]
        self.decoder = nn.Sequential(*self.modules)
    
    def forward(self, caption_rpr, image_size, token_mask=None):
        ### [b, l, d] -> [b, 3, h, w]
        batch_size, len_caption, dim = caption_rpr.shape
        h, w = image_size
        
        temp = einops.repeat(caption_rpr, 'b l d -> (b l) d h w', h=h, w=w)
        position_embd_2d = positionalencoding2d(dim, h, w).to(next(self.parameters()).device)
        temp = temp + position_embd_2d
        temp = self.decoder(temp)
        temp = einops.rearrange(temp, '(b l) c w h -> b l c w h', b = batch_size)
        recons = temp[:, :, :-1, :, :]
        masks = temp[:, :, -1:, :, :]
        masks[token_mask == 0] = -torch.inf
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        result = {
            'output': recon_combined,
            'token_outputs': recons,
            'masks': masks
        }
        return result


class Dinosaur(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch_size = args.patch_size
        self.encoder = self.get_encoder(args)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.slot_attn = SlotAttention(num_slots = args.num_slots,
                                       dim = args.dim,
                                       iters = args.iters,
                                       eps = args.epsilon, 
                                       hidden_dim = args.slot_hdim
                                       )
        self.decoder = MLP_Decoder(args.dim, args.dim)
    

    def get_encoder(self, args):
        if "vit" in args.arch:
            if 'vit_tiny' == args.arch:
                model = vits.vit_tiny(patch_size=args.patch_size, num_classes=0)
            if 'vit_small' == args.arch:
                model = vits.vit_small(patch_size=args.patch_size, num_classes=0)
            print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
        elif "xcit" in args.arch:
            model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        elif args.arch in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[args.arch](num_classes=0)
            model.fc = nn.Identity()
        else:
            print(f"Architecture {args.arch} non supported")
            sys.exit(1)
        model.cuda()
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        model.eval()
        return model

    def forward(self, inputs):
        # [b, 3, 32, 32] -> [b, N, D_emb]
        B, nc, w, h = inputs.shape
        result = {}
        x = self.encoder(inputs)
        result['dino_output'] = x
        x = self.slot_attn(x)
        result['slot_attention_output'] = x
        x = self.decoder(x, image_size=(w // self.patch_size, h // self.patch_size))
        result['decoder_output'] = x 
        return result
