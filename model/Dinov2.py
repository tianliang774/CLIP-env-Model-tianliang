import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models

dino_backbones = {
    'dinov2_s': {
        'name': 'dinov2_vits14',
        'embedding_size': 384,
        'patch_size': 14
    },
    'dinov2_b': {
        'name': 'dinov2_vitb14',
        'embedding_size': 768,
        'patch_size': 14
    },
    'dinov2_l': {
        'name': 'dinov2_vitl14',
        'embedding_size': 1024,
        'patch_size': 14
    },
    'dinov2_g': {
        'name': 'dinov2_vitg14',
        'embedding_size': 1536,
        'patch_size': 14
    },
}


class linear_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class conv_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5):
        super(conv_head, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 64, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=7),
            nn.Conv2d(64, num_classes, (3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        x = torch.sigmoid(x)
        return x


class DinoV2_Generator(nn.Module):
    def __init__(self, num_classes, backbone='dinov2_s', head='conv', backbones=None):
        super(DinoV2_Generator, self).__init__()
        if backbones is None:
            backbones = dino_backbones
        self.heads = {
            'conv': conv_head
        }
        self.backbones = backbones
        # Convert 1-channel image to 3-channel image
        self.preprocess = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.backbone = load(r'model/facebookresearch_dinov2_master', self.backbones[backbone]['name'], source="local")
        self.backbone.eval()
        self.num_classes = num_classes  # add a class for background if needed
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']
        self.head = self.heads[head](self.embedding_size, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)
        x = self.preprocess(x.cuda())
        with torch.no_grad():
            x = self.backbone.forward_features(x)
            x = x['x_norm_patchtokens']
            x = x.permute(0, 2, 1)
            tx = x.reshape(batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1]))
        x = self.head(tx)
        return tx, x


if __name__ == "__main__":
    # TEST CODE
    test = torch.randn((1, 1, 504, 504))
    model = DinoV2_Generator(5).cuda()
    feat, res = model(test)
    print(res.shape)
    print(feat.shape)
