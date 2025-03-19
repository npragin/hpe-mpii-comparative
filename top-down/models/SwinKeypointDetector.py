import torch.nn as nn
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b

class SwinKeypointDetectorHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = self.net(x)
        return out

# TODO: If using pretrained we need to do some finetuning techniques
class SwinKeypointDetector(nn.Module):
    def __init__(self, num_keypoints, mlp_hidden_dim, swin_variant="swin_t", pretrained=False):
        super().__init__()

        self.num_keypoints = num_keypoints

        if swin_variant == 'swin_t':
            weights = "DEFAULT" if pretrained else None
            self.swin = swin_v2_t(weights=weights)
            feature_dim = 768
        elif swin_variant == 'swin_s':
            weights = "DEFAULT" if pretrained else None
            self.swin = swin_v2_s(weights=weights)
            feature_dim = 768
        elif swin_variant == 'swin_b':
            weights = "DEFAULT" if pretrained else None
            self.swin = swin_v2_b(weights=weights)
            feature_dim = 1024

        self.swin.head = nn.Identity()
        
        self.coords_head = SwinKeypointDetectorHead(feature_dim, mlp_hidden_dim, num_keypoints * 2) # *2 for (x, y) pairs
 
        self.visibility_head = SwinKeypointDetectorHead(feature_dim, mlp_hidden_dim, num_keypoints) # *1 for 2 visibility classes

    def forward(self, x):
        height, width = x.size()[2:]

        encoded = self.swin(x)

        norm_coords = self.coords_head(encoded)
        norm_coords = norm_coords.reshape(-1, self.num_keypoints, 2)

        visibility = self.visibility_head(encoded)
        visibility = visibility.reshape(-1, self.num_keypoints, 1)

        # Concatenated or separate outputs?
        # out = torch.cat([coords, visibility], dim=2)
                
        if self.training:
            # During training, return normalized coordinates
            return norm_coords, visibility
        else:
            # During inference, rescale to original dimensions
            scaled_coords = norm_coords.clone()
            scaled_coords[:, :, 0] *= width
            scaled_coords[:, :, 1] *= height
            return scaled_coords, visibility