from .classification_model_base import ClassificationModelBase
import torch


class ResBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim, s, activation, **specs):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_dim, out_dim, kernel_size=(3,3), stride=s, **specs)
        self.bano_1 = torch.nn.BatchNorm2d(out_dim)
        self.activation_1 = activation()
        self.conv_2 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=(3,3), stride=1, **specs)
        self.bano_2 = torch.nn.BatchNorm2d(out_dim)
        if in_dim != out_dim:
            self.conv_res = torch.nn.Conv2d(in_dim, out_dim, kernel_size=(1,1), stride=2, padding=0, bias=False)
            self.bano_res = torch.nn.BatchNorm2d(out_dim)
        else:
            self.conv_res = torch.nn.Identity()
            self.bano_res = torch.nn.Identity()
        self.activation_2 = activation()

    def forward(self, x, res):
        x = self.conv_1(x)
        x = self.bano_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.bano_2(x)
        res = self.conv_res(res)
        res = self.bano_res(res)
        x += res
        x = self.activation_2(x)
        return x

    
class ResNet18Extractor(torch.nn.Module):
    def __init__(self, in_shape, **kwargs):
        super().__init__()
        specs = {'bias': False, 'padding': 1}
        self.activation = torch.nn.ReLU
        self.conv = torch.nn.Conv2d(in_shape[0], 64, kernel_size=(3,3), stride=1, **specs)
        self.batch_norm = torch.nn.BatchNorm2d(64)
        self.activation_0 = self.activation()
        self.layer_1_block_1 = ResBlock(64, 64, 1, self.activation, **specs)
        self.layer_1_block_2 = ResBlock(64, 64, 1, self.activation, **specs)
        self.layer_2_block_1 = ResBlock(64, 128, 2, self.activation, **specs)
        self.layer_2_block_2 = ResBlock(128, 128, 1, self.activation, **specs)
        self.layer_3_block_1 = ResBlock(128, 256, 2, self.activation, **specs)
        self.layer_3_block_2 = ResBlock(256, 256, 1, self.activation, **specs)
        self.layer_4_block_1 = ResBlock(256, 512, 2, self.activation, **specs)
        self.layer_4_block_2 = ResBlock(512, 512, 1, self.activation, **specs)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation_0(x)
        res = x
        x = self.layer_1_block_1(x, res)
        res = x
        x = self.layer_1_block_2(x, res)
        res = x
        x = self.layer_2_block_1(x, res)
        res = x
        x = self.layer_2_block_2(x, res)
        res = x
        x = self.layer_3_block_1(x, res)
        res = x
        x = self.layer_3_block_2(x, res)
        res = x
        x = self.layer_4_block_1(x, res)
        res = x
        x = self.layer_4_block_2(x, res)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return x


class ResNet18(ClassificationModelBase):

    def __init__(self, device, in_shape, num_classes, **kwargs):
        super().__init__(device)
        self.in_shape = in_shape
        self.device = device
        self.extractor = ResNet18Extractor(in_shape)
        self.classifier = torch.nn.Linear(512, num_classes, bias=True)
        self.initialize()