from .classification_model_base import ClassificationModelBase
import torch


class ResBlock(torch.nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, s, activation):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_dim, hid_dim, kernel_size=(1,1), stride=s, padding=0, bias=False)
        self.bano_1 = torch.nn.BatchNorm2d(hid_dim)
        self.activation_1 = activation()
        self.conv_2 = torch.nn.Conv2d(hid_dim, hid_dim, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bano_2 = torch.nn.BatchNorm2d(hid_dim)
        self.activation_2 = activation()
        self.conv_3 = torch.nn.Conv2d(hid_dim, out_dim, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bano_3 = torch.nn.BatchNorm2d(out_dim)
        self.activation_3 = activation()

    def forward(self, x, res):
        x = self.conv_1(x)
        x = self.bano_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.bano_2(x)
        x = self.activation_2(x)
        x = self.conv_3(x)
        x = self.bano_3(x)
        x += res
        x = self.activation_3(x)
        return x
    

class ResNet50Extractor(torch.nn.Module):
    def __init__(self, in_shape, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.activation = torch.nn.ReLU
        self.conv = torch.nn.Conv2d(in_shape[0], 64, kernel_size=(7,7), stride=2, padding=3, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(64)
        self.activation_0 = self.activation()
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        self.conv_res_1 = torch.nn.Conv2d(64, 256, kernel_size=(1,1), padding=0, stride=1, bias=False)
        self.bano_res_1 = torch.nn.BatchNorm2d(256)
        self.layer_1_block_1 = ResBlock(64, 64, 256, 1, self.activation)
        self.layer_1_block_2 = ResBlock(256, 64, 256, 1, self.activation)
        self.layer_1_block_3 = ResBlock(256, 64, 256, 1, self.activation)
        self.conv_res_2 = torch.nn.Conv2d(256, 512, kernel_size=(1,1), padding=0, stride=2, bias=False)
        self.bano_res_2 = torch.nn.BatchNorm2d(512)
        self.layer_2_block_1 = ResBlock(256, 128, 512, 2, self.activation)
        self.layer_2_block_2 = ResBlock(512, 128, 512, 1, self.activation)
        self.layer_2_block_3 = ResBlock(512, 128, 512, 1, self.activation)
        self.layer_2_block_4 = ResBlock(512, 128, 512, 1, self.activation)
        self.conv_res_3 = torch.nn.Conv2d(512, 1024, kernel_size=(1,1), padding=0, stride=2, bias=False)
        self.bano_res_3 = torch.nn.BatchNorm2d(1024)
        self.layer_3_block_1 = ResBlock(512, 256, 1024, 2, self.activation)
        self.layer_3_block_2 = ResBlock(1024, 256, 1024, 1, self.activation)
        self.layer_3_block_3 = ResBlock(1024, 256, 1024, 1, self.activation)
        self.layer_3_block_4 = ResBlock(1024, 256, 1024, 1, self.activation)
        self.layer_3_block_5 = ResBlock(1024, 256, 1024, 1, self.activation)
        self.layer_3_block_6 = ResBlock(1024, 256, 1024, 1, self.activation)
        self.conv_res_4 = torch.nn.Conv2d(1024, 2048, kernel_size=(1,1), padding=0, stride=2, bias=False)
        self.bano_res_4 = torch.nn.BatchNorm2d(2048)
        self.layer_4_block_1 = ResBlock(1024, 512, 2048, 2, self.activation)
        self.layer_4_block_2 = ResBlock(2048, 512, 2048, 1, self.activation)
        self.layer_4_block_3 = ResBlock(2048, 512, 2048, 1, self.activation)
        self.pool_2 = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation_0(x)
        x = self.pool_1(x)
        res = self.conv_res_1(x)
        res = self.bano_res_1(res)
        x = self.layer_1_block_1(x, res)
        res = x
        x = self.layer_1_block_2(x, res)
        res = x
        x = self.layer_1_block_3(x, res)
        res = self.conv_res_2(x)
        res = self.bano_res_2(res)
        x = self.layer_2_block_1(x, res)
        res = x
        x = self.layer_2_block_2(x, res)
        res = x
        x = self.layer_2_block_3(x, res)
        res = x
        x = self.layer_2_block_4(x, res)
        res = self.conv_res_3(x)
        res = self.bano_res_3(res)
        x = self.layer_3_block_1(x, res)
        res = x
        x = self.layer_3_block_2(x, res)
        res = x
        x = self.layer_3_block_3(x, res)
        res = x
        x = self.layer_3_block_4(x, res)
        res = x
        x = self.layer_3_block_5(x, res)
        res = x
        x = self.layer_3_block_6(x, res)
        res = self.conv_res_4(x)
        res = self.bano_res_4(res)
        x = self.layer_4_block_1(x, res)
        res = x
        x = self.layer_4_block_2(x, res)
        res = x
        x = self.layer_4_block_3(x, res)
        x = self.pool_2(x)
        x = x.view(x.shape[0], -1)
        return x


class ResNet50(ClassificationModelBase):

    def __init__(self, device, in_shape, num_classes, **kwargs):
        super().__init__(device)
        self.in_shape = in_shape
        self.device = device
        self.activation = torch.nn.ReLU()
        self.extractor = ResNet50Extractor(in_shape)
        self.classifier = torch.nn.Linear(2048, num_classes, bias=True)
        self.initialize()