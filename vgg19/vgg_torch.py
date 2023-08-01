import os
import torch
import torch.nn as nn
import numpy as np
import inspect
import torchvision.transforms as T
import torchvision.models as models
import PIL.Image as Image


# VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19(nn.Module):
    def __init__(self, vgg19_pth_path, vgg19_npy_path=None):
        super(Vgg19, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path

        self.vgg19_npy_path = vgg19_npy_path
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        self.conv1_1 = self.conv_layer("conv1_1")
        self.conv1_2 = self.conv_layer("conv1_2")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = self.conv_layer("conv2_1")
        self.conv2_2 = self.conv_layer("conv2_2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = self.conv_layer("conv3_1")
        self.conv3_2 = self.conv_layer("conv3_2")
        self.conv3_3 = self.conv_layer("conv3_3")
        self.conv3_4 = self.conv_layer("conv3_4")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = self.conv_layer("conv4_1")
        self.conv4_2 = self.conv_layer("conv4_2")
        self.conv4_3 = self.conv_layer("conv4_3")
        self.conv4_4 = self.conv_layer("conv4_4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = self.conv_layer("conv5_1")
        self.conv5_2 = self.conv_layer("conv5_2")
        self.conv5_3 = self.conv_layer("conv5_3")
        self.conv5_4 = self.conv_layer("conv5_4")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc_layer("fc6", 25088, 4096)
        self.fc6 = self.fc_layer("fc6", 25088, 4096)
        self.fc7 = self.fc_layer("fc7", 4096, 4096)
        self.fc8 = self.fc_layer("fc8", 4096, 1000)

        weight_files = torch.load(vgg19_pth_path)
        self.load_state_dict(weight_files, strict=True)

    def forward(self, rgb, include_top=True):
        # red, green, blue = torch.chunk(rgb, chunks=3, dim=1)

        # x = torch.cat([
        #     blue - VGG_MEAN[0]/255,
        #     green - VGG_MEAN[1]/255,
        #     red - VGG_MEAN[2]/255,
        # ], dim=1)
        rgb = rgb.clone()
        x = T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(rgb)

        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.conv3_2(x))
        x = torch.relu(self.conv3_3(x))
        x = torch.relu(self.conv3_4(x))
        x = self.pool3(x)

        x = torch.relu(self.conv4_1(x))
        x = torch.relu(self.conv4_2(x))
        x = torch.relu(self.conv4_3(x))
        x = torch.relu(self.conv4_4(x))
        x = self.pool4(x)

        x = torch.relu(self.conv5_1(x))
        x = torch.relu(self.conv5_2(x))
        x = torch.relu(self.conv5_3(x))
        x = torch.relu(self.conv5_4(x))
        x = self.pool5(x)

        if include_top:
            x = x.view(-1, 25088)  # flatten
            x = torch.relu(self.fc6(x))  # 问题的元凶，每次运行到这里结果不同
            x = torch.relu(self.fc7(x))
            x = self.fc8(x)
            # x到这里是logits
            x = torch.softmax(x, dim=1)

        return x

    def fprop(self, rgb, include_top=False):
        # forward propagation，前向传播的缩写
        # similar to forward, but return the intermediate outputs
        # red, green, blue = torch.chunk(rgb, chunks=3, dim=1)
        # red, green, blue = torch.split(rgb, split_size_or_sections=1, dim=1)
        # print(red.grad_fn, green.grad_fn)  # SplitBackward0
        #
        # x = torch.cat((
        #     blue - VGG_MEAN[0],
        #     green - VGG_MEAN[1],
        #     red - VGG_MEAN[2],
        # ), dim=1)
        rgb = rgb.clone()
        x = T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(rgb)

        x = torch.relu(self.conv1_1(x))
        conv1_1 = x.clone()  # save the output of conv1_1
        x = torch.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2_1(x))
        conv2_1 = x.clone()  # save the output of conv2_1
        x = torch.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = torch.relu(self.conv3_1(x))
        conv3_1 = x.clone()  # save the output of conv3_1
        x = torch.relu(self.conv3_2(x))
        x = torch.relu(self.conv3_3(x))
        x = torch.relu(self.conv3_4(x))
        x = self.pool3(x)

        x = torch.relu(self.conv4_1(x))
        conv4_1 = x.clone()  # save the output of conv4_1
        x = torch.relu(self.conv4_2(x))
        conv4_2 = x.clone()  # save the output of conv4_2
        x = torch.relu(self.conv4_3(x))
        x = torch.relu(self.conv4_4(x))
        x = self.pool4(x)

        x = torch.relu(self.conv5_1(x))
        conv5_1 = x.clone()  # save the output of conv5_1
        x = torch.relu(self.conv5_2(x))
        x = torch.relu(self.conv5_3(x))
        x = torch.relu(self.conv5_4(x))
        x = self.pool5(x)

        if include_top:
            # the same as before
            x = x.view(-1, 25088)  # flatten
            x = torch.relu(self.fc6(x))
            x = torch.relu(self.fc7(x))
            x = self.fc8(x)
            # x到这里是logits
            x = torch.softmax(x, dim=1)

        return conv4_2, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

    def conv_layer(self, name):
        # with torch.no_grad():
        filt = self.get_conv_filter(name)
        # print(filt.shape)  # (3, 3, 3, 64)，即卷积核height，width，in_channel，out_channel
        conv = nn.Conv2d(filt.shape[2], filt.shape[3], kernel_size=filt.shape[0], padding=(filt.shape[0] - 1) // 2,
                         bias=True)
        # print(filt.shape)
        # print(conv.weight.data.shape)  # torch.Size([128, 64, 3, 3])， 即out，in，size[0]，size[1]
        # print(conv.bias.data.shape)  # torch.Size([64])
        conv.weight.data = torch.from_numpy(filt).permute(3, 2, 0, 1).to(self.device)
        conv.bias.data = torch.from_numpy(self.get_bias(name)).to(self.device)
        return conv

    def fc_layer(self, name, in_channel, out_channel):
        # with torch.no_grad():
        fc = nn.Linear(in_channel, out_channel)
        fc_weight = self.get_fc_weight(name)
        # print(fc.weight.data.shape)  # torch.Size([4096, 25088])
        # print(fc.bias.data.shape)  # torch.Size([4096])
        fc.weight.data = torch.from_numpy(fc_weight).T.to(self.device)
        fc.bias.data = torch.from_numpy(self.get_bias(name)).to(self.device)
        # print(fc_weight.shape)  # (25088, 4096)
        return fc

    def get_conv_filter(self, name):
        return self.data_dict[name][0]

    def get_bias(self, name):
        # print(len(self.data_dict[name]))  # 2
        # print(self.data_dict[name][1].shape)  # (64,)
        return self.data_dict[name][1]

    def get_fc_weight(self, name):
        return self.data_dict[name][0]

    def get_params(self):
        """
        Provides access to the model's parameters.
        :return: A list of all Variables defining the model parameters.
        """

        if hasattr(self, 'params'):
            return list(self.params)

        # Catch eager execution and assert function overload.
        try:
            if torch.is_tensor():
                raise NotImplementedError("For Eager execution - get_params "
                                          "must be overridden.")
        except AttributeError:
            pass

        # For graph-based execution
        scope_vars = list(self.named_parameters())

        if len(scope_vars) == 0:
            self.make_params()
            scope_vars = list(self.named_parameters())
            assert len(scope_vars) > 0

        # Make sure no parameters have been added or removed
        if hasattr(self, "num_params"):
            if self.num_params != len(scope_vars):
                print("Scope: ", self.scope)
                print("Expected " + str(self.num_params) + " variables")
                print("Got " + str(len(scope_vars)))
                for var in scope_vars:
                    print("\t" + str(var))
                assert False
        else:
            self.num_params = len(scope_vars)

        return scope_vars

# if __name__ == '__main__':
#     vgg = Vgg19().eval()
#     vgg = vgg.to(vgg.device)
#     im = Image.open("../physical-attack-data/content/t-shirt/t-shirt2.jpg")
#     im_tens = T.ToTensor()(im).to(vgg.device)
#     im_tens = T.Resize([224, 224])(im_tens)
#     # print(type(vgg))
#     result = vgg(im_tens)
#     re = torch.argmax(result)
#     print(re)