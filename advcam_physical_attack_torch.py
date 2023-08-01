import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
from vgg19.vgg_torch import Vgg19
from torchvision import models
from utils_torch import *
from physical_adaption_utils_torch import Physical_Adaptor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def content_loss(const_layer, var_layer, weight):
    # print(const_layer.grad_fn, var_layer.grad_fn)
    # <ReluBackward0 object at 0x000001D9E7C5BA90> <ReluBackward0 object at 0x000001D9E7C5BB20>
    return torch.mean((var_layer - const_layer) ** 2) * weight


def style_loss(CNN_structure, const_layers, content_const_layers, var_layer0, var_layer1, var_layer2, var_layer3,
               var_layer4, content_segs, style_segs, weight, style_layers_var_name=[]):
    """
    这段代码的目的是计算风格损失，即内容图和风格图在不同层次上的特征之间的差异。具体来说：

    - CNN_structure是一个列表，存储了卷积神经网络的结构，每个元素是一个层次的名称，比如"conv1_1/Relu"或"pool1"等。
    - const_layers, content_const_layers, var_layers是三个列表，存储了内容图、风格图和变量图在不同层次上的特征张量，每个元素是一个四维张量，形状为(batch, channel, height, width)。
    - content_segs, style_segs是两个列表，存储了内容图和风格图的分割掩码，每个元素是一个四维张量，形状为(batch, height, width, 1)。
    - weight是一个浮点数，表示风格损失的权重系数。
    - loss_styles是一个空列表，用于存储不同层次上的风格损失。
    - for循环遍历CNN_structure中的每一层次名称，根据层次类型进行分割掩码的下采样或平均池化，以保持和特征张量的形状一致。
    - 如果当前层次名称和var_layers中对应位置的层次名称相同，说明该层次需要计算风格损失。打印出该层次名称，并获取对应位置的特征张量和分割掩码。
    - layer_style_loss是一个浮点数，表示当前层次上的风格损失。初始化为0.0。
    - for循环遍历content_segs和style_segs中的每一对分割掩码，分别计算变量图和内容图或风格图在该区域上的Gram矩阵（特征之间的内积），并进行归一化处理。然后计算两个Gram矩阵之间的平方差，并乘以内容掩码的均值。将这个值累加到layer_style_loss中。
    - 将layer_style_loss乘以weight，并添加到loss_styles中。
    - 最后返回loss_styles，一个包含不同层次上风格损失的列表。
    """
    content_segs_resize = copy.deepcopy(content_segs)
    style_segs_resize = copy.deepcopy(style_segs)
    loss_styles = torch.empty(0).to(device)  # an empty list to store the style losses
    layer_count = float(len(const_layers))  # the number of layers to compute style loss
    layer_index = 0  # the index of the current layer

    # print(content_segs[0].shape)  # torch.Size([1, 1, 400, 400])
    _, _, content_seg_height, content_seg_width = content_segs_resize[0].shape  # the height and width of the content segmentation mask
    _, _, style_seg_height, style_seg_width = style_segs_resize[0].shape  # the height and width of the style segmentation mask
    for layer_name in CNN_structure:  # loop over the CNN structure
        # downsampling segmentation
        # 分析：如果当前层是池化层，就手动将content_segs和style_segs缩小一半
        if "pool" in layer_name:  # if the current layer is a pooling layer
            content_seg_width, content_seg_height = int(math.floor(content_seg_width / 2)), int(
                math.floor(content_seg_height / 2))  # reduce the content mask size by half
            style_seg_width, style_seg_height = int(math.floor(style_seg_width / 2)), int(
                math.floor(style_seg_height / 2))  # reduce the style mask size by half

            for i in range(len(content_segs_resize)):  # loop over the segmentation masks
                # F.interpolate是resize
                # print("++++++++++++++++++", content_segs_resize[i].shape, content_seg_height, content_seg_width)
                content_segs_resize[i] = F.interpolate(content_segs_resize[i],
                                                       size=(content_seg_height, content_seg_width),
                                                       mode='bilinear')  # resize the content mask with bilinear interpolation
                style_segs_resize[i] = F.interpolate(style_segs_resize[i], size=(style_seg_height, style_seg_width),
                                                     mode='bilinear')  # resize the style mask with bilinear interpolation

        # 分析：如果当前层是卷积层，就手动将content_segs和style_segs进行一次平均池化（？）
        elif "conv" in layer_name:  # if the current layer is a convolutional layer
            for i in range(len(content_segs_resize)):  # loop over the segmentation masks
                # F.pad将矩阵周围填充以保证使用池化之后形状不变，都为torch.Size([1, 1, 224, 224])
                # print(content_segs_resize[i].shape)
                content_segs_resize[i] = F.avg_pool2d(
                    F.pad(content_segs_resize[i], (1, 1, 1, 1), mode='constant'), kernel_size=3,
                    stride=1)  # apply average pooling with padding to the content mask
                style_segs_resize[i] = F.avg_pool2d(
                    F.pad(style_segs_resize[i], (1, 1, 1, 1), mode='constant'), kernel_size=3,
                    stride=1)  # apply average pooling with padding to the style mask

        # 如果这个层是第layer_index个需要用来计算风格损失的层
        if layer_name == style_layers_var_name[layer_index]:
            # variable/conv1_1/Relu:0 ['variable/conv1_1/Relu:0', 'variable/conv2_1/Relu:0', 'variable/conv3_1/Relu:0', 'variable/conv4_1/Relu:0', 'variable/conv5_1/Relu:0']
            # print(layer_name, style_layers_var_name)
        # if the current layer is one of the layers to compute style loss
            print("Setting up style layer: <{}>".format(layer_name))  # print the layer name
            const_layer = const_layers[layer_index]  # get the feature tensor of the style image at this layer
            content_const_layer = content_const_layers[layer_index]  # get the feature tensor of the content image at this layer
            # var_layer = var_layers[layer_index]  # get the feature tensor of the variable image at this layer
            # 使用str拼接成变量名
            var_layer = eval(f"var_layer{layer_index}")

            layer_index = layer_index + 1  # increment the layer index

            layer_style_loss = 0.0  # initialize the style loss at this layer as zero
            # content_segs, style_segs是两个list，且第0个元素有内容，第1个元素是全0
            for content_seg, style_seg in zip(content_segs_resize, style_segs_resize):  # loop over each pair of segmentation masks
                gram_matrix_var = gram_matrix(
                    var_layer * content_seg)  # compute the Gram matrix of the variable image feature multiplied by the content mask

                content_mask_mean = torch.mean(content_seg)  # compute the mean value of the content mask
                # print(content_mask_mean)  # 单个数值
                gram_matrix_var = torch.where(content_mask_mean > 0.,
                                              gram_matrix_var / (torch.numel(var_layer) * content_mask_mean),
                                              gram_matrix_var)  # normalize the Gram matrix by dividing by the number of elements and the mean value if it is positive

                cur_style_mask_mean = torch.mean(style_seg)  # compute the mean value of the current style mask
                # print(cur_style_mask_mean)
                # 当整张风格图像全是攻击区域时，取content_seg的平均作为style_mask的平均
                style_mask_mean = torch.where(torch.logical_and(content_mask_mean > 0., cur_style_mask_mean == 0.),
                                              torch.mean(content_seg), torch.mean(
                        style_seg))  # use either the mean value of the current style mask or that of the content mask depending on their values

                # print(content_mask_mean, cur_style_mask_mean)  # tensor(0.1756, device='cuda:0') tensor(0., device='cuda:0')
                if(torch.logical_and(content_mask_mean > 0., cur_style_mask_mean == 0.)):
                    cur_const_layer = content_const_layer
                    # [400, 400]
                    gram_matrix_const = gram_matrix(content_const_layer * content_seg)
                else:
                    cur_const_layer = const_layer
                    # [470, 470]
                    gram_matrix_const = gram_matrix(const_layer * style_seg)
                # cur_const_layer = torch.where(torch.logical_and(content_mask_mean > 0., cur_style_mask_mean == 0.),
                #                               content_const_layer,
                #                               const_layer)  # use either the feature tensor of the content image or that of the style image depending on their values
                # gram_matrix_const = torch.where(torch.logical_and(content_mask_mean > 0., cur_style_mask_mean == 0.),
                #                                 gram_matrix(content_const_layer * content_seg), gram_matrix(
                #         const_layer * style_seg))  # compute the Gram matrix of the content image feature or the style image feature multiplied by the corresponding mask
                # torch.numel返回tensor中元素个数
                gram_matrix_const = torch.where(style_mask_mean > 0.,
                                                gram_matrix_const / (torch.numel(cur_const_layer) * style_mask_mean),
                                                gram_matrix_const)  # normalize the Gram matrix by dividing by the number of elements and the mean value if it is positive

                diff_style_sum = torch.mean(torch.square(
                    gram_matrix_const - gram_matrix_var)) * content_mask_mean  # compute the mean squared difference between the two Gram matrices and multiply by the mean value of the content mask
                layer_style_loss = layer_style_loss + diff_style_sum  # add this value to the layer style loss
            # multiply the layer style loss by the weight and append it to the loss styles list
            loss_styles = torch.cat((loss_styles, (layer_style_loss * weight).unsqueeze(0)), dim=0)
    return loss_styles  # return the list of style losses at different layers


def total_variation_loss(output, weight):
    # compute the squared difference between adjacent pixels along the height and width dimensions
    # 由于图像维度顺序的问题，此处修改了取出的像素，torch中图像维度是BCHW
    # print(output.shape)  # torch.Size([1, 3, 400, 400])
    diff_h = (output[:, :, :-1, :-1] - output[:, :, :-1, 1:]) ** 2
    diff_w = (output[:, :, :-1, :-1] - output[:, :, 1:, :-1]) ** 2
    # sum up the differences and divide by 2
    tv_loss = torch.sum(diff_h + diff_w) / 2.0
    # multiply by the weight and return the loss
    return tv_loss * weight


def targeted_attack_loss(pred, orig_pred, target, weight):
    """
    Note:
        balance can be adjusted by user for better results, range in [2,5] is recommended
    Arguments:
        pred {logits} -- Logits output by threat model (input: adv)
        orig_pred {int} --  Original (correct) prediction by threat model
        target {int} -- Target lable assigned by user, range in [0,999]
        weight {float32} -- Attack weight assigned by user (args.attack_weight)

    Returns:
        [type] -- [description]
    """
    balance = 5
    # convert the original prediction and the target to one-hot vectors
    # orig_pred = torch.eye(1000)[orig_pred].to(device)
    # target = torch.eye(1000)[target].to(device)
    orig_pred = torch.as_tensor(orig_pred).to(device)
    target = torch.as_tensor(target).to(device)
    # compute the negative cross entropy loss between the original prediction and the logits
    loss1 = -F.cross_entropy(pred[0], orig_pred)
    # compute the positive cross entropy loss between the target and the logits
    loss2 = F.cross_entropy(pred[0], target)
    # print(loss1, loss2)
    # sum up the weighted losses and return the attack loss
    loss_attack = torch.sum(balance * loss2 + loss1) * weight
    return loss_attack


def untargeted_attack_loss(pred, orig_pred, weight):
    """
    Arguments:
        pred {logits} -- Logits output by threat model (input: adv)
        orig_pred {int} -- Original (correct) prediction by threat model
        weight {float32} -- Attack weight assigned by user (args.attack_weight)

    Returns:
        untargeted_attack_loss
    """
    # convert the original prediction to a one-hot vector
    # orig_pred = torch.eye(1000)[orig_pred]
    orig_pred = torch.as_tensor(orig_pred).to(device)
    # compute the negative cross entropy loss between the original prediction and the logits
    loss1 = -F.cross_entropy(pred[0], orig_pred)
    # multiply by the weight and return the attack loss
    loss_attack = loss1 * weight
    return loss_attack


def attack(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prepare input images
    content_image, style_image, content_seg, style_seg = load_imgs_from_path(args)
    # T.ToPILImage()(content_image[0]).show()

    content_image = content_image.to(device)
    ae = content_image
    ae.requires_grad = True
    style_image = style_image.to(device)
    # content_seg = content_seg.to(device)
    # style_seg = style_seg.to(device)
    # content_masks, style_masks = load_seg(content_seg, style_seg)

    # input_image = torch.tensor(content_image, requires_grad=True).to(device)
    # input_image = content_image

    # get the output of the last layer for classification
    # resized_content_image = T.Resize((224, 224))(torch.tensor(content_image))

    # 先把图片输入模型获取初始输出
    vgg_const = Vgg19("./vgg19/vgg19-renamed.pth").to(device).eval()
    resized_content_image = F.interpolate(content_image, size=(224, 224)).to(device)
    prob = vgg_const(resized_content_image)
    pred = print_prob(prob[0], './synset.txt')
    # 设置真实标签
    args.true_label = torch.argmax(prob)

    # get the output of the intermediate layers for style transfer
    # content_fv, *style_fvs_c = vgg_const.fprop(resized_content_image, include_top=False)
    # content_layer_const = content_fv
    # style_layers_const_c = [fv for fv in style_fvs_c]

    # _, *style_fvs = vgg_const.fprop(style_image, include_top=False)
    # style_layers_const = [fv for fv in style_fvs]
    del vgg_const

    # # get the output of the intermediate layers for the input image
    # vgg_var = Vgg19()
    # # content_layer_var, style_layers_var是层的输出，而非层本身
    # _, style_layers_var0, style_layers_var1, style_layers_var2, \
    #     style_layers_var3, style_layers_var4 = vgg_var.fprop(ae, include_top=False)

    # layer_structure_all = ['variable/conv1_1/Relu:0', 'variable/conv1_2/Relu:0', 'variable/pool1:0',
    #                        'variable/conv2_1/Relu:0', 'variable/conv2_2/Relu:0', 'variable/pool2:0',
    #                        'variable/conv3_1/Relu:0', 'variable/conv3_2/Relu:0', 'variable/conv3_3/Relu:0',
    #                        'variable/conv3_4/Relu:0', 'variable/pool3:0', 'variable/conv4_1/Relu:0',
    #                        'variable/conv4_2/Relu:0', 'variable/conv4_3/Relu:0', 'variable/conv4_4/Relu:0',
    #                        'variable/pool4:0', 'variable/conv5_1/Relu:0']
    # style_layers_var_name = ['variable/conv1_1/Relu:0', 'variable/conv2_1/Relu:0', 'variable/conv3_1/Relu:0',
    #                          'variable/conv4_1/Relu:0', 'variable/conv5_1/Relu:0']
    # Option loss: Content Loss
    # loss_content = content_loss(content_layer_const, content_layer_var, float(args.content_weight))

    # Style Loss
    # loss_styles_list = style_loss(layer_structure_all, style_layers_const, style_layers_const_c, style_layers_var0,
    #                               style_layers_var1, style_layers_var2, style_layers_var3, style_layers_var4,
    #                               content_masks, style_masks, float(args.style_weight), style_layers_var_name)

    # loss_style = torch.as_tensor(0.0).to(device)
    # for loss in loss_styles_list:

    optimizer = torch.optim.Adam([ae], lr=args.learning_rate)
    vgg_official = models.vgg19(pretrained=True).features
    # freeze VGG params to avoid change
    for param in vgg_official.parameters():
        param.requires_grad_(False)
    vgg_official.to(device)
    content_features = get_features(content_image, vgg_official)
    for i in tqdm(range(args.max_iter + 1)):
        # get the output of the intermediate layers for the input image
        # vgg_var = Vgg19("./vgg19/vgg19-renamed.pth")
        # content_layer_var, style_layers_var是层的输出，而非层本身
        # _, style_layers_var0, style_layers_var1, style_layers_var2, \
        #     style_layers_var3, style_layers_var4 = vgg_var.fprop(ae, include_top=False)

        # Style Loss
        # loss_styles_list = style_loss(layer_structure_all, style_layers_const, style_layers_const_c, style_layers_var0,
        #                               style_layers_var1, style_layers_var2, style_layers_var3, style_layers_var4,
        #                               content_masks, style_masks, float(args.style_weight), style_layers_var_name)
        # loss_style = loss_style + loss_styles_list[j]
        # loss_style = torch.as_tensor(0.0).to(device)
        # for k in range(5):
        #     loss_style = loss_style + loss_styles_list[k]
        # content_width, content_height = content_image.shape[-1], content_image.shape[-2]
        # physical_adaptor = Physical_Adaptor(args, content_seg, content_image, input_image, content_width,
        #                                     content_height)
        ae.requires_grad = True
        # loss_styles_list：style 1~5 loss
        # 将loss_styles_list的前n项和作为loss_style

        # with torch.no_grad():  # disable gradient computation for the attack model
        vgg_attack = Vgg19("./vgg19/vgg19-renamed.pth").to(device).eval()

        # with torch.no_grad():
        # vgg_attack(physical_adaptor.resized_img)
        # pred = vgg_attack.logits
        # pred = vgg_attack(physical_adaptor.resized_img)

        pred = vgg_attack(ae)
        if args.targeted_attack == 1:
            # print(pred[0], pred[0].shape)  # [1, 1000]
            loss_attack = targeted_attack_loss(pred=pred, orig_pred=args.true_label, target=args.target_label,
                                               weight=args.attack_weight)
        else:
            loss_attack = untargeted_attack_loss(pred=pred, orig_pred=args.true_label, weight=args.attack_weight)

        # output_image = ae.squeeze(0)  # remove the batch dimension

        # loss_tv = total_variation_loss(input_image, float(args.tv_weight))
        loss_tv = total_variation_loss(ae, float(args.tv_weight))

        # get the output of the intermediate layers for the input image
        # vgg_var = Vgg19()
        # content_layer_var, style_layers_var是层的输出，而非层本身
        # content_layer_var, style_layers_var0, style_layers_var1, style_layers_var2, \
        #     style_layers_var3, style_layers_var4, = vgg_var.fprop(ae, include_top=False)

        target_features = get_features(ae, vgg_official)

        # loss_content = content_loss(content_features['conv4_2'], target_features['conv4_2'], float(args.content_weight))
        loss_content = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2) * args.content_weight


        loss_style_2 = style_loss2(model=vgg_official
                                   , style_img=style_image, target_img=ae,
                                   style_loss_weight=float(args.style_weight))

        # total_loss = loss_tv + loss_content + loss_style + loss_attack
        total_loss = loss_content + loss_style_2 + loss_attack

        # optimizer = torch.optim.Adam([input_image], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        # physical_adaptor.resized_img = physical_adaptor.resized_img.detach()

        # physical_adaptor.resized_img = physical_adaptor.resized_img.detach()
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)  # compute the gradients for the input image
        grad = ae.grad.data
        ae = ae - (args.learning_rate)*grad
        # optimizer.step()  # update the input image
        ae = ae.detach()
        ae.requires_grad = True

        prob = vgg_attack(ae)  # get the prediction for the updated input image
        pred, prob = print_prob(prob[0], './synset.txt')
        print('Iteration {} / {}\n\tContent loss: {}'.format(i, args.max_iter, loss_content.item()))
        # for j, style_loss_ in enumerate(loss_styles_list):
        #     print('\tStyle {} loss: {}'.format(j + 1, style_loss_.item()))
        print('\tStyle loss_2: {}'.format(loss_style_2.item()))
        print('\tTV loss: {}'.format(loss_tv.item()))
        print('\tAttack loss: {}'.format(loss_attack.item()))
        print('\tTotal loss: {}'.format(total_loss.item() - loss_tv.item()))
        print('\tloss: {}'.format(total_loss.item()))
        print('\tCurrent prediction: {}'.format(pred))
        del total_loss, loss_tv, loss_attack, loss_content

        if i % args.save_iter == 0:
            content_image_name = args.content_image_path.split('/')[-1].split('.')[0]
            suc = 'non'
            if args.targeted_attack == 1:
                if pred == args.target_label:
                    suc = 'suc'
            else:
                if pred != args.true_label:
                    suc = 'suc'
            final = im_convert(ae)
            matplotlib.image.imsave(os.path.join(args.serial, suc + f'_{i}.jpg'), final)
            # save_result(output_image, os.path.join(args.serial, suc + f'_{i}.jpg'))

