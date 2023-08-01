import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_imgs_from_path(args, target_size=(224, 224)):
    in_transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # convert("RGB")的作用是将读取出的四通道图像转换为三通道
    content_image = Image.open(args.content_image_path).convert("RGB").resize(
        target_size)  # content_image is a PIL image of size (400, 400)

    content_width, content_height = content_image.size  # content_width and content_height are both 400
    # content_image = T.ToTensor()(content_image).unsqueeze(
    #     0)  # content_image is a torch tensor of shape (1, 3, 400, 400)
    content_image = in_transform(content_image)[:3, :, :].unsqueeze(0)

    # style_image不需要resize
    style_image = Image.open(args.style_image_path).convert("RGB")  # style_image is a PIL image of variable size
    style_width, style_height = style_image.size  # style_width and style_height are the width and height of the style image
    # style_image = T.ToTensor()(style_image).unsqueeze(
    #     0)  # style_image is a torch tensor of shape (1, 3, style_height, style_width)
    style_image = in_transform(style_image)[:3, :, :].unsqueeze(0)


    content_seg = Image.open(args.content_seg_path).convert("RGB").resize((content_width, content_height),
                                                                          resample=Image.BILINEAR)  # content_seg is a PIL image of size (400, 400)
    content_seg = (T.ToTensor()(content_seg)).unsqueeze(
        0)  # content_seg is a torch tensor of shape (1, 3, 400, 400)
    style_seg = Image.open(args.style_seg_path).convert("RGB").resize((style_width, style_height),
                                                                      resample=Image.BILINEAR)  # style_seg is a PIL image of variable size
    style_seg = (T.ToTensor()(style_seg)).unsqueeze(
        0)  # style_seg is a torch tensor of shape (1, 3, style_height, style_width)
    return content_image, style_image, content_seg, style_seg  # returns four tensors of different shapes


def load_seg(content_seg, style_seg):
    """
    这段代码的目的是根据内容图和风格图的分割图，提取出不同颜色对应的掩码，用于后续的风格迁移。具体来说：
    color_codes是一个列表，存储了两种颜色的名称，分别是UnAttack和Attack，代表不同的区域。
    content_shape和style_shape是两个列表，存储了内容图和风格图的宽度和高度。
    _extract_mask是一个内部函数，接收一个分割图和一个颜色名称作为参数，返回一个二值化的掩码，其中该颜色对应的区域为1，其他区域为0。这个函数通过比较分割图的RGB通道的值来判断颜色，比如UnAttack对应的是黑色，所以RGB通道都小于0.5；Attack对应的是白色，所以RGB通道都大于0.8。
    color_content_masks和color_style_masks是两个空列表，用于存储内容图和风格图的不同颜色掩码。
    for循环遍历color_codes中的每一种颜色，调用_extract_mask函数来生成对应的掩码，并将其扩展为四维张量（增加batch维度和channel维度），然后分别添加到color_content_masks和color_style_masks中。
    最后返回两个列表，每个列表包含两个四维张量，分别对应UnAttack和Attack的掩码。这些掩码可以用于后续的风格迁移算法中，实现不同区域的风格融合。
    """
    color_codes = ['UnAttack', 'Attack']
    # content_shape = [content_seg.shape[1], content_seg.shape[0]]  # content_shape is a list of [400, 400]
    # style_shape = [style_seg.shape[1], style_seg.shape[0]]  # style_shape is a list of [style_height, style_width]

    def _extract_mask(seg, color_str):
        # print(seg.shape) torch.Size([1, 3, 400, 400])
        _, c, h, w = seg.shape  # h, w, c are the height, width and channel of seg
        # TODO: 此处第0维直接取0会不会有问题
        if color_str == "UnAttack":
            mask_r = (seg[0, 0, :, :] < 0.5).to(torch.uint8)  # mask_r is a torch tensor of shape (h, w) with 0 or 1 values
            mask_g = (seg[0, 1, :, :] < 0.5).to(torch.uint8)  # mask_g is a torch tensor of shape (h, w) with 0 or 1 values
            mask_b = (seg[0, 2, :, :] < 0.5).to(torch.uint8)  # mask_b is a torch tensor of shape (h, w) with 0 or 1 values
        elif color_str == "Attack":
            mask_r = (seg[0, 0, :, :] > 0.8).to(torch.uint8)  # mask_r is a torch tensor of shape (h, w) with 0 or 1 values
            mask_g = (seg[0, 1, :, :] > 0.8).to(torch.uint8)  # mask_g is a torch tensor of shape (h, w) with 0 or 1 values
            mask_b = (seg[0, 2, :, :] > 0.8).to(torch.uint8)  # mask_b is a torch tensor of shape (h, w) with 0 or 1 values

        return (mask_r * mask_g * mask_b).to(torch.float32)  # return a torch tensor of shape (h, w) with 0 or 1 values

    color_content_masks = []  # an empty list to store the content masks
    color_style_masks = []  # an empty list to store the style masks
    for i in range(len(color_codes)):
        color_content_masks.append(_extract_mask(content_seg, color_codes[i]).unsqueeze(0).unsqueeze(
            1))  # append a torch tensor of shape (1, 1, h, w) to the list
        color_style_masks.append(_extract_mask(style_seg, color_codes[i]).unsqueeze(0).unsqueeze(
            1))  # append a torch tensor of shape (1, 1, style_height, style_width) to the list
    # color_content_masks = torch.empty(0).to(device)
    # color_style_masks = torch.empty(0).to(device)
    # for i in range(len(color_codes)):
    #     color_content_masks = torch.cat((color_content_masks, _extract_mask(content_seg, color_codes[i]).
    #                                      unsqueeze(0).unsqueeze(1).unsqueeze(2)), dim=0)
    #     color_style_masks = torch.cat((color_style_masks, _extract_mask(style_seg, color_codes[i]).
    #                                      unsqueeze(0).unsqueeze(1).unsqueeze(2)), dim=0)

    return color_content_masks, color_style_masks  # return two lists of tensors


def gram_matrix(activations):
    """
    这个函数的作用是计算一个特征张量的Gram矩阵，即特征之间的内积。Gram矩阵可以反映特征的相关性和分布，用于计算风格损失。具体来说：

    - activations是一个四维张量，表示一个图像在某一层次上的特征，形状为(batch, height, width, channel)。
    - height, width, num_channels是三个整数，表示特征的高度、宽度和通道数。
    - gram_matrix是一个二维张量，表示特征的Gram矩阵，形状为(channel, channel)。

    - 首先，将activations的维度进行置换，使其变为(batch, channel, height, width)。
    - 然后，将activations的形状变为(channel, width * height)，即将每个通道的特征展平为一维向量。
    - 最后，将activations和它的转置进行矩阵乘法，得到Gram矩阵。Gram矩阵的每个元素表示两个通道的特征之间的内积。`
    """
    height = activations.shape[2]  # get the height of the activations
    width = activations.shape[3]  # get the width of the activations
    num_channels = activations.shape[1]  # get the number of channels of the activations

    gram_matrix = activations
    gram_matrix = gram_matrix.reshape(num_channels,
                                      width * height)  # reshape the activations to (channel, width * height)
    gram_matrix = torch.matmul(gram_matrix,
                               gram_matrix.t())  # compute the matrix multiplication of the activations and its transpose
    return gram_matrix  # return the Gram matrix


def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    # Need the layers for the content and style representations of an image
    # 注意，这里为了适应vgg_torch中的layer名字，将layers中字典的key设置为了conv***，而非原来的数字
    if layers is None:
        # layers = {'conv1_1': 'conv1_1',
        #           'conv2_1': 'conv2_1',
        #           'conv3_1': 'conv3_1',
        #           'conv4_1': 'conv4_1',
        #           'conv4_2': 'conv4_2',  ## content representation
        #           'conv5_1': 'conv5_1'}
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        if type(layer) is torch.nn.modules.linear.Linear:
            break
        # print(name, type(layer))
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def style_loss2(model, style_img, target_img, style_loss_weight):
    """
    Args:
        model: vgg19(nn.Module)
        style_img: tensor
        target_img: tensor. This is the AE img.
        var_layer0-5: Layers of vgg19 to get feature map.
        style_loss_weight:

    Returns:
        style_loss(tensor)
    """
    # resize the style_img to the same size as target_img
    style_img = T.Resize(target_img.shape[-2:])(style_img)
    # get content and style features only once before training
    style_features_ = get_features(style_img, model)
    # get the features from your target image
    target_features = get_features(target_img, model)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features_[layer]) for layer in style_features_}
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        # print(style_features_, target_features)
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    return style_loss * style_loss_weight

def save_result(img_, str_):
    """
    Args:
        img_(tensor): 待保存的图像
        str_(string): 文件保存路径

    Returns:
        None
    """
    result = T.ToPILImage()(img_)  # convert the tensor to a PIL image
    result.save(str_)


def print_prob(prob, file_path):
    """
    print_prob函数的作用是根据一个概率向量和一个文件路径，打印出该概率向量对应的类别标签和概率值。具体来说：

    - prob是一个一维的tensor数组，表示一个图像在1000个类别上的概率分布，每个元素的值在0到1之间。
    - file_path是一个字符串，表示一个文件的路径，该文件存储了1000个类别的名称，每行一个名称，按照索引顺序排列。
    - synset是一个列表，存储了从文件中读取的类别名称，每个元素是一个去掉空白符的字符串。
    - pred是一个一维的numpy数组，表示按照概率从大到小排序后的类别索引，每个元素是一个整数。
    - top1是一个字符串，表示概率最大的类别名称，从synset中根据pred[0]索引得到。
    - top5是一个列表，存储了概率最大的五个类别名称和概率值，每个元素是一个元组，从synset和prob中根据pred[i]索引得到。
    - 函数打印出top1和top5，并返回pred[0]和prob[pred[0]]，即概率最大的类别索引和概率值。
    """
    synset = [l.strip() for l in open(file_path).readlines()]  # synset is a list of strings of length 1000

    # print prob
    pred = torch.argsort(prob, descending=True)  # pred is a torch tensor of shape (1000,) with indices of sorted prob

    # Get top1 label
    top1 = synset[pred[0]]  # top1 is a string of the most probable label
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in
            range(5)]  # top5 is a list of tuples of the top 5 labels and probs
    print(("Top5: ", top5))
    return pred[0], prob[pred[0]]  # return the index and prob of the most probable label
