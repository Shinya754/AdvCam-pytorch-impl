import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import os
import math


class Physical_Adaptor():
    def __init__(self, args, content_seg, content_image, input_image, content_width, content_height):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # mask the content image and the style image with the segmentation masks
        # content_seg以外是内容图像，内部则是风格迁移后的图像
        masked_content = content_image * (1 - content_seg)
        masked_style = input_image * content_seg
        # add the masked images to get the transformed image
        self.transformed_image = masked_content + masked_style
        # create a placeholder for the background image
        self.background = torch.zeros((1, 3, content_height, content_width))
        # overlay the transformed image on the background image with random transformation
        self.img_with_bg = self.img_random_overlay(self.background, content_seg, content_width, self.transformed_image)
        # clip the pixel values to 0-255 range
        # print(self.img_with_bg)
        # self.img_with_bg = torch.clamp(self.img_with_bg, 0.0, 255.0)
        # resize the image to (224, 224), F.interpolate是张量的缩放
        self.resized_img = F.interpolate(self.img_with_bg, size=(224, 224), mode='bilinear')
        # get the background path from args
        self.bg_path = args.background_path

    def _transform_vector(self, width, x_shift, y_shift, im_scale, rot_in_degrees):
        """
            If one row of transforms is [a0, a1, a2, b0, b1, b2(, c0, c1)],
            then it maps the output point (x, y) to a transformed input point
            (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
            where k = c0 x + c1 y + 1.
            The transforms are inverted compared to the transform mapping input points to output points.
            """
        rot = float(rot_in_degrees) / 90. * (math.pi / 2)
        # Standard rotation matrix
        # (use negative rot because torchvision.transforms.functional.affine will do the inverse)
        rot_matrix = np.array([[math.cos(-rot), -math.sin(-rot)], [math.sin(-rot), math.cos(-rot)]])

        # Scale it
        # (use inverse scale because torchvision.transforms.functional.affine will do the inverse)
        inv_scale = 1. / im_scale
        xform_matrix = rot_matrix * inv_scale
        a0, a1 = xform_matrix[0]
        b0, b1 = xform_matrix[1]

        # At this point, the image will have been rotated around the top left corner,
        # rather than around the center of the image.
        #
        # To fix this, we will see where the center of the image got sent by our transform,
        # and then undo that as part of the translation we apply.
        x_origin = float(width) / 2
        y_origin = float(width) / 2

        x_origin_shifted, y_origin_shifted = np.matmul(xform_matrix, np.array([x_origin, y_origin]), )

        x_origin_delta = x_origin - x_origin_shifted
        y_origin_delta = y_origin - y_origin_shifted

        # Combine our desired shifts with the rotation-induced undesirable shift
        a2 = x_origin_delta - (x_shift / (2 * im_scale))
        b2 = y_origin_delta - (y_shift / (2 * im_scale))

        # Return these values in the order that torchvision.transforms.functional.affine expects
        return np.array(
            [a0 * 180 / math.pi, a1 * 180 / math.pi, b0 * 180 / math.pi, b1 * 180 / math.pi, a2, b2]).astype(np.float32)

    def _random_transformation(self, min_scale, width, max_rotation):
        """Random resize and rotation.
            此处用到的几个参数：
                min_scale: 0.4
                width: 400
                max_rotation: 25

        Arguments:
            min_scale {float32} -- Minimize scale of adv compared to background (supposed the scale of background as 1)
            width {float32} -- Width of adv.
            max_rotation {float32} -- Max rotation degree of adv.

        """

        # sample a random scale from [min_scale, 0.6]
        im_scale = np.random.uniform(low=min_scale, high=0.6)

        # calculate the padding after scaling
        padding_after_scaling = (1 - im_scale) * width

        # sample a random horizontal and vertical shift from the padding range
        x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)/2
        y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)/2

        # sample a random rotation from [-max_rotation, max_rotation]
        rot = np.random.uniform(-max_rotation, max_rotation)

        shear = np.random.uniform(-10, 10)

        transform_list = [rot, x_delta, y_delta, im_scale, shear]

        return transform_list

        # generate the transformation parameters
        # 此处进行修改：列表内改为[角度，平移x，平移y，缩放尺寸，投影]，即固定长度为5
        # return self._transform_vector(width,
        #                               x_shift=x_delta,
        #                               y_shift=y_delta,
        #                               im_scale=im_scale,
        #                               rot_in_degrees=rot)

    def select_random_background(self, content_height, content_width):
        """"
        The function return a random background from specified path.
        """
        # bg_dic = {'t-shirt':'./background/t-shirt','traffic': './physical-attack-data/background/traffic_bg','banana':'./physical-attack-data/background/banana'}
        files = os.listdir(self.bg_path)  # get the list of files in the background path
        rand_num = np.random.randint(0, len(files))  # sample a random index
        file_name = os.path.join(self.bg_path, files[rand_num])  # get the file name of the selected background
        bg = np.array(Image.open(file_name).convert("RGB").resize((content_height, content_width)),
                      dtype=np.float32)  # load the background image as a numpy array and resize it
        # bg = bg  # not sure why this line is needed
        bg = np.expand_dims(bg, 0)  # add a batch dimension to the background array
        return torch.from_numpy(bg)  # convert the numpy array to a torch tensor and return it

    def img_random_overlay(self, bg, img_mask, width, adv_img, min_scale=0.4, max_rotation=25):
        """create an adv with background
            创建img_with_bg，在这个函数里会进行图像的转换

        Arguments:
            bg {tensor} -- selected background
            img_mask {tensor} -- Rotation and resize
            width {float32} -- Width of img
            adv {tensor} -- adversarial_img

        Keyword Arguments:
            min_scale {float} -- [description] (default: {0.4})
            max_rotation {int} -- [description] (default: {25})

        Returns:
            [type] -- [description]
        """
        bg = torch.squeeze(bg, [0]).to(self.device)  # remove the batch dimension of the background tensor
        adv_img = torch.squeeze(adv_img, [0])  # remove the batch dimension of the adversarial image tensor
        # print(bg.shape, adv_img.shape)  # torch.Size([3, 400, 400]) torch.Size([3, 400, 400])
        random_xform_list = self._random_transformation(min_scale, width,max_rotation)  # get a random transformation vector as a torch tensor
        # print(img_mask.shape)  # torch.Size([1, 3, 400, 400])
        output = T.functional.affine(adv_img, angle=random_xform_list[0],
                                  translate=[random_xform_list[1], random_xform_list[2]],
                                  scale=random_xform_list[3], shear=random_xform_list[4])  # apply the affine transformation to the adversarial image tensor and permute the dimensions to (channel, height, width)
        input_mask = T.functional.affine(img_mask, angle=random_xform_list[0],
                                  translate=[random_xform_list[1], random_xform_list[2]],
                                  scale=random_xform_list[3], shear=random_xform_list[4])  # apply the affine transformation to the segmentation mask tensor and permute the dimensions to (channel, height, width)
        background_mask = 1 - input_mask  # get the inverse of the segmentation mask tensor
        input_with_background = (background_mask * bg) + (
                input_mask * output)  # overlay the transformed adversarial image on the background image with masking

        # For simulating lightness change
        color_shift_input = input_with_background + input_with_background * torch.tensor(
            np.random.uniform(-0.3, 0.3))  # add some random noise to simulate lightness change
        img_with_bg = color_shift_input  # add a batch dimension and permute the dimensions to (batch, height, width, channel)
        return img_with_bg  # return the image with background overlay
