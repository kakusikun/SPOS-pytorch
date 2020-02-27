import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from database.transform.base_transform import BaseTransform

class Tensorize(BaseTransform):
    '''
    To transform the data to tensor with scale [0, 1]
    '''    
    def apply_image(self, img):
        '''
        transform image to tensor
        Args:
            img (PIL image, numpy.ndarray): image to be transformed into tensor
        Return:
            img (torch.Tensor): tensor
        '''
        img = TF.to_tensor(img)
        s = {'state': None}
        return img, s
