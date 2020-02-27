import torch
import numpy as np
import torchvision.transforms as T
from database.transform.base_transform import BaseTransform

class RandCrop(BaseTransform):
    '''
    To pad and crop the image
    Args:
        size (int, tuple): final size of image
        pad (int): padding of image
    '''    
    def __init__(self, size, pad):
        self.handle = T.RandomCrop(size, pad)

    def apply_image(self, img):
        '''
        To pad and crop the image
        Args:
            img (PIL image): image to be transformed into tensor
        Return:
            img (PIL image)
        '''
        img = self.handle(img)
        s = {'state': None}
        return img, s

    #TODO: bbox and pts
