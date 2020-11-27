# Importing Libraries
import torch
import numpy

# Defining the transforms
class ToTensor(object):

    def __call__(self, image):
        # imagem numpy: C x H x W
        # imagem torch: C X H X W
                        
        image = image.transpose((0, 1, 2))
        return torch.from_numpy(image)

class DuplicateArray(object):
    
    def __call__(self, image):
        image = image.repeat(3, axis=0)
        return image