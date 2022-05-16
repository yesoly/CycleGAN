# init_weight, ImagePool, LossDisplayer

import random
import torch
import torch.nn as nn

def init_weight(module):
    class_name = module.__class__.__name__

    if class_name.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm2d") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant(module.bias.data, 0.0)
    
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


class LossDisplayer():
    def __init__(self, name_list):
        self.count = 0
        self.name_list = name_list
        self.loss_list = [0] * len(self.name_list)

    def record(self, losses):
        self.count += 1
        for i, loss in enumerate(losses):
            self.loss_list[i] += loss.item()

    def get_avg_losses(self):
        return [loss / self.count for loss in self.loss_list]

    def display(self):
        for i, total_loss in enumerate(self.loss_list):
            avg_loss = total_loss / self.count
            print(f"{self.name_list[i]}: {avg_loss:.4f}   ", end="")

    def reset(self):
        self.count = 0
        self.loss_list = [0] * len(self.name_list)