"""

 Morphological skeletonization as implemented for clDice:
 Shit, Suprosanna et al. “clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020): 16555-16564.

"""
import torch
import torch.nn.functional as F

'''class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=5):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)
        
    def soft_erode3D(self,img):
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)

    def soft_avg_pool(self, img):

        if len(img.shape)==4:
            p1 = -F.avg_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.avg_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.avg_pool2d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.avg_pool2d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.avg_pool2d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)
        
    def iter_erode(self,img, num_iter=2):
        for _ in range(num_iter):
            img = self.soft_erode(img)
        return img

    def iter_erode3D(self,img, num_iter=2):
        for _ in range(num_iter):
            img = self.soft_erode(img)
        return img

    def iter_dilate(self,img, num_iter=2):
        for _ in range(num_iter):
            img = self.soft_dilate(img)
        return img

    def iter_dilate3D(self,img, num_iter=2):
        for _ in range(num_iter):
            img = self.soft_dilate3D(img)
        return img

    def soft_dilate(self, img):

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_dilate3D(self, img):
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img, num_iter=None):
        
        #if num_iter:
        #    self.num_iter = num_iter
        if num_iter is not None:
            self.num_iter = int(num_iter)

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def soft_close(self, img):
        return self.soft_erode(self.soft_dilate(img))
        
    def forward(self, img):

        return self.soft_skel(img)'''

import torch
import torch.nn.functional as F
from typing import Optional

class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter: int = 5):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)
        else:
            raise RuntimeError("Unsupported input shape for soft_erode.")

    def soft_erode3D(self, img: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)

    def soft_avg_pool(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape)==4:
            p1 = -F.avg_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.avg_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.avg_pool2d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.avg_pool2d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.avg_pool2d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)
        else:
            raise RuntimeError("Unsupported input shape for soft_avg_pool.")

    def iter_erode(self, img: torch.Tensor, num_iter: int = 2) -> torch.Tensor:
        for _ in range(num_iter):
            img = self.soft_erode(img)
        return img

    def iter_erode3D(self, img: torch.Tensor, num_iter: int = 2) -> torch.Tensor:
        for _ in range(num_iter):
            img = self.soft_erode3D(img)
        return img

    def iter_dilate(self, img: torch.Tensor, num_iter: int = 2) -> torch.Tensor:
        for _ in range(num_iter):
            img = self.soft_dilate(img)
        return img

    def iter_dilate3D(self, img: torch.Tensor, num_iter: int = 2) -> torch.Tensor:
        for _ in range(num_iter):
            img = self.soft_dilate3D(img)
        return img

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))
        else:
            raise RuntimeError("Unsupported input shape for soft_dilate.")

    def soft_dilate3D(self, img: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(img))

    '''def soft_skel(self, img: torch.Tensor, num_iter: Optional[int] = None) -> torch.Tensor:
        if num_iter is not None:
            self.num_iter = int(num_iter)

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel'''
    
    def soft_skel(self, img: torch.Tensor, num_iter: Optional[int] = None) -> torch.Tensor:
        local_num_iter = int(num_iter) if num_iter is not None else self.num_iter

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for _ in range(local_num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def soft_close(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_erode(self.soft_dilate(img))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_skel(img)
