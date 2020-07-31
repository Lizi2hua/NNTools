import torch
import numpy as np
from IoU import iou

a=np.array([90,10,120,40])
b=np.array([20,70,60,120])
i,u,iu=iou(a,b)
print(i,u,iu)

a=np.array([10,10,20,20])
b=np.array([15,15,30,30])
i,u,iu=iou(a,b)
print(i,u,iu)
