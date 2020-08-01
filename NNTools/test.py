import torch
import numpy as np
from IoU import iou
from NMS import NMS
from draw_rectangle import draw_rect
# a=np.array([90,10,120,40])
# b=np.array([20,70,60,120])
# i,u,iu=iou(a,b)
# print(i,u,iu)
#
# a=np.array([ 533. ,   41. ,  622.  , 175.])
# b=np.array([ 490   ,67  , 563 ,  251. ])
# i,u,iu=iou(a,b)
# print(i,u,iu)
path="test_img.jpg"
boxes=[[0.97,133,80,225,265],
       [0.89,157,69,261,238],
       [0.85,148,132,238,282],
       [0.70,105,112,195,303],
       [0.69,88,50,187,193],
       [0.70,316,209,378,312],
       [0.50,298,173,348,340],
       [0.90,490,67,563,251],
       [0.70,446,46,526,181],
       [0.79,533,41,622,175],
       [0.85,429,87,619,216]]
box=NMS(boxes,thresh=0.1)
draw_rect(path,boxes)
draw_rect(path,box)