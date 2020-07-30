import numpy as np
# 矩形框的IoU
def iou(box1,box2):
    size1=box1.size
    size2=box2.size
    assert size1==4 and size2==4,"coordinates size must be 4,but got boxsize1:{},boxsize2:{}".format(size1,size2)
    


a=np.array([1,2,3,4])
b=np.array([1,2,4])

print(a.size)
iou(a,b)




# 不规则框的IoU
