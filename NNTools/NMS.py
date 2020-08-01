import numpy as np
from IoU import iou



def NMS(boxes,thresh=0.2):
       # The input data shoulde be like:[[Cofidence,top_left_x,top_left_y,botton_right_x,botton_right_y],[],[]]
       boxes=np.array(boxes,dtype=float)
       #进行降序排序
       boxes=boxes[np.argsort(boxes[:,0])[::-1]]
       # print(boxes)
       #将boxes第一个元素与剩下元素计算Iou，大于阈值则删除，小于则保留
       for i in range(len(boxes)):
              j=i+1
              # print("i=", i)
              # print("boxed len",len(boxes))
              if i >=len(boxes):
                     break
              compare_base_box=boxes[i][1:]
              # print(max_confidence_box)
              for item in boxes[j:]:
                     # print(j)
                     # print(item)
                     item=item[1:]
                     _,_,IOU=iou(item,compare_base_box)
                     # print(IOU)
                     #因为np.delele删除后的boxes维度会缩小一行， 所以删除时，遍历指针j相当于+1
                     if IOU>=thresh:
                            boxes=np.delete(boxes,j,0)
                            # print('delte!')
                     else:j+=1
                     # print("________")
              # print(boxes)
       return boxes




