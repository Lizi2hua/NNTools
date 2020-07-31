import numpy as np
# IoU in Rectangle Condition
def iou(box1,box2):
    #[top_left_x,top_left_y,botton_right_x,botton_right_y]
    size1=box1.size
    size2=box2.size
    assert size1==4 and size2==4,\
        "coordinates size must be 4,but got boxsize1{},boxsize2{}.format(size1,size2)"
    assert box1[0]<=box1[2] and box1[1]<=box1[3] and box2[0]<=box2[2] and box2[1]<=box2[3],\
        "top_xtop_y's value must smaller than or equals to botton_xbotton_y's value"

    #compute intersection area
    inter_top_left_x=max(box1[0],box2[0])
    inter_top_left_y=max(box1[1],box2[1])
    inter_botton_right_x=min(box1[2],box2[2])
    inter_botton_right_y=min(box1[3],box2[3])
    # judge box1 and box2 whehter have intersection
    if inter_top_left_x >= inter_botton_right_x\
        and inter_top_left_y >=inter_botton_right_y:
        inter_area=0
    else:
        inter_area=(inter_botton_right_x-inter_top_left_x)*(inter_botton_right_y-inter_top_left_y)

    #computer union area union_area=box1_area+box2_area-inter_area
    box1_area=(box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area=box1_area+box2_area-inter_area

    return inter_area,union_area,inter_area/union_area



