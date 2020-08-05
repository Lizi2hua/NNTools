import numpy as np
import os
import json
import traceback
from PIL import Image
from  draw_rectangle import *
from IoU import  *

#配置信息
anno_file=r"C:\Users\Administrator\Desktop\test_data\label"
img_file=r"C:\Users\Administrator\Desktop\test_data\img"
save_path=r"C:\Users\Administrator\Desktop\test_data"
face_size=[12,24,48]
label_txt=r'C:\Users\Administrator\Desktop\test_data\label.txt'

# 检查文件是否存在
for size in face_size:
    if os.path.exists(os.path.join(save_path,str(size))):
        raise FileExistsError('File already exist,please delete!')
#生成图片
for size in face_size:
    # 裁剪的是正方形人脸，这样缩放时人脸的变形小
    print("generate {}*{} size imgae".format(size,size))
    #创建正例，负例，part人脸文件夹
    pos_save_dir=os.path.join(save_path,str(size),"positive")
    part_save_dir=os.path.join(save_path,str(size),"part")
    neg_save_dir=os.path.join(save_path,str(size),"negative")

    for dir_path in [pos_save_dir,part_save_dir,neg_save_dir]:
        os.makedirs(dir_path)

    # 样本标本用‘.txt’文件存
    pos_anno_file=os.path.join(save_path,str(size),"positive.txt")
    part_anno_file=os.path.join(save_path,str(size),"part.txt")
    neg_anno_file=os.path.join(save_path,str(size),"negative.txt")
    # 计数器，给标签和文件命名
    positive_count=0
    negative_count=0
    part_count=0
    # 以“write"的方式打开文件
    try:
        pos_anno_file=open(pos_anno_file,"w")
        part_anno_file=open(part_anno_file,"w")
        neg_anno_file=open(neg_anno_file,"w")

        for i,line in enumerate(open(label_txt)):
            try:
                strs=line.split()
                img_path=strs[0]
                img_path=os.path.join(img_file,img_path)

                with Image.open(img_path) as img:
                    img_w,img_h=img.size
                    topx=int(strs[1])
                    topy=int(strs[2])
                    bottonx=int(strs[3])
                    bottony=int(strs[4])
                    w=bottonx-topx
                    h=bottony-topy
                    # rect=[1.0,topx,topy,bottonx,bottony]
                    # draw_rect(img_path,[rect])

                    if max(w,h)<40 \
                            or topx<0 or topy<0 or w<0 or h<0:
                    #图片最大48*48，所以应该切到40或以上,不满足的舍弃，下面的保证数据是准确的
                            continue
                    box=[topx,topy,bottonx,bottony]
                    #计算人脸中心坐标
                    cx=topx+w/2#cx:center x
                    cy=topy+h/2#cy:center y
                    # 生成部分人脸和人脸的图片
                    crop_boxes=[]
                    resized_faces=[]
                    for _ in range(10):
                        w_=np.random.randint(-w*0.3,w*0.3)#横向随机偏移0.2w
                        h_=np.random.randint(-h*0.3,h*0.3) #纵向随机偏移0.2h
                        cx_=cx+w_
                        cy_=cy+h_
                        #生成与偏移后的中心点的偏移值，使用该偏移值作为框的宽高（w=h)
                        offset_len=np.random.randint(int(min(w,h)*0.8),np.ceil(max(w,h)*1.25))
                        x1_=np.max(cx_-offset_len/2,0)#防止框的左上角出现在图片左上角之外
                        y1_=np.max(cy_-offset_len/2,0)
                        x2_=x1_+offset_len
                        y2_=y1_+offset_len

                        crop_box=[x1_,y1_,x2_,y2_]
                        crop_boxes.append(crop_box)
                        # rect=[1.0,x1_,y1_,x2_,y2_]
                        # draw_rect(img_path,[rect])
                        #从原图裁切
                        crop_face=img.crop(crop_box)
                        #缩放到12*12，24*24，48*48
                        resized_face=crop_face.resize((size,size),Image.ANTIALIAS)
                        resized_faces.append(resized_face)
                    #
                    print(crop_boxes)
                    print(resized_faces)
                    # exit()
                    IOUs=iou(box,np.array(crop_boxes))
                    print(IOUs)
                    exit()
https://pan.baidu.com/s/1RE_bnW8_I2uzIYnInGXBsw#list/path=%2Fsharelink3586954125-871700781165905%2F20%E5%B9%B405%E6%9C%88%E7%8F%AD%E7%AC%AC19%E6%9C%9F%2F20200804_mtcnn01&parentPath=%2Fsharelink3586954125-871700781165905




            except Exception as e:
                traceback.print_exc()





    except Exception as e:
        traceback.print_exc() #获取异常信息

    finally:
        pos_anno_file.close()
        neg_anno_file.close()
        part_anno_file.close()



