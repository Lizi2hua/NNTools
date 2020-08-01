
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def draw_rect(img_path,label_data):
#注意图片的坐标是以左上角为原点的，画布也应该以左上角为原点
    img=plt.imread(img_path)
    fig,ax=plt.subplots(1)
    ax.imshow(img)
    for rect_data in label_data:
        top=(rect_data[1],rect_data[2])
        w=rect_data[3]-rect_data[1]
        h=rect_data[4]-rect_data[2]
        rect=patches.Rectangle(top,w,h,linewidth=1,edgecolor='r',fill=False)
        plt.annotate("{}".format(rect_data[0]),top)
        ax.add_patch(rect)
    plt.show()
