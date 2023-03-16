import os
from PIL import Image
import numpy as np
import tqdm


def main():
    # 数据集通道数
    img_channels = 3
    
    # 数据集路径
    img_dir = "../dataset/train"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    img_name_list = [os.path.join(img_dir,i) for i in os.listdir(img_dir) if i.endswith(".jpg")]
    
    img_dir = "../dataset/test"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    img_name_list.extend([os.path.join(img_dir,i) for i in os.listdir(img_dir) if i.endswith(".jpg")])
    
    it=iter(img_name_list)
    
    # 累计mean和std，三个通道，这里是RGB，PIL库中的Image.open 默认RGB，cv2.imread是BGR
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    # 统计数据集长度
    print(f"INFO: {len(img_name_list)} imgs in total")
    for img_name in tqdm.tqdm(img_name_list,total=len(img_name_list)):
        img_path = next(it)
        # 对数据集进行归一化
        img = np.array(Image.open(img_path)) / 255.
        # 对每个维度进行统计，Image.open打开的是HWC格式，最后一维是通道数
        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")

def m2():
    from torchvision.transforms import ToTensor#用于把图片转化为张量
    import numpy as np#用于将张量转化为数组，进行除法
    from torchvision.datasets import ImageFolder#用于导入图片数据集
    
    means = [0,0,0]
    std = [0,0,0]#初始化均值和方差
    transform=ToTensor()#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
    dataset=ImageFolder("../dataset/",transform=transform)#导入数据集的图片，并且转化为张量
    num_imgs=len(dataset)#获取数据集的图片数量
    for img,a in dataset:#遍历数据集的张量和标签
        for i in range(3):#遍历图片的RGB三通道
            # 计算每一个通道的均值和标准差
            means[i] += img[i, :, :].mean()
            std[i] += img[i, :, :].std()
    mean=np.array(means)/num_imgs
    std=np.array(std)/num_imgs#要使数据集归一化，均值和方差需除以总图片数量
    print(mean,std)#打印出结果




if __name__ == '__main__':
    main()
    m2()

