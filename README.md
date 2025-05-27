# MMDET
本项目实现了神经网络期中作业任务2：在VOC数据集上训练并测试模型 Mask R-CNN 和 Sparse R-CNN。

## 环境准备
安装[mmdetection](https://github.com/open-mmlab/mmdetection)官方指南安装mmdetection 3.3.0。  
进入mmdetection 文件夹 `cd mmdetection`

## 数据准备
下载[VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)和[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)并解压至data文件夹，可以看到VOCdevkit文件夹（可以直接用于Sparse R-CNN训练测试），但
对于Mask R-CNN则需要转成带标注的COCO格式，执行
```
cd data
mkdir VOC0712COCO
cd ..
python tools/dataset_converters/pascal_voc.py data/VOCdevkit -o data/VOC0712COCO/annotations --out-format coco
```
并把VOCdevkit里对应的图片(jpg)文件复制到`VOC0712COCO/imgs`文件夹下，文件结构如下:
```
VOCO0712COCO/
│
├── annotations/ # json files of coco format
└── imgs/ 
    ├── VOC2007
    │  └── JPEGImages
    │    └── 000001.jpg
    └── VOC2012
      └── JPEGImages
        └── 000001.jpg    
```


## 训练与测试
### 训练
**Mask R-CNN训练**   
- 在configs下准备相关训练文件configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc0712.py
- 执行训练脚本`bash tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc0712.py 8`  

**Sparse R-CNN训练**  
- 在configs下准备相关训练文件configs/sparse_rcnn/sparse-rcnn_r50_fpn_2x_voc0712.py
- 执行训练脚本`bash tools/dist_train.sh configs/sparse_rcnn/sparse-rcnn_r50_fpn_2x_voc0712.py 8`
  
### 测试
下载训练好的模型权重并放置到work_dirs文件夹下  
**Mask R-CNN测试**   
执行
```
python tools/test.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc0712.py work_dirs/mask_rcnn.pth --show-dir work_dirs/show_dir/mask_rcnn
```

**Sparse R-CNN测试**  
执行
```
python tools/test.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_2x_voc0712.py work_dirs/sparse_rcnn.pth --show-dir work_dirs/show_dir/sparse_rcnn
```

### 模型权重
模型权重下载：[link](https://pan.baidu.com/s/1UVzUoa4DzH7Bn0tm7u6LVA?pwd=2tds) 提取码：2tds

