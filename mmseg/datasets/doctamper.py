from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class DocTamperDataset(CustomDataset):
    """DocTamper dataset."""

    # 类别名字，0是背景/真实，1是篡改区域
    CLASSES = ('background', 'tampered')

    # 对应的可视化调色板（黑色和白色）
    PALETTE = [[0, 0, 0], [1, 1, 1]]

    def __init__(self, **kwargs):
        super(DocTamperDataset, self).__init__(
            img_suffix='.jpg',       # 请根据 DocTamper 原图格式修改 (如 .jpg 或 .png)
            seg_map_suffix='.png',   # 请根据 DocTamper 标签格式修改 (必须是灰度图，值为0和1)
            #reduce_zero_label=False, # 二分类通常不需要 reduce_zero_label
            **kwargs)