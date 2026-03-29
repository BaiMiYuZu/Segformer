# dataset settings
dataset_type = 'DocTamperDataset' # 或者如果你注册了 'DocTamperDataset'，请用对应的名字
data_root = 'data/doctamper/'  # 【注意】：请替换为你 doctamper 数据集的实际根目录

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 核心：和 RTM 一样设定 512x512 的裁剪尺寸，坚决不用 Resize
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # 注意：如果你的篡改 mask 已经是 0(背景)和1(篡改) 的灰度图，无需做其他处理
    dict(type='LoadAnnotations', reduce_zero_label=False),

    # 【RTM 提点的核心操作】：随机裁剪并过滤掉大部分全是背景的区域
    dict(type='Resize', img_scale=(768, 768), ratio_range=(0.75, 1.25), keep_ratio=True),
    dict(
        type='TamperFocusedRandomCrop',
        crop_size=crop_size,
        foreground_index=1,
        min_foreground_pixels=512,
        min_foreground_ratio=0.003,
        num_attempts=20),

    dict(type='RandomFlip', prob=0.5), # 50% 概率随机翻转，增加数据多样性
    dict(type='PhotoMetricDistortion'), # 颜色抖动
    dict(type='RandomRotate', prob=0.3, degree=5, pad_val=0, seg_pad_val=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0), # 如果原图比 512 小，进行边缘填充
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # 【修改 1】：给一个较大的占位尺寸，满足底层的断言要求
        img_scale=(512, 512),
        flip=False,
        transforms=[
            # 【修改 2】：加上 Resize，但务必设为 keep_ratio=True！
            # 这样它只会等比例缩放图片，绝不会把长方形强行压成正方形，最大程度保留真实篡改痕迹。
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    # 【建议】：既然 RTM 把 batch_size 开到了 12，如果你的显存允许，
    # 强烈建议把你原本的 samples_per_gpu=2 改大，例如改为 4、8 甚至 12
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train', # 【注意】：请根据 doctamper 实际的图片存放目录修改
        ann_dir='ann_dir/train', # 【注意】：请根据 doctamper 实际的标签存放目录修改
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline))
