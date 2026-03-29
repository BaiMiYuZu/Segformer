_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/doctamper.py', # 【关键修改】：换成你的数据集配置
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mit_b3',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2, # 【关键修改】：DocTamper 为二分类任务，设为2
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        # 组合损失 BCE + Lovasz (IoU替代)
        loss_decode=[
        # 1. 带权重的交叉熵：给背景极小的权重(例如0.05)，给篡改区域极大的权重(例如0.95)
        dict(type='CrossEntropyLoss',
             use_sigmoid=False,
             loss_name='loss_ce',
             loss_weight=1.0,
             class_weight=[0.05, 0.95]), # 【关键】：强行惩罚模型漏检篡改区域的行为

        # 2. Dice Loss：辅助优化边界和结构
        dict(type='DiceLoss',
             loss_name='loss_dice',
             loss_weight=1.0)]
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


data = dict(samples_per_gpu=2)
evaluation = dict(interval=4000, metric=['mIoU', 'mDice'])
