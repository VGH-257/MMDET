_base_ = [
    '../_base_/models/rpn_r50_fpn.py', '../_base_/datasets/voc0712coco.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = 'data/VOC0712COCO/'

val_evaluator = dict(type='CocoMetric',
                    ann_file=data_root + 'annotations/voc07_test.json',
                    format_only=False,                     
                    metric='proposal_fast')
test_evaluator = val_evaluator


model = dict(
    type='RPN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)))



# inference on val dataset and dump the proposals with evaluate metric
# data_root = 'data/coco/'
# test_evaluator = [
#     dict(
#         type='DumpProposals',
#         output_dir=data_root + 'proposals/',
#         proposals_file='rpn_r50_fpn_1x_val2017.pkl'),
#     dict(
#         type='CocoMetric',
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         metric='proposal_fast',
#         backend_args={{_base_.backend_args}},
#         format_only=False)
# ]

# inference on training dataset and dump the proposals without evaluate metric
# data_root = 'data/coco/'
# test_dataloader = dict(
#     dataset=dict(
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/')))
#
# test_evaluator = [
#     dict(
#         type='DumpProposals',
#         output_dir=data_root + 'proposals/',
#         proposals_file='rpn_r50_fpn_1x_train2017.pkl'),
# ]