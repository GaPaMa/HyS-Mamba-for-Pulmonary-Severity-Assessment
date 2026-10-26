import os

_data_root = os.environ.get("HYSMAMBA_DATA_ROOT", "data")

project_dir = "outputs/Dino_ViM_RALO"
use_timestamp = True

description = "HyS Vision DaMamba"

train_dataset = dict(
    type="ralo_bouthaina_multi",
    path=_data_root,
    dataset="RALO_Aug",
    subset="train",
    ann_file=os.path.join(_data_root, 'RALO_Aug/lung_rep1.csv'),
    img_dir=os.path.join(_data_root, 'RALO_Aug/Lung_Rep1'),
    val_ratio=0.10,
    val_seed=42,
    aug_label_mode="avg_lr",
    transform = [
          dict(name='Resize', height=224, width=224),
          dict(name='RandomHorizontalFlip', p=0.5),
          dict(name='RandomBrightnessContrast', brightness_limit=0.1, contrast_limit=0.1, p=0.3),
          dict(name='ColorJitter', brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
          dict(name='RandomRotation', limit=5, p=0.2),
          dict(name='GaussianBlur', blur_limit=(3, 5), sigma_limit=(0.1, 1.0), p=0.1),
          dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          dict(name='RandomSquareErasing', max_holes=2, max_height=32, max_width=32, p=0.3),
     ],
)

valid_dataset = dict(
    type="ralo_bouthaina_multi",
    path=_data_root,
    dataset="RALO_Aug",
    subset="val",
    ann_file=os.path.join(_data_root, 'RALO_Aug/lung_rep1.csv'),
    img_dir=os.path.join(_data_root, 'RALO_Aug/Lung_Rep1'),
    val_ratio=0.10,
    val_seed=42,
    aug_label_mode="avg_lr",
    transform=[
        dict(name='Resize', height=224, width=224),
        dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
)

test_dataset = dict(
    type="ralo_bouthaina_multi",
    path=_data_root,
    dataset="RALO_Aug",
    subset="test",
    ann_file=os.path.join(_data_root, 'RALO_Aug/lung_rep1.csv'),
    img_dir=os.path.join(_data_root, 'RALO_Aug/Lung_Rep1'),
    aug_label_mode="avg_lr",
    transform=[
        dict(name='Resize', height=224, width=224),
        dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
)

model=dict(
    type= "dino_coag_damamba_biconvvss_regressor", 
    convnext_kwargs={
        "timm_name": "convnext_tiny",
        "in_chans": 3,
        "pretrained": True,
    },
    vit_kwargs={
        "pretrained_model_name_or_path": "google/vit-base-patch16-224",
     },
    vit_layers= [3, 6, 9, 11],
    inner_dim= 256,
    proj_rank= 256,
    proj_out_dim= 256,
    drop= 0.1,
    drop_path= 0.1,
    head_hidden= 256, #512
    head_drop= 0.1,
    output_layers= [2],
    output_activation="sigmoid",
    output_scale=8.0,
    center_dim= 224,
    freeze_vit= True,
    freeze_convnext= False,
    use_coag=True, # use_coag=False tests without contextual alignment gating.
    fusion_mode="damamba", # fusion_mode="conv" replaces DaMamba with a non-Mamba conv fusion block. "identity" removes semantic-to-spatial fusion at this stage.
    projection_mode="simple", #"biconvvss", # projection_mode="fapm" replaces BiConvVSS with the original adapter FAPM. ="simple" removes BiConvVSS refinement/bidirectionality.
)

trainer = dict(

     batch_size = 32,
     num_workers=8,
     pin_memory=True,
     max_epochs = 80,
     max_grad_norm=1.0,
     gradient_accumulation_steps = 1,
     auto_resume=False,
     early_stopping_patience = 200,
     checkpoints=dict(save_top_k = 15,    
                      ),

    optimizer = dict(
          type = "AdamW",
          lr = 1e-5,
          weight_decay = 0.01,
          use_param_groups=True,
     ),

     scheduler=dict(
          type= "WarmupCosineAnnealingLR",
          T_max = 5000,
          eta_min = 1e-6,
          warmup_steps=500
    ),
     loss = dict(
          type = "multi_loss",
          losses = [
               dict(name = "mae_loss", weight = 0.4),
               dict(name = "mse_loss", weight = 0.6),
          ]
     ),


     metrics = dict(
          type = "lung_severity_assesmnt_metrics",
          metrics = ["mae","pc","cs_0.5","cs_1.0"],
          dim_names= ["GE","LO"],

     )
)


