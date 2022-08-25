from mmseg.apis import init_segmentor
from mmcv import Config
from mmseg.models import build_segmentor

# def hrnet(pretrained):
#     config_file = '/home/liuxiangyu/mmsegmentation/configs/ocrnet/ocrnet_hr48_512x512_80k_ade20k.py'
#     checkpoint_file = '../checkpoints/ocrnet_hr48_512x512_80k_ade20k_20200615_021518-d168c2d1.pth'
#     # build the model from a config file and a checkpoint file
#     model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
#     return model

def hrnet(pretrained):
    cfg = Config.fromfile('/home/liuxiangyu/mmsegmentation/configs/ocrnet/ocrnet_hr48_512x512_80k_ade20k.py')
    cfg.load_from = '../checkpoints/ocrnet_hr48_512x512_80k_ade20k_20200615_021518-d168c2d1.pth'
    model = build_segmentor(cfg.model)
    return model

