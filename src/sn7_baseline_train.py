import solaris as sol
import os
import hrnet

hrnet_dict = {
    'model_name': 'hrnet',
    'weight_path': None,
    'weight_url': None,
    'arch': hrnet.hrnet
}

root_dir = '/home/liuxiangyu/CosmiQ_SN7_Baseline/'
config_path = os.path.join(root_dir, 'yml/sn7_baseline_train.yml')
config = sol.utils.config.parse(config_path)
print('Config:')
print(config)

# make model output dir
os.makedirs(os.path.dirname(config['training']['model_dest_path']), exist_ok=True)

trainer = sol.nets.train.Trainer(config=config, custom_model_dict=hrnet_dict)
trainer.train()
