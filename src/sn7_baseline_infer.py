import solaris as sol
import os
import hrnet

hrnet_dict = {
    'model_name': 'hrnet',
    'weight_path': None,
    'weight_url': None,
    'arch': hrnet.hrnet
}


config_path = '../yml/sn7_baseline_infer.yml'
config = sol.utils.config.parse(config_path)
print('Config:')
print(config)

# make infernce output dir
os.makedirs(os.path.dirname(config['inference']['output_dir']), exist_ok=True)

inferer = sol.nets.infer.Inferer(config, custom_model_dict=hrnet_dict)
inferer()
