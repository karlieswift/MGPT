"""
@Env: /anaconda3/python3.10
@Time: 2023/7/3-22:28
@Auth: karlieswift
@File: eval_vit_t5.py
@Desc:
"""
import argparse
import torch
from mgpt.common.config import Config
from mgpt.common.registry import registry
from mgpt.conversation.conversation import Chat, CONV_VISION
import os
from PIL import Image

def answer(vis_processor,model):
    paths = './eval_pics'
    for path in os.listdir(paths):
        image_path = os.path.join(paths, path)
        print(image_path, '---------------------------------------------------------------------')
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        raw_image = Image.open(image_path)
        # raw_image = Image.open('/public/home/mswanghao/project/InstructMiniGPT/166.png')
        # prepare the image
        image = vis_processor(raw_image).unsqueeze(0).to(device)
        text = model.generate({"image": image, "prompt": "What is unusual about this image?"})
        print(text)



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path",default='eval_configs/vitt5_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("-file_index", "--file_index", help="csv_files", type=int, default="1")
    parser.add_argument("-std", "--std", help="std", type=float, default="0.01")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config(args)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_cls.from_config(model_config).to(device)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
answer(vis_processor,model)



#  output_text,_=chat.answer(conv=chat_state,img_list=img_list,max_new_tokens=100,max_length=2000)

