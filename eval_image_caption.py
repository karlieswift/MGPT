"""
@Env: /anaconda3/python3.10
@Time: 2023/7/3-19:54
@Auth: karlieswift
@File: eval_image_caption.py
@Desc: 
"""
"""
@Env: /anaconda3/python3.10
@Time: 2023/7/3-19:54
@Auth: karlieswift
@File: eval.py
@Desc: 
"""
import argparse
import torch
import os

from mgpt.common.config import Config
from mgpt.common.dist_utils import get_rank
from mgpt.common.registry import registry
from mgpt.conversation.conversation import Chat, CONV_VISION

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path",default='eval_configs/vitopt_eval.yaml', help="path to configuration file.")
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

path='./eval_pics'
for pic_path in os.listdir(path):
    image_path=os.path.join(path,pic_path)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='{}'.format(device))
    img_list = []
    chat_state = CONV_VISION.copy()
    chat.upload_img(image_path, chat_state, img_list)
    text_input = "please describle this picture"
    chat.ask(text_input, chat_state)
    output_text=chat.answer_image_caption(conv=chat_state,img_list=img_list,image_path=image_path,max_new_tokens=100,max_length=2000)[0]
    # output_text,_=chat.answer(conv=chat_state,img_list=img_list,max_new_tokens=100,max_length=2000)[0]
    print("picture {} image caption：".format(image_path), output_text)
