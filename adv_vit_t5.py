"""
@Env: /anaconda3/python3.10
@Time: 2023/7/6-20:51
@Auth: karlieswift
@File: adv_vit_t5.py
@Desc: 
"""

import argparse
import torch
from PIL import Image
from mgpt.common.config import Config
from mgpt.common.registry import registry
from mgpt.conversation.conversation import Chat
from mgpt.utils.del_files import del_files
from mgpt.utils.save_pic import tensor2pic

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

del_files('./adv_pics')


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/vitt5_eval.yaml', help="path to configuration file.")
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

alpha = 100
epsilon = 0.6

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='{}'.format(device))
orgin_image = None
raw_image = Image.open('./dog.jpg')
for epoch in range(1, 101):
    if epoch > 1:
        raw_image = Image.open('./adv_pics/{}.jpg'.format(epoch - 1))
    image = vis_processor(raw_image).unsqueeze(0).to('cuda')
    image.requires_grad = True
    samples = {'image': image, 'text_input': ['What is the weather like today?'],
               'text_output': ['Young people in the rain.'],
               'image_id': torch.tensor([0]), 'epoch': 0, 'num_iters_per_epoch': 50, 'iters': 5}

    loss = -model(samples)["loss"]
    loss.backward()
    print("loss", loss)
    if epoch == 1:
        orgin_image = image
        tensor2pic(x=orgin_image.detach().cpu().squeeze(0), path='./adv_pics/{}.jpg'.format("orgin"), std=std,
                   mean=mean)
    grad = image.grad
    adv_images = image + alpha * grad.sign()
    # adv_images=torch.clamp(adv_images,min=a.min(),max=a.max())
    adv_images = torch.max(torch.min(adv_images, orgin_image + epsilon), orgin_image - epsilon)

    tensor2pic(x=adv_images.detach().cpu().squeeze(0), path='./adv_pics/{}.jpg'.format(epoch), std=std, mean=mean)
    model.eval()
    generated_text = model.generate({'image': image, 'prompt': ['What is unusual about this image?']})

    print("第{}次攻击: 生成内容:{}".format(epoch, generated_text))
