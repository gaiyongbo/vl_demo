from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2VLModel,Qwen2VLConfig
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import numpy as np
import time
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from qwen_vl_utils import process_vision_info


def test_vit_embedding(vit):

    model_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct")

    processor = AutoProcessor.from_pretrained(model_dir)

    processor = AutoProcessor.from_pretrained(model_dir)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "./demo.jpg",
                },
                {"type": "text", "text": "描述图像内容"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vit.device)


    pixel_values = inputs['pixel_values'].type(vit.get_dtype())
    image_embeds = vit(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])

    print(image_embeds,image_embeds.shape)


def get_vit(save_path):

    model_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct")

    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )

    print(model.num_parameters())

    processor = AutoProcessor.from_pretrained(model_dir)

    print(model.visual.num_parameters())

    test_vit_embedding( model.visual)

    model.visual.save_pretrained(save_path,safe_serialization=False)



    config = Qwen2VLConfig.from_pretrained(model_dir)

    loaded_vit = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
    state_dict = torch.load(save_path + '/pytorch_model.bin')
    # 步骤 4: 将加载的权重应用到模型
    loaded_vit.load_state_dict(state_dict)

    loaded_vit.to("cuda")
    loaded_vit.eval()
    loaded_vit.half()

    test_vit_embedding(loaded_vit)

def sepout_vit():

    model_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct")

    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)

    model.eval()

    vit =  model.visual


   
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "./demo.jpg",
                },
                {"type": "text", "text": "描述图像内容"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print('text')
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")


    pixel_values = inputs['pixel_values'].type(vit.get_dtype())
    image_embeds = vit(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])

    print(image_embeds,image_embeds.shape)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


if __name__ == '__main__':

    #sepout_vit()

    save_path = "./models/qwen_vit/"
    get_vit(save_path)