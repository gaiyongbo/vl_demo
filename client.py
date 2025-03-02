import requests
import json
import time
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor,Qwen2VLConfig
def load_model(save_path):

    config = Qwen2VLConfig.from_pretrained(save_path)

    configv = config.vision_config
    #configv.torchscript=True
    #configv._attn_implementation = "eager"

    loaded_vit = Qwen2VisionTransformerPretrainedModel._from_config(config)
    state_dict = torch.load(save_path + '/pytorch_model.bin',map_location='cuda')
    # 步骤 4: 将加载的权重应用到模型
    loaded_vit.load_state_dict(state_dict)

    return loaded_vit 




def reqeust():


    model_path = './vit_model/models/qwen_vit'
    image = "./IMG_5899.JPG"
    prompt = "描述图像内容"

    print(f"\nuser give image {image}, add prompt {prompt}.\n")
    url = "http://0.0.0.0:8080"


    print("\nstart to encode image to token embedding values\n")

    vit = load_model(model_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    processor = AutoProcessor.from_pretrained(model_path)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

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

    vit = vit.to("cuda")
    vit.eval()

    with torch.no_grad():
        image_embeds = vit(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
    image_embeds = image_embeds.cpu().numpy()
    print(f"\nshape of image token embeddings{image_embeds.shape}, value of token embeddings {image_embeds}\n")
    image_embeds = image_embeds.tolist()
    image_grid_thw = inputs['image_grid_thw'].cpu().numpy().tolist()
    data = {'prompt': prompt, "tokens":image_embeds,"image_grid_thw":image_grid_thw}

    data = json.dumps(data, ensure_ascii=False).encode('utf-8')

    print("\nstart to send token embeddings and prompt to server\n")
    response = requests.post(url, data=data)
    data = json.loads(response.text)

    print(f"\nget predict result {data}\n")

if __name__ == '__main__':
    reqeust()
