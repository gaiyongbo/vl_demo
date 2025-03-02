from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, Qwen2VLProcessor
from vison_process import process_vision_info
from modelscope import snapshot_download
import numpy as np
import time

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, processor

    json_post_raw = await request.json()

    prompt = json_post_raw["prompt"]

    image_tokens = json_post_raw["tokens"]
    image_grid_thw = json_post_raw["image_grid_thw"]

    image_tokens = np.array(image_tokens)
    image_grid_thw = np.array(image_grid_thw)

    print(f"\nServer get prompt: '{prompt}'\n")
    time.sleep(1)
    print(f"\nServer get image tokens with size  {image_tokens.shape}, and the value is {image_tokens}...\n")
    time.sleep(1)
    print(f"\nStart to generate text response ....\n")

    image_embedding = torch.from_numpy(image_tokens).to('cuda')
    image_grid_thw = torch.from_numpy(image_grid_thw).to('cuda')
    text = predict(model, processor, "/mnt/workspace/vl_demo/demo.jpg",prompt,image_embedding,image_grid_thw)
    print(f"\nServer response {text}\n")

    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": text,
        "status": 200,
        "time": time_stamp
    }
    return answer


def predict(model,processor, image, prompt,image_embedding,image_grid_thw):


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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        image_embedding=image_embedding,
        padding=True,
        return_tensors="pt",
    )
    inputs['image_grid_thw'] = image_grid_thw
    #inputs['pixel_values'] = None
    inputs = inputs.to("cuda")


    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text



if __name__ == '__main__':
  
    model_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct")

    print(f"download model into :{model_dir}")

    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="cuda"
    )

    print(f"model loaded!")
    processor = Qwen2VLProcessor.from_pretrained(model_dir)

    #test
    #image = "/mnt/workspace/yongbo/yongbin/demo.jpg"
    #text = predict(model, processor, image, "描述图像内容, 图像中的驾驶员有没有专注驾驶， 有没有目视前方？")
    #print(text)

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)

   

    # Inference: Generation of the output
    
