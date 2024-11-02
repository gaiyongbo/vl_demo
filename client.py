import requests
import json
from image_encoder import image_encoder
import time

image = "./demo.jpg"
prompt = "描述图像内容"

print(f"\nuser give image {image}, add prompt {prompt}.\n")
time.sleep(1)

url = "http://127.0.0.1:8080"


print("\nstart to encode image to token embedding values\n")
time.sleep(1)
start = time.time()
token_embedding = image_encoder(image)
end = time.time()
print(f"\nend to encode image to token embedding values, time cost :{end-start} sencond.\n")
time.sleep(1)

print(f"\nshape of image token embeddings{token_embedding.shape}, value of token embeddings {token_embedding}\n")

token_embedding = token_embedding.tolist()
data = {'prompt': prompt, "tokens":token_embedding}


data = json.dumps(data, ensure_ascii=False).encode('utf-8')
time.sleep(1)

print("\nstart to send token embeddings and prompt to server\n")
response = requests.post(url, data=data)
data = json.loads(response.text)

print(f"\nget predict result {data}\n")

