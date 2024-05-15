# !pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import time

start = time.time()
print('---=== 1 ===---')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print('---=== 2 ===---')

#model_id = "openai/whisper-large-v3"
#model_id = "openai/whisper-medium"
#model_id = "openai/whisper-small"
model_id = "openai/whisper-tiny"

print('---=== 2.5 ===---')

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

print('---=== 3 ===---')

model.to(device)

print('---=== 4 ===---')

processor = AutoProcessor.from_pretrained(model_id)

print('---=== 5 ===---')

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

print('---=== 6 ===---')

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

print('---=== 7 ===---')

sample = dataset[0]["audio"]

finish = time.time()
res = finish - start

print(f'Время работы: {time.time() - start} секунд')
start = time.time()

print('---=== 8 ===---')

#result = pipe("/opt/audio_recognition/upload/zayka.mp3")
result = pipe("/opt/audio_recognition/upload/zayka.mp3", return_timestamps=True, generate_kwargs={"language": "russian"})
print(result["text"])
print(f'Время работы: {time.time() - start} секунд')

#start = time.time()

#print('---=== 9 ===---')

#result = pipe("/opt/audio_recognition/upload/repka.mp3", return_timestamps=True, generate_kwargs={"language": "russian"})
#print(result["text"])
#print(f'Время работы: {time.time() - start} секунд')
