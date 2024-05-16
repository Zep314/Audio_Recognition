# !pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from mutagen.mp3 import MP3
import time
import os

# Подготовка
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Выбор модели
#model_id = "openai/whisper-large-v3"
model_id = "openai/whisper-medium"
#model_id = "openai/whisper-small"
#model_id = "openai/whisper-tiny"

# Загружаем модель
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

# Загружаем модель в устройство обработки
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Очередь обработки
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

# Выбираем датасет (уже обученный) для работы
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# Собственно обработка
start = time.time()
filename = "/opt/audio_recognition/upload/zayka.mp3"
result = pipe(filename, generate_kwargs={"language": "russian"})
print(f'Размер файла: {os.path.getsize(filename)} байт')
mp3 = MP3(filename)
print(f'Длительность записи: {mp3.info.length:.3f} секунд')
print(f'Время обработки: {time.time() - start:.3f} секунд')
print(f'Результат: {result["text"]}')
