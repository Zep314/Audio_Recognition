import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import time

class AudioRecognition:
    """
        Распознавание звукового файла
    """
    def __init__(self):
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
        self.pipe = pipeline(
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


    def recognition(self, filename):
        # Собственно обработка
        result = self.pipe(filename, return_timestamps=True, generate_kwargs={"language": "russian"})
        return result["text"]
