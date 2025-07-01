import time
import torch
from transformers import pipeline

from audio_recorder import AudioRecorder
import threading
from queue import Queue


class WhisperInfer:
    def __init__(self):
        print("Loading whisper-v3")
        self.pipe = pipeline("automatic-speech-recognition",
                             "openai/whisper-large-v3",
                             torch_dtype=torch.float16,
                             device="cuda:0")

        self.pipe.model = self.pipe.model.to_bettertransformer()

    def predict(self, audio_file):
        generate_kwargs = {
            # "beam_size": 5,
            "language": 'en',
            "temperature": 0.2,
            "repetition_penalty": 3.0,
            # "condition_on_previous_text": True,
            "task": "transcribe"
        }
        outputs = self.pipe(audio_file,
                            chunk_length_s=30,
                            batch_size=24,
                            return_timestamps=True,
                            max_new_tokens=64,
                            generate_kwargs=generate_kwargs)

        return outputs["text"]


class RealtimeWhisper:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.last_transcribe_time = time.time()

        self.whisper_infer = WhisperInfer()

        self.result_queue = Queue()
        self.last_result = ""

        self.run_thread = threading.Thread(target=self.run, args=())
        self.run_thread.start()

    def run(self):
        time.sleep(2)
        self.last_transcribe_time = time.time()
        time.sleep(2)
        while True:
            time_elapsed = time.time() - self.last_transcribe_time
            if time_elapsed < 1:  # We want at least 1 seconds of audio
                time.sleep(1-time_elapsed)
                continue
            time_elapsed = min(time_elapsed, 20)  # if longer than 20 we just have to skip some to catch up
            self.last_transcribe_time = time.time()

            duration: int = max(5, int(time_elapsed))  # At least 5 seconds of audio
            audio_file = self.recorder.audio_sample_to_wav(duration)
            transcription = self.whisper_infer.predict(audio_file)

            print(transcription)

            self.result_queue.put_nowait(transcription)

    def get_results(self):
        texts = []
        q = self.result_queue
        while not q.empty():
            texts.append(q.get_nowait())
        if not texts:
            texts.append(self.last_result)
        else:
            self.last_result = texts[0]

        return texts


if __name__ == '__main__':
    w = RealtimeWhisper()
    while True:
        time.sleep(5)


