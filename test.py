from audio_recognition import AudioRecognition

ai = AudioRecognition()

#file = 'C:\\Work\\python\\Vodokanal\\autio_recognition\\upload\\zayka.mp3'
file = '/opt/audio_recognition/upload/zayka.mp3'

result = ai.recognition(file)

print(result)
