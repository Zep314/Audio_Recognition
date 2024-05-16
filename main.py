import os
import time
from flask import *
from audio_recognition import AudioRecognition
from timeout_decorator import timeout, TimeoutError
from mutagen.mp3 import MP3

#UPLOAD_DIR = 'C:\\Work\\python\\Vodokanal\\autio_recognition\\upload'
UPLOAD_DIR = '/opt/audio_recognition/upload'

app = Flask(__name__)

ai = AudioRecognition()

@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/about/')
def about():
    return render_template('about.html')

@timeout(5 * 60) # set maximum execution time in seconds
@app.route('/success', methods=['POST'])
def success():
    f = None
    if request.method == 'POST':
        f = request.files['file']
    filename = f'{UPLOAD_DIR}{os.sep}{f.filename}'
    f.save(filename)

    start = time.time()
    result = ai.recognition(filename)
    duration_recognition = f'{time.time() - start:.3f}'
    mp3 = MP3(filename)
    duration_file = f'{mp3.info.length:.3f}'
    file_size = f'{os.path.getsize(filename)}'

    return render_template("success.html", 
                            recog_text=result,
                            duration_recognition=duration_recognition,
                            duration_file=duration_file,
                            file_size=file_size
                        )


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
