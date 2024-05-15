import os
from flask import *
from audio_recognition import AudioRecognition
from timeout_decorator import timeout, TimeoutError

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
    result = ai.recognition(filename)
    return render_template("success.html", recog_text=result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
