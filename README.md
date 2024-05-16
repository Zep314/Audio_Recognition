# Распознавание аудиофайлов в текст

Взял модель **openai/whisper-large-v3** вот [отсюда](https://huggingface.co/openai/whisper-large-v3)

Сделал тестовый сервер. Можно потестить: [Ссылка](http://92.255.234.30:55555/)

Пробовал детские сказки. Файлы брал [отсюда](https://papaskazki.ru/Barto.php)

Распознает долго. Запись, длительностью 15 сек. обрабатывается около 2-х минут.

Виртуальная машина была с 32 процессорами и 16Гб ОЗУ. Если ОЗУ меньше - то программа падает - нехватает памяти. Если меньше процессоров - работает медленнее.

Нужно использовать GPU. Вот [ноутбук](https://colab.research.google.com/drive/1KTqd3b3usoVtBh2ko3N7OPSzdFstkuRl?usp=sharing) на Google colab. Там можно поставить среду выполнения с графическим ускорителем. Работает гораздо веселее (секунды на обработку)  

Главная страница
![screen1.png](screen1.png)

Страница результата работы
![screen2.png](screen2.png)

#### Подготовка виртуалки:

Стандартный Linux Debian 12.1 с обновленными портами

    sudo apt install git python3.11-venv ffmpeg
    cd /opt
    git clone https://github.com/Zep314/Audio_Recognition.git
    cd /opt/Audio_Recognition
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]
    pip install -r requirements.txt
    puthon ./main.py

Первый раз запускается довольно долго. Выкачивает из сети ~ 4 Гб файлов.

Если менять модель - то, сответственно, снова будет выкачивать много из сети.
