import requests
from pathlib import Path 


if Path('haar_cascades.xml').is_file():
    print("haar_cascades.xml file already exists, skipping download....")
else:
    print("Downloading haar_cascades.xml")
    request = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
    with open('haar_cascades.xml','wb') as f:
        f.write(request.content)
