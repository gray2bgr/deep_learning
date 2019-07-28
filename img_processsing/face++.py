# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import matplotlib.pyplot as plt
import time
import json
from skimage import io
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "HoHrjI6XPacLAdDrh66Qv4KdKTbHRGEQ"
secret = "eBpq-FUguZdOnhkKw78Z5pCP1K0eCKEj"
# filepath = r"test.jpg"
filepaths = [os.path.join('../val2014/val2014',i) for i in os.listdir('../val2014/val2014')]
for i in  tqdm(range(len(filepaths))):
    if i < 5890-1:
        continue
    filepath = filepaths[i]
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(filepath, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('1')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append(
        "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8') # change to byte type



    http_body = b'\r\n'.join(data)

    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)

    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try:
        # post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # get response
        qrcont = resp.read()
        # if you want to load as json, you should decode first,
        # for example: json.loads(qrcont.decode('utf-8'))
        #print(qrcont.decode('utf-8'))

        dic = json.loads(qrcont.decode('utf-8'))
        #print(dic)
        # print(type(dic))
        faces_datas = dic['faces']
        with open('../val2014/json/'+filepath.split('\\')[-1].replace('.jpg','.json'),'w') as f_json:
            for faces_data in  faces_datas:
                # faces_data = faces_data[0]
                # print(faces_data.get('face_token'))
                # print(faces_data.get('face_rectangle'))
                dictFace = faces_data.get('face_rectangle')

                # left = dictFace['left']
                # top = dictFace['top']
                # height = dictFace['height']
                # width = dictFace['width']

                #landmark
                dictLandmark = faces_data.get('landmark')
                # print(dictLandmark)
                f_json.write(str(dictFace)+'\n'+str(dictLandmark)+'\n')
                # img = Image.open(filepath)
                # print(type(img))
                # img1 = np.array(img)
                # draw = ImageDraw.Draw(img)
                # draw.line([(left,top),(left+width,top),(left+width,top+height),(left,top+width),(left,top)],'red')
                # new_img = img1[top:(top + height),left:(left + width)]
                # io.imsave(r"res.jpg",new_img)
                # for i in dictLandmark:
                #     top = dictLandmark[i]['y']
                #     left = dictLandmark[i]['x']
                #     draw.line([(left,top),(left+1,top),(left+1,top+1),(left,top+1),(left,top)],'blue')
                # img.show()

    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'))