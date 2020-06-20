import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

train_json=json.load(open(r"..\datasets\street_signs\mchar_train.json"))
# ../表示上级目录
# <div STYLE="page-break-after：always；"></div>
# {"000000.png": {"height": [30.0], "label": [5],
#  "left": [43.0], "top": [7.0], "width": [19.0]}
def parse_json(d):
    arr=np.array([
        d['top'],d['height'],d['left'],d['width'],d['label']
    ])
    arr=arr.astype(int)
    return arr





