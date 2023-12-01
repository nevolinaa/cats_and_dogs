import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def show_image_and_label(img_path, model):
    img = Image.open(img_path)

    plt.imshow(img)
    plt.show()

    img_test = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    img_test = img_test.astype(np.float32) / 255.0
    img_test = cv2.resize(img_test, (128, 128), interpolation=cv2.INTER_AREA)
    img_test = img_test.transpose(2, 0, 1)
    img_test = np.expand_dims(img_test, axis=0)
    img_test = torch.from_numpy(img_test)

    pred = model(img_test)
    pred_label = F.softmax(pred).detach().numpy().argmax()

    if pred_label == 1:
        print("It's a cat!")
    else:
        print("It's a dog!")
