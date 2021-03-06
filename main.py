import numpy as np
from PIL import Image, ImageOps
import torch
from matplotlib import pyplot as plt
from ImageRecognition import getPattern, ImageRecognition

    

if __name__ == "__main__":

    image_path = "test.jpg" 

    img = ImageOps.grayscale(Image.open(image_path).resize((50,50)))
    np_img = np.array(img)
    plt.imshow(np_img, cmap="gray")
    plt.show()
    

    neural_net = ImageRecognition()
    neural_net.load_state_dict(torch.load("models/model300.pt"))


    pattern = getPattern(np_img, neural_net)
    print(pattern)
