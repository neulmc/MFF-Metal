import numpy as np
import os

dirs = ['Unet_FPS-n','NestedUNet_FPS-n','AttU_Net_FPS-n','transnet_FPS-n','LMFFNet_FPS-n','OurModel_FPS-n','OurModel_FPS-c60n','OurModel_FPS-c180n']
for dir in dirs:
    print(dir)
    num = []
    with open(dir + '/adam-0-log.txt', 'r') as file:
        for line in file:
            if 'FPS: ' in line:
                num.append(float(line.replace('FPS: ','')))
        num = np.array(num)
        print(np.mean(num))
