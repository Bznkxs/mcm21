from matplotlib import pyplot as plt
import os
import sys
import cv2
import imghdr
from tqdm import tqdm
plt.figure()
plt.ion()

path = os.path.join(os.path.dirname(__file__), 'outputs001')
print(path)
filenames = []
for filename in tqdm(os.listdir(path)):
    fname = filename
    filename = os.path.join(path, filename)
    if os.path.isfile(filename):
        if fname[:-4].isdigit() and fname[-4:] == '.png':
            filenames.append(int(fname[:-4]))

filenames.sort()
filenames = filenames[:500]
print(filenames)

files = []
for filename in tqdm(filenames):
    filename = str(filename) + '.png'
    filename = os.path.join(path, filename)
    file = cv2.imread(filename)
    files.append(file)
            #plt.imshow(file)
            #plt.pause(0.1)
print(len(files))
plt.show()
while True:
    for i, file in tqdm(enumerate(files)):
        #plt.cla()
        plt.title(str(i) + '/' + str(len(files)))
        plt.imshow(file)
        plt.pause(0.01)
