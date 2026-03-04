import numpy as np
import cv2 as cv
import os
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def plot_corners(frame, corners):
    for i in np.int32(corners):
        x,y = i.ravel()
        cv.circle(frame,(x,y),3,255,-1)
    
    plt.imshow(frame),plt.show()

def plot_sample(img1, img2, flow):
    f, axarr = plt.subplots(3, figsize=(8, 8))
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)
    axarr[2].imshow(flow)

    plt.show()

class FlowSet(Dataset):
    def __init__(self, dir_path, occ=True):
        
        self.root = dir_path
        self.occ = occ # occlusion flag

        self.image_dir = os.path.join(self.root, "training", "image_2")

        flow_folder = "flow_occ" if occ else "flow_noc"
        self.flow_dir = os.path.join(self.root, "training", flow_folder)
        self.samples = self._collect_samples()


    def _collect_samples(self):
        flow_files = sorted(glob(os.path.join(self.flow_dir, "*_10.png"))) # file names
        samples = []

        for flow_path in flow_files:
            filename = os.path.basename(flow_path)
            img1_path = os.path.join(self.image_dir, filename)
            img2_path = os.path.join(
                self.image_dir, filename.replace("_10.png", "_11.png")
            )

            if os.path.exists(img1_path) and os.path.exists(img2_path):
                samples.append((img1_path, img2_path, flow_path))

        return samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img1_path, img2_path, flow_path = self.samples[index]
        img1, img2 = cv.imread(img1_path), cv.imread(img2_path)
        flow = cv.imread(flow_path)

        return img1, img2, flow