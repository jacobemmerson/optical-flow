import numpy as np
import cv2 as cv
import argparse
from PIL import Image
from time import perf_counter
from utils import FlowSet
    
def main():

    args = argparse.ArgumentParser(
        prog='LK Sparse Optical Flow timer',
        description='Measure the time for Lucas-Kande sparse optical flow.'
    )
    args.add_argument('--plot', action='store_true', help='Whether to plot using OpenCVs plotting functions.')
    args = args.parse_args()
    plot = args.plot

    # params for Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners = 100,
        qualityLevel = 0.3,
        minDistance = 7,
        blockSize = 7 
    )

    # params for lucas kanade optical flow
    lk_params = dict(
        winSize  = (15, 15),
        maxLevel = 2,
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    color = np.random.randint(0, 255, (100, 3))
    data = FlowSet('../data', occ=True)

    start = perf_counter()
    for idx, (img1, img2, _) in enumerate(data):
        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # corners/features to track flow using ST method
        f0 = cv.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)

        # calculate optical flow
        f1, st, err = cv.calcOpticalFlowPyrLK(img1_gray, img2_gray, f0, None, **lk_params)

        if plot:
            mask = np.zeros_like(img1)
            if f1 is not None:
                good_new = f1[st==1]
                good_old = f0[st==1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                img2 = cv.circle(img2, (int(a), int(b)), 5, color[i].tolist(), -1)
        
            img = cv.add(img2, mask)
            cv.imshow('old', img1)
            cv.imshow('frame', img)

        k = cv.waitKey(1) & 0xFF
        if k == 27: # Break if ESC is pressed while plotting
            break

    end = perf_counter()
    print(f"Total time = {end - start:.4f} | Average = {(end - start) / len(data):.4f}")


if __name__ == "__main__":
    main()