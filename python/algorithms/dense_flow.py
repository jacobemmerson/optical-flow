import numpy as np
import cv2 as cv
import argparse
from PIL import Image
from time import perf_counter_ns
from algorithms.utils import FlowSet
    
def dense(plot=False, return_average=False):

    # Parameters for farneback optical flow
    farneback_params = dict( 
        pyr_scale=0.3,
        levels=2,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.5,
        flags=0
    )

    data = FlowSet('../data', occ=True)
    time = 0
    frames = 0
    for idx, (img1, img2, flow) in enumerate(data):
        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        start = perf_counter_ns()
        predicted_flow = cv.calcOpticalFlowFarneback(img1_gray, img2_gray, None, **farneback_params)
        end = perf_counter_ns()
        time += end - start
        frames += 1
        if plot:
            cv.imshow('frame', img2)
            cv.imshow('flow', flow)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                return
            
    return (time / frames) * 1e-6