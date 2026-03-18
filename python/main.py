'''
Averages the runtime from N iterations of sparse and denseflow.
Reports the average time spent running the algorithm. Time delta is calculated in the respective files to prevent measuring overhead.
'''
import numpy as np
import argparse
from algorithms.sparse_flow import sparse
from algorithms.dense_flow import dense
from typing import List, Callable
import cv2 as cv


def main():

    
    args = argparse.ArgumentParser(
        prog='Optical Flow timer',
        description='Measure the time for Lucas-Kande sparse and Farneback dense optical flow algorithms.'
    )
    args.add_argument('--iterations', default=1, help='The number of iterations to average over for a measure of time.')
    args.add_argument('--measure', default=0, help='Which algorithm to measure; 0 = Sparse, 1 = Dense, 2 = Both')
    args = args.parse_args()
    N = int(args.iterations)
    to_measure = int(args.measure)
    
    print(f"cuda enabled: {cv.cuda.getCudaEnabledDeviceCount()}")
    
    def measure(func: Callable, iterations: int =N):
        return [func() for i in range(iterations)]
    
    def report(times: List[int]):
        mu = np.mean(times)
        sd = np.var(times)
        print(f"Total Time (ms): {np.sum(times):.6f} | Average per Iteration (ms): {mu:.6f} | Variance: {sd:.6f}")
    
    if to_measure in {0, 2}:
        print(f"Using {N} Iterations...")
        print("Measuring LK Sparse Optical Flow Algorithm...")
        report(measure(sparse))

    if to_measure in {1, 2}:
        print("Measuring Farneback Dense Optical Flow Algorithm...")
        report(measure(dense))

if __name__ == "__main__":
    main()