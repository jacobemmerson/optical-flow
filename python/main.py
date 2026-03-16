'''
Averages the runtime from N iterations of sparse and denseflow.
Reports the average time spent running the algorithm. Time delta is calculated in the respective files to prevent measuring overhead.
'''
import numpy as np
import argparse
from algorithms.sparse_flow import sparse
from algorithms.dense_flow import dense
from typing import List, Callable


def main():

    
    args = argparse.ArgumentParser(
        prog='Optical Flow timer',
        description='Measure the time for Lucas-Kande sparse and Farneback dense optical flow algorithms.'
    )
    args.add_argument('--iterations', default=5, help='The number of iterations to average over for a measure of time.')
    args = args.parse_args()
    N = args.iterations

    def measure(func: Callable, iterations: int =N):
        return [func() for i in range(iterations)]
    
    def report(times: List[int]):
        mu = np.mean(times)
        sd = np.var(times)
        print(f"Total Time: {np.sum(times):.4f} | Average: {mu:.4f} | Variance: {sd:.4f}")
    
    print(f"Using {N} Iterations...")
    print("Measuring LK Sparse Optical Flow Algorithm...")
    report(measure(sparse))

    print("Measuring Farneback Dense Optical Flow Algorithm...")
    report(measure(dense))

if __name__ == "__main__":
    main()