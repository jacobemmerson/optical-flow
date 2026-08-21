| Arm | Kernel | DRAM read | DRAM write | DRAM %peak | SM %peak | L1 hit% | L2 hit% | Occupancy% | Regs/thread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | lk_warp_kernel | 2,373,792 | 260,224 | 5.79 | 40.42 | 96.16 | 69.33 | 89.04 | 64 |
| opencv | transformSmart | 20,608 | 1,504 | 4.65 | 2.17 | 0 | 72.93 | 26.12 | 18 |
| opencv | sparseKernel | 823,744 | 146,752 | 1.33 | 81.72 | 93.86 | 74.96 | 96.51 | 40 |
