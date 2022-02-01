import time
import logging

point0 = time.time()
# import dataloading
point1 = time.time()

print(f'Time for data loading is {point1 - point0:2.2f} seconds')

import datapreparation
point2 = time.time()

print(f'Time for data loading is {point2 - point1:2.2f} seconds')