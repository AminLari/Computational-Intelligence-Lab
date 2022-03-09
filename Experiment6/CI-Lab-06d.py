# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:39:03 2020

@author: Asus
"""

import numpy as np

W = np.array([[0, 1, 1, -1], [1, 0, 1, -1], [1, 1, 0, -1], [-1, -1, -1, 0]])
x = np.random.uniform(-1.0, 1.0, (4, 1))

# x = ([[1],[-1],[1],[1]])

y_after = np.sign(np.dot(W, x))
y_before = x
flag = 0
while flag == 0:
    y_after = np.sign(np.dot(W, y_before))
    if np.array_equal(y_before, y_after):
        flag = 1
    y_before = y_after
print('random input:\n', x.T)
print('pattern:\n', y_after.T)
