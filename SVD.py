import matplotlib.pyplot as plt
import numpy as np
import cv2
dog = cv2.imread('dog.png')
cv2.imshow('dog',dog)
gray = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray dog',gray)
U, S, VT = np.linalg.svd(gray,full_matrices=False)
S = np.diag(S)
j = 0
for r in (5, 20, 100):
    #construct approximate image
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    plt.figure(j+1)
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r =' + str(r))
    plt.show()

cv2.waitKey(0)