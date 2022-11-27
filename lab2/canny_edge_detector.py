import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def dataloader(dir_name = 'data'):    
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = cv2.imread(dir_name + '/' + filename, cv2.IMREAD_GRAYSCALE)
            imgs.append(img)
    return imgs


def visualize_part(imgs, format = 'gray' ):
    plt.figure(figsize=(15, 20))
    for i, img in enumerate(imgs):
        plt.subplot(int(len(imgs)//2+1), 2, int(i+1//2+1))
        plt.imshow(img, format)

    plt.show()


def visualize(imgs, images_original, format = 'gray', lowthreshold = 0.4, highthreshold = 0.7):
    plt.figure(figsize=(50, 50))

    for i, img in enumerate(zip(images_original, imgs)):
        plt_idx = (i+1)*3 - 2
        plt.subplot(len(imgs), 3, plt_idx)
        plt.imshow(img[0], format)

        plt_idx = (i+1)*3 - 1
        plt.subplot(len(imgs), 3, plt_idx)
        plt.imshow(img[1], format)

        plt_idx = (i+1)*3
        plt.subplot(len(imgs), 3, plt_idx)
        plt.imshow(cv2.Canny(img[0], lowthreshold * 255, highthreshold * 255), format)

    plt.show()


class cannyEdgeDetector:
    def __init__(self, imgs, lowthreshold=0.12, highthreshold=0.15):
        self.weak_pixel     = 100
        self.strong_pixel   = 255
        self.lowThreshold   = lowthreshold
        self.highThreshold  = highthreshold
        self.imgs           = imgs
        self.imgs_final     = []
        self.img_smoothed   = []
        self.gradient       = []
        self.theta          = []
        self.nonMaxImg      = []
        self.thresholdImg   = []
    
    
    def conv(self, image, kernel, stride = 1, padding = 1):
        image = np.pad(image, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)

        kernel_height, kernel_width = kernel.shape
        padded_height, padded_width = image.shape

        output_height = (padded_height - kernel_height) // stride + 1
        output_width = (padded_width - kernel_width) // stride + 1

        new_image = np.zeros((output_height, output_width)).astype(np.float32)
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
        
        return new_image


    def gaussian_kernel(self):
        kernel = np.array(
            [[ 1,  4,  7,  4,  1], 
             [ 4, 16, 26, 16,  4], 
             [ 7, 26, 41, 26,  7],
             [ 4, 16, 26, 16,  4],
             [ 1,  4,  7,  4,  1] ]
        ) /273.
        return kernel
    
    def sobel_filters(self, img):

        kernel_x = np.array([[-1, 0, 1], 
                             [-2, 0, 2], 
                             [-1, 0, 1]])
        gradient_x = self.conv(img, kernel_x)

        kernel_y = np.array([[ 1, 2, 1], 
                             [ 0, 0, 0], 
                             [-1,-2,-1]])
        gradient_y = self.conv(img, kernel_y)

        gradient = np.hypot(gradient_x, gradient_y)
        gradient = gradient / gradient.max() * 255
        theta = np.arctan2(gradient_y, gradient_x)
        return gradient, theta
    

    def non_max_suppression(self, img, angle):
        x, y    = img.shape
        new_img = np.zeros((x,y))
        angle   = angle * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,x-1):
            for j in range(1,y-1):
                front = 255
                back  = 255

                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    front = img[i, j+1]
                    back  = img[i, j-1]
         
                elif (22.5 <= angle[i,j] < 67.5):
                    front = img[i+1, j-1]
                    back  = img[i-1, j+1]
              
                elif (67.5 <= angle[i,j] < 112.5):
                    front = img[i+1, j]
                    back  = img[i-1, j]

                elif (112.5 <= angle[i,j] < 157.5):
                    front = img[i-1, j-1]
                    back  = img[i+1, j+1]


                if (img[i,j] >= front) and (img[i,j] >= back):
                    new_img[i,j] = img[i,j]
                else:
                    new_img[i,j] = 0

        return new_img


    def threshold(self, img):

        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        strong_i, strong_j = np.where(img >= highThreshold)
        weak_i  , weak_j   = np.where((img <= highThreshold) & (img >= lowThreshold))

        img  = np.zeros(img.shape)
        img[strong_i, strong_j] = self.strong_pixel
        img[weak_i, weak_j]     = self.weak_pixel

        return img


    def hysteresis(self, img):

        M, N   = img.shape
        weak   = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    if (   (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i  , j-1] == strong)            or                 (img[i  , j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0

        return img
    

    def detect(self):
        for img in self.imgs: 
            self.img_smoothed.append(self.conv(img, self.gaussian_kernel(), padding = 2))
            gradient, theta  = self.sobel_filters(self.img_smoothed[-1])
            self.gradient.append(gradient)
            self.theta.append(theta)
            self.nonMaxImg.append(self.non_max_suppression(self.gradient[-1], self.theta[-1]))
            self.thresholdImg.append(self.threshold(self.nonMaxImg[-1]))
            self.imgs_final.append(self.hysteresis(self.thresholdImg[-1]))

        return self.imgs_final


if __name__ == '__main__':
    images_original = dataloader()
    detector = cannyEdgeDetector(images_original, lowthreshold = 0.10, highthreshold = 0.15)
    imgs_final = detector.detect()
    visualize(imgs_final, images_original, lowthreshold = 0.5, highthreshold = 0.8)

    visualize_part(detector.img_smoothed)
    visualize_part(detector.nonMaxImg)
    visualize_part(detector.thresholdImg)
    visualize_part(detector.imgs_final)