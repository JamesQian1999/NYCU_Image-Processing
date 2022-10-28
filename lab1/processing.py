import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def save_img(args, img, name = 'processing'):

    if(args.g):
        plt.imsave('test/'+ args.name + '_' + name +'.png', img , cmap='gray')
    else:
        plt.imsave('test/'+ args.name + '_' + name +'.png', img )


def to01(img):
    img = (img + np.abs(img.min()))
    img /= img.max()
    return img

def conv(image, kernel, stride = 1, padding = 1):
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


def get_histogram(image, bins = 256):

    histogram = np.zeros(bins)
    for v in range(256):
        histogram[v] +=  (image == v).sum()
    
    return histogram


def median(image, kernel_size = 3, stride = 1, padding = 1):
    image = np.pad(image, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)

    kernel_height, kernel_width = kernel_size, kernel_size
    padded_height, padded_width = image.shape

    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1

    new_image = np.zeros((output_height, output_width)).astype(np.float32)
    for y in range(0, output_height):
        for x in range(0, output_width):
            tmp = (image[y * stride:y * stride + kernel_height, x * stride:x * stride + kernel_width]).reshape(-1,1)
            tmp.sort(axis = 0)
            new_image[y][x] = tmp[(kernel_size*kernel_size-1)//2].astype(np.float32)
    
    return new_image

def his_equl(args, img):

    if args.g:
        his = get_histogram(img)
        p = his/his.sum()
        cdf = np.cumsum(p)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i,j] = 255 * cdf[ img[i,j]]

        save_img(args, img , 'his_equl' )

        plt.bar(np.arange(len(his)), his)
        plt.savefig('test/' + args.name + '_his_equl_before.png') 

        plt.clf()
        plt.bar(np.arange(len(his)),get_histogram(img))
        plt.savefig('test/' + args.name + '_his_equl_after.png')

    else:
        his = { 'b' : get_histogram(img[:,:,0]),
                'g' : get_histogram(img[:,:,1]), 
                'r' : get_histogram(img[:,:,2]) }
        
        cdf = { 'b' : np.cumsum(his['b']/his['b'].sum()),
                'g' : np.cumsum(his['g']/his['g'].sum()), 
                'r' : np.cumsum(his['r']/his['r'].sum()) }

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                count = 0
                for c in his.keys():
                    img[i,j,count] = 255 * cdf[c][img[i,j,count]]
                    count+=1

        save_img(args, img , 'his_equl' )

        plt.figure(figsize=(25,10))
        count = 1
        for c in his.keys():
            plt.subplot(1,3,count)
            plt.title(c)
            plt.bar(np.arange(256), his[c])
            count+=1

        plt.savefig('test/' + args.name + '_his_equl_before.png') 


        plt.clf()
        his = { 'b' : get_histogram(img[:,:,0]),
                'g' : get_histogram(img[:,:,1]), 
                'r' : get_histogram(img[:,:,2]) }
        count = 1
        for c in his.keys():
            plt.subplot(1,3,count)
            plt.title(c)
            plt.bar(np.arange(256), his[c])
            count+=1

        plt.savefig('test/' + args.name + '_his_equl_after.png')
    
    return img

def power_transformation(arg, img, gamma = 1.5, c = 1.):
    img = img.astype('float64') / 255.0
    img = c * np.power(img, gamma)
    save_img(args, img, 'power_transformation' )

    return img

def gradients(arg, img):

    kernel_h = np.array(
        [[ -1, 0, 1], 
         [ -1, 0, 1], 
         [ -1, 0, 1]]
    )
    kernel_w = np.array(
        [[ -1, -1, -1], 
         [  0,  0,  0], 
         [  1,  1,  1]]
    )

    if arg.g:
        w = conv(img, kernel_w)
        h = conv(img, kernel_w)
        img = w + h
    else:
        w = []
        h = []
        for i in range(3):
            w.append(conv(img[:,:,i], kernel_w))
            h.append(conv(img[:,:,i], kernel_h))
        w = np.stack((w[0], w[1], w[2]), axis=2)
        h = np.stack((h[0], h[1], h[2]), axis=2)
        img = w + h
        img = to01(img)

    save_img(args, img, 'gradients' )

    return img


def median_filter(arg, img, kernel = 3):
    if arg.g:
        img = median(img, kernel)
        save_img(args, img, 'median_filter' )
        return img
    else:
        output = []
        for i in range(3):
            output.append(median(img[:,:,i], kernel))

        img = np.stack((output[0], output[1], output[2]), axis=2)
        save_img(args, img/255, 'median_filter' )
        return img


def threshold(img, threshold ):
    idx = img > threshold
    img[idx] = 255
    idx = img <= threshold
    img[idx] = 0

    save_img(args, img/255, 'threshold' )
    return img

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--name', default='computer', type=str)
    parser.add_argument('-g', action="store_true")
    args = parser.parse_args()

    #loading data
    if(args.g):
            img = cv2.imread('data/' + args.name + '.png', cv2.IMREAD_GRAYSCALE)
    else:
            img = cv2.imread('data/' + args.name + '.png')


    test = img.copy()

    img = power_transformation(args, img)
    img = his_equl(args, (img*255).astype('uint8')) 
    for i in range(3):
        img = median_filter(args, img)

    threshold(test, (test.max()+test.min())/2)
    gradients(args, test)

