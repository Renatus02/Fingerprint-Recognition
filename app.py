import cv2
import os
import sys
import numpy
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin

os.chdir("E:\SEMESTER 5\Biometric\Fingerprint Recognition Python")

def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)
    temp3 = numpy.array(temp2)
    
    enhanced_img = numpy.array(temp0)
    filter0 = numpy.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6
    
    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

    return temp2


def get_descriptors(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = image_enhance.image_enhance(img)
    img = numpy.array(img, dtype=numpy.uint8)
    # Threshold
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Normalize to 0 and 1 range
    img[img == 255] = 1
    
    #Thinning
    skeleton = skeletonize(img)
    skeleton = numpy.array(skeleton, dtype=numpy.uint8)
    skeleton = removedot(skeleton)
    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125
    # Extract keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
        for y in range(0, harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))
    # Define descriptor
    orb = cv2.ORB_create()
    # Compute descriptors
    _, des = orb.compute(img, keypoints)
    return (keypoints, des)


def main():
    source_image_name = sys.argv[1]
    source_img = cv2.imread("database/" + source_image_name, cv2.IMREAD_GRAYSCALE)
    source_kp, source_des = get_descriptors(source_img)

    dataset_dir = "database"
    matches_found = False

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".png"):
            target_img = cv2.imread(os.path.join(dataset_dir, filename), cv2.IMREAD_GRAYSCALE)
            target_kp, target_des = get_descriptors(target_img)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(source_des, target_des), key= lambda match:match.distance)
            score = sum(match.distance for match in matches) / len(matches) if matches else float('inf')

            if score < 33:
                print(f"Fingerprint matches with {filename}. Score: {score}")
                matches_found = True

    if not matches_found:
        print("No matching fingerprint found in the dataset.")


if __name__ == "__main__":
    try:
        main()
    except:
        raise
