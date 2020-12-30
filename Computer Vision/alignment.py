# -*- coding: utf-8 -*-
import cv2
import numpy as np
from os import sys
import random

image1_path = 'scene.pgm'
image2_path = 'book.pgm'

def getRegions(image1, image2):
    sift = cv2.SIFT_create()
    image1_keypoints = sift.detect(image1, None)
    image2_keypoints = sift.detect(image2, None)
    
    image1_regions=cv2.drawKeypoints(image1,image1_keypoints,
                                     outImage = None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('image1_regions.jpg',image1_regions)
    image2_regions=cv2.drawKeypoints(image2,image2_keypoints,
                                     outImage = None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('image2_regions.jpg',image2_regions)
    
    image1_keypoints, image1_descriptors = sift.compute(image1,image1_keypoints)
    image2_keypoints, image2_descriptors = sift.compute(image2,image2_keypoints)
    
    return image1_keypoints, image2_keypoints, image1_descriptors, image2_descriptors

def putativeMatches(image1_descriptors, image2_descriptors):
    desc1_length = image1_descriptors.shape[0]
    desc2_length = image2_descriptors.shape[0]
    putative_matches = []
    
    for i in range(desc1_length):
        best_match = (0, sys.maxsize)
        second_best_match = (0, sys.maxsize)
        for j in range(desc2_length):
            dist = np.linalg.norm(image1_descriptors[i] - image2_descriptors[j])
            if dist < best_match[1]:
                second_best_match = best_match
                best_match = (j, dist)
            elif dist < second_best_match[1]:
                second_best_match = (j, dist)
        if (best_match[1]/second_best_match[1]) < 0.9:
            putative_matches.append((i, best_match[0], best_match[1]))
        
    return putative_matches

def visualizeMatches(image1, image2, image1_keypoints, image2_keypoints, putative_matches):
    combine = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1]+image2.shape[1], image1.shape[2]))
    combine[0:image1.shape[0],0:image1.shape[1]] = image1
    combine[0:image2.shape[0],image1.shape[1]:image1.shape[1]+image2.shape[1]] = image2
    color = np.random.randint(0,256,3)
    color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
    radius = 10
    thickness = 2
    
    for match in putative_matches:
        point1 = tuple(np.round(image1_keypoints[match[0]].pt).astype(int))
        point2 = tuple(np.round(image2_keypoints[match[1]].pt).astype(int) + np.array([image1.shape[1], 0]))
        cv2.line(combine, point1, point2, color, thickness)
        cv2.circle(combine, point1, radius, color, thickness)
        cv2.circle(combine, point2, radius, color, thickness)
    cv2.imwrite('combine.jpg', combine)

def transformationParameters(image1_keypoints, image2_keypoints, matches):
    A = []
    b = []
    for point in matches:
        row1 = [image1_keypoints[point[0]].pt[0],image1_keypoints[point[0]].pt[1],1,0,0,0]
        row2 = [0,0,0,image1_keypoints[point[0]].pt[0],image1_keypoints[point[0]].pt[1],1]
        A.append(row1)
        A.append(row2)
        b.append(image1_keypoints[point[1]].pt[0])
        b.append(image2_keypoints[point[1]].pt[1])
    A = np.array(A)
    b = np.array(b)
    t = np.linalg.solve(A, b.T)
    return t
    
def transformation(image_keypoints, match, t):
    x, y = image_keypoints[match[0]].pt[0], image_keypoints[match[0]].pt[1]
    xt = t[0]*x + t[1]*y + t[2]
    yt = t[3]*x + t[4]*y + t[5]
    return xt, yt

def RANSAC(image1_keypoints, image2_keypoints, putative_matches):
    best_transform = []
    best_inliers = []
    for i in range(100):
        random_points = random.sample(putative_matches, 3)
        t = transformationParameters(image1_keypoints, image2_keypoints, random_points) 
        
        inliers = []
        for match in putative_matches:
            x1, y1 = transformation(image1_keypoints, match, t)
            x2, y2 = image2_keypoints[match[1]].pt[0], image2_keypoints[match[1]].pt[1]
            
            radius = np.linalg.norm(np.array((x1, y1))-np.array((x2, y2)))
            if radius <= 10:
                inliers.append(match)
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = t
    return best_transform, best_inliers

def finalTransformation(image1, best_transform, best_inliers):    
    #final_transform_parameters = transformationParameters(image1_keypoints, image2_keypoints, best_inliers)
    H = [[best_transform[0], best_transform[1], best_transform[4]],
         [best_transform[2], best_transform[3], best_transform[5]],
         [0, 0, 1]]
    H = np.array(H) 
        
    transformed_image = cv2.warpPerspective(image1, H, (image1.shape[1], image1.shape[0]))
    cv2.imwrite('transformed_image.jpg', transformed_image)
    return H
    
def align(image1_path = image1_path, image2_path = image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    image1_keypoints, image2_keypoints, image1_descriptors, image2_descriptors = getRegions(image1, image2)
    
    putative_matches = putativeMatches(image1_descriptors, image2_descriptors)
    
    visualizeMatches(image1, image2, image1_keypoints, image2_keypoints, putative_matches)
    
    best_transform, best_inliers = RANSAC(image1_keypoints, image2_keypoints, putative_matches)
    
    H = finalTransformation(image1, best_transform, best_inliers)
    return H

if __name__ == '__main__':
    align(image1_path = image1_path, image2_path = image2_path)