# -*- coding: utf-8 -*-
import numpy as np

world_file = 'world.txt'
image_file = 'image.txt'

def loadFileContent(filename):
    content = open(filename, 'r').read()
    content = content.split()
    content = [float(element) for element in content] 
    return content

def loadMatrices(world_file, image_file):
    world_content = loadFileContent(world_file)
    world =[]
    for i in range(0, len(world_content), 3):
        world.append([world_content[i], world_content[i+1], world_content[i+2]])
        
    image_content = loadFileContent(image_file)
    image =[]
    for i in range(0, len(image_content), 2):
        image.append([image_content[i], image_content[i+1]])
        
    return world, image

def makeHomogeneous(matrix):
    for i in range(len(matrix)):
        matrix[i].append(1)
    return matrix

def constructA(world, image):
    A = []
    zeroes = np.zeros(4)
    for i in range(len(image)):
        term1 = np.dot(-1*image[i][2], world[i].T)
        term2 = np.dot(image[i][1], world[i].T)
        term3 = np.dot(image[i][2], world[i].T)
        term4 = np.dot(-1*image[i][0], world[i].T)
        
        row1 = []
        row2 = []
        row1.extend(zeroes)
        row1.extend(term1)
        row1.extend(term2)
        row2.extend(term3)
        row2.extend(zeroes)
        row2.extend(term4)
        
        A.append(row1)
        A.append(row2)
    return A

def parameters(A):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p = vh[-1]
    p = p.reshape(3,4)
    return p

def camaraProjectionCenter(p):
    u, s, vh = np.linalg.svd(p, full_matrices=True)
    c = vh[-1]
    return c

def verify(P, world, image):
    projection = np.matmul(world, P.T)
    distance = 0
    for i in range(image.shape[0]):
        distance += np.linalg.norm(image[i] - projection[i])
    distance /= image.shape[0]
    print("Distance between projection and actual images: ", distance)

def getCamaraParameters(world_file=world_file, image_file=image_file):
    world, image = loadMatrices(world_file, image_file)
    world = makeHomogeneous(world)
    image = makeHomogeneous(image)
    world, image = np.array(world), np.array(image)
    A = constructA(world, image)
    P = parameters(A)
    C = camaraProjectionCenter(P)
    verify(P, world, image)
    print("Camera Parameters: ", P)
    print("Camera projection: ", C)
    return P, C

if __name__ == '__main__':
    P, C = getCamaraParameters(world_file=world_file, image_file=image_file)