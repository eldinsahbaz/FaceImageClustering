import cv2
import os
from sys import maxint
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import community
import networkx as nx
import matplotlib.pyplot as plt

def getFiles():
    print('getting files')
    images = list()
    rootdir = os.path.dirname(os.path.realpath(__file__))

    for subdir, dirs, files in os.walk(rootdir):
        os.chdir(rootdir)
        for file in files:
            f = os.path.join(subdir, file)
            print(f)
            if '.jpg' in f:
                    images.append(cv2.imread(os.path.join(subdir, f)))
    return images
    #return resize(normalize_intensity(images))

def getFaces(images):
    print('getting faces')
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    profileCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        frontalFaces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        profileFaces = profileCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        print "Found {0} faces!".format(len(frontalFaces) + len(profileFaces))

        faces = list()

        for (x, y, w, h) in frontalFaces:
            faces.append(image[y:y+h, x:x+w])

        for (x, y, w, h) in profileFaces:
            faces.append(image[y:y+h, x:x+w])

    return resize(normalize_intensity(faces))

#https://github.com/rragundez/PyData/blob/master/face_recognition_system/operations.py
def resize(images, size=(100, 100)):
    print('resizing')
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # using different OpenCV method if enlarging or shrinking
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

#https://github.com/rragundez/PyData/blob/master/face_recognition_system/operations.py
def normalize_intensity(images):
    print('normalizing')
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def eigenFaces(faces, faceSize=(100,100)):
    print('computing eigenFaces')
    #my_scores.sort(key=lambda row: row[1:], reverse=True)
    if len(faces) == 0:
        return faces

    numPixels = faceSize[0]*faceSize[1]
    faceMatrix = np.matrix(np.zeros((numPixels, len(faces)), dtype=np.float))

    for i in range(0, len(faces)):
        faceMatrix[:, i] = np.matrix(faces[i]).reshape((numPixels, 1))

    avg = np.matrix(np.sum(faceMatrix, axis=1))/faceSize[1]
    A = faceMatrix - avg

    Arows = (A.shape)[0]
    Acols = (A.shape)[1]

    values, vectors = np.linalg.eig(np.matmul(A.T, A))
    eigenfaces = np.matrix(normalize(np.matmul(A, vectors), axis=0, norm='l1'))

    return eigenfaces

def similarity(eigenfaces):
    print('computing similarity matrix')
    numRows = (eigenfaces.shape)[0]
    numCols = (eigenfaces.shape)[1]
    similarityMatrix = np.matrix(np.zeros((numCols, numCols), dtype=np.float))

    for i in range(0, numCols):
        for j in range(0, numCols):
            similarityMatrix[i,j] = np.power(np.linalg.norm(eigenfaces[:, i] - eigenfaces[:, j]), 2)

    return (1-normalize(similarityMatrix, axis=0, norm='l1'))

#http://perso.crans.org/aynaud/communities/
def graphClustering(similarityMatrix):
    print('clustering')
    similarityMatrix = sp.sparse.csr_matrix(similarityMatrix)
    G = nx.from_scipy_sparse_matrix(similarityMatrix)
    partition = community.best_partition(G)

    #drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20, node_color = str(count / size))


    nx.draw_networkx_edges(G,pos, alpha=0.5)
    plt.show()

graphClustering(similarity(eigenFaces(getFaces(getFiles()))))
