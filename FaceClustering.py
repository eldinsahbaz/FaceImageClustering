import cv2, os, math, pywt
from sys import maxint
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
import scipy.ndimage as nd
from sklearn.cluster import *
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import *
from numpy import *
from sklearn.metrics import *
from scipy.stats.mstats import zscore
from PIL import Image

#data collection
def getFiles():
    print('getting files')
    imagesTitles = list()
    images = list()
    rootdir = os.path.dirname(os.path.realpath(__file__))

    for subdir, dirs, files in os.walk(rootdir):
        os.chdir(rootdir)
        for file in files:
            f = os.path.join(subdir, file)
            if '.jpg' in f:
                    imagesTitles.append(file)

    imagesTitles.sort(key=str.lower)
    for title in imagesTitles:
        #print(title)
        #raw_input()
        images.append(cv2.imread(title, 0))

    return images
    #return resize(normalize_intensity(images))

def getFaces(images):
    print('getting faces')
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    profileCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    faces = list()

    for image in images:
        #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        frontalFaces = faceCascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        profileFaces = profileCascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        #print "Found {0} faces!".format(len(frontalFaces) + len(profileFaces))

        for (x, y, w, h) in frontalFaces:
            faces.append(np.array(image[y:y+h, x:x+w]))

        for (x, y, w, h) in profileFaces:
            faces.append(np.array(image[y:y+h, x:x+w]))

    return normalize_intensity(resize(faces))

def resize(images, size=(100, 100)):
    print('resizing')
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(np.array(image_norm))

    return images_norm


#data pre-processing
def normalize_intensity(images):
    #https://github.com/rragundez/PyData/blob/master/face_recognition_system/operations.py
    print('normalizing')
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        images_norm.append(np.array(cv2.equalizeHist(image)))
    return images_norm

def createFaceMatrix(faces, faceSize=(100, 100)):
    numPixels = faceSize[0]*faceSize[1]
    faceMatrix = np.matrix(np.zeros((numPixels, len(faces)), dtype=np.float))

    for i in range(0, len(faces)):
        faceMatrix[:, i] = np.matrix(faces[i].reshape((numPixels, 1)))

    return faceMatrix

def eigenFaces(faceMatrix):
    print('computing eigenFaces')

    if len(faces) == 0:
        return faces

    avg = np.mean(faceMatrix, axis=1)
    A = faceMatrix - avg

    values, vectors = np.linalg.eig(np.dot(A.T, A))

    indices = np.argsort(values)
    indices = indices[::-1]
    k = int(math.floor(np.matrix(indices).shape[1]*0.2))

    if k == 0:
        indices = indices[0]
    else:
        indices = indices[:k]


    eigenfaces = normalize(np.dot(A, vectors), axis=0, norm='l1')
    eigenfaces = np.matrix(eigenfaces[:, indices])
    eigenfaces = np.matrix(normalize(np.dot(eigenfaces.T, A), axis=0, norm='l1'))

    return eigenfaces

def similarity(weights):
    print('computing similarity matrix')
    numRows = (weights.shape)[0]
    numCols = (weights.shape)[1]
    similarityMatrix = np.zeros((numCols, numCols), dtype=np.float)

    for i in range(0, numCols):
        for j in range(0, numCols):
            similarityMatrix[i,j] = sp.spatial.distance.cosine(weights[:,i], weights[:,j])#np.power(np.linalg.norm(weights[:, i] - weights[:, j]), 2)

    similarityMatrix = 1-(MinMaxScaler().fit_transform(similarityMatrix))
    #similarityMatrix = kneighbors_graph(similarityMatrix, 5, mode='connectivity', include_self=True)
    return similarityMatrix#similarityMatrix 

#data post-processing
def groupLabels(clusterLabels):
    indices = dict()
    for i in range(0, len(clusterLabels)):
        if clusterLabels[i] not in indices:
            indices[clusterLabels[i]] = [i]
        else:
            indices[clusterLabels[i]].append(i)
    return indices

def showFaces(faces, labels):
    indices = groupLabels(labels)
    for k,v in indices.iteritems():
        for i in v:
            cv2.imshow('img', faces[i])
            raw_input("cluster " + str(k))

def graphSubClusters(faces, similarityMatrix, labels):
    for k,v in groupLabels(labels).iteritems():
        print('cluster {0}'.format(k))
        nx.draw(nx.from_numpy_matrix(np.matrix(similarityMatrix)[v, :][:, v]))
        plt.show()

def cleanClusters(faces, similarityMatrix, labels):
    #first remove outlying clusters
    indices = groupLabels(labels)
    inter_cluster_variances = list()
    toRemove = list()
    for k,v in indices.iteritems():
        print('cluster {0}'.format(k))
        inter_cluster_variances.append(sum(sum(np.power(similarityMatrix[:, v][v, :], 2), 1))/(len(v)-1))

    inter_cluster_zscores = zscore(inter_cluster_variances)
    toRemove_inter_cluster = list()
    for index in range(0, len(inter_cluster_zscores)):
        if inter_cluster_zscores[index] <= (-1) or inter_cluster_zscores[index] >= 1:
            toRemove_inter_cluster.append(index)

    for i in toRemove_inter_cluster:
        toRemove.extend(indices.pop(i, None))

    similarityMatrix = np.delete(similarityMatrix, toRemove, 0)
    similarityMatrix = np.delete(similarityMatrix, toRemove, 1)
    labels = np.delete(labels, toRemove)
    faces = np.delete(np.array(faces), toRemove, 0)
    print(inter_cluster_zscores, toRemove_inter_cluster)

    #them remove the individual images
    silhouetteSamples = zscore(silhouette_samples(similarityMatrix, labels, metric='precomputed'))
    print(silhouetteSamples)
    below = (silhouetteSamples <= (-1))
    above = (silhouetteSamples >= 1)
    toRemove = list()
    for index in range(0, len(below)):
        if below[index]:
            toRemove.append(index)

    for index in range(0, len(above)):
        if above[index]:
            toRemove.append(index)

    toRemove.sort()
    print('toRemove', toRemove)
    
    similarityMatrix = np.delete(similarityMatrix, toRemove, 0)
    similarityMatrix = np.delete(similarityMatrix, toRemove, 1)
    labels = np.delete(labels, toRemove)
    faces = np.delete(np.array(faces), toRemove, 0)
    return (faces, similarityMatrix, labels)

#writing data to files
def saveToFile(faces, labels, pretense='original'):
    for k,v in groupLabels(labels).iteritems():
        print('cluster {0}'.format(k))
        for i in v:
            Image.fromarray(faces[i]).save('cluster{0}_{1}_{2}.jpg'.format(k,i, pretense))

faces = getFaces(getFiles())
faceMatrix = createFaceMatrix(faces)
eigfaces = eigenFaces(faceMatrix)
similarityMatrix = similarity(eigfaces)

labels = AffinityPropagation().fit_predict(similarityMatrix)
nx.draw(nx.from_numpy_matrix(similarityMatrix))
plt.show()
graphSubClusters(faces, similarityMatrix, labels)
showFaces(faces, labels)
saveToFile(faces, labels)

(faces, similarityMatrix, labels) = cleanClusters(faces, similarityMatrix, labels)
nx.draw(nx.from_numpy_matrix(similarityMatrix))
plt.show()
graphSubClusters(faces, similarityMatrix, labels)
showFaces(faces, labels)
saveToFile(faces, labels, pretense='cleaned')
