from lloyd import Field
from scipy.spatial import voronoi_plot_2d
import numpy as np
import dlib
import os
import warnings
from multiprocessing import Process
import cv2
import sys
from scipy.spatial import Delaunay
from functools import cmp_to_key
import matplotlib.pyplot as plt
import sharedmem

# input image file
input = ""
# output image file
output = ""
center = np.zeros((2,))
switch = '0 : OFF   1 : ON'
douglas_epsilon = "douglas epsilon"
douglas_distance = "douglas threshold"
median_mode = "0 : Mean   1 : Median"
save_edge = "Save edges"
show_lloyd = "Show Lloyd"
use_lloyd = "Use Lloyd"

def nothing(x):
    pass
# create named window to display image and trackbar
cv2.namedWindow('image')
cv2.namedWindow('trackbar')
# create trackbars to control various parameters
cv2.createTrackbar('Sample Points','trackbar',1,3000,nothing)
cv2.createTrackbar(douglas_epsilon,'trackbar',1,100,nothing)
cv2.createTrackbar(douglas_distance,'trackbar',1,100,nothing)
cv2.createTrackbar(median_mode, 'trackbar',0,1,nothing)
cv2.createTrackbar(use_lloyd, 'trackbar',0,1,nothing)
cv2.createTrackbar(save_edge, 'trackbar',0,1,nothing)
cv2.createTrackbar(show_lloyd, 'trackbar',0,1,nothing)
cv2.createTrackbar(switch, 'trackbar',0,1,nothing)

# Download the file from - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Exxtract the file in the same directory as the code
predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")

# function divides the data into equal parts for each process
def divideData(tri, length):
    for i in range(0, len(tri), length):
        yield tri[i:i + length]

# draws the output image with appropriate coloring scheme
def draw(triIndices, tri, meshImage, inputImage):
    coloringScheme = cv2.getTrackbarPos(median_mode,'trackbar')
    for index in triIndices:
        if coloringScheme == 0:
            meshImage[tri==index, :] = np.mean(inputImage[tri==index, :], axis=0)
        else:
            meshImage[tri==index, :] = np.median(inputImage[tri==index, :].reshape(-1, 3), axis=0)            

# divides the rendering part to 4 processes and completes the low poly image
def getMeshImage(tri, inputImage):
    allPoints = np.transpose(np.where(np.ones(inputImage.shape[:2])))
    triIndices = tri.find_simplex(allPoints)
    triIndices = triIndices.reshape(inputImage.shape[:2])
    temp = np.unique(triIndices)
    data = list(divideData(temp, len(temp) // 4))
    meshImage = sharedmem.empty(inputImage.shape)
    meshImage.fill(0)
    processes = [Process(target=draw, args=(data[i], triIndices, meshImage, inputImage)) for i in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    meshImage = meshImage.astype(np.uint8)
    return meshImage

# calculates perpendicular distance between a point and a line given by p2 and p3
def perpendicularDistance(p1,p2,p3):
    return  np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

# calculates the distance between 2 points
def distance(p1,p2):
    return np.linalg.norm(p1-p2)

# implements the Douglas-Peucker algorithm
def DouglasPeucker(PointList, epsilon, threshold):
    if len(PointList)<=2 :
        return PointList
    dmax = 0
    index = 0
    end = len(PointList)
    for i in range(1,end-1):
        d = perpendicularDistance(PointList[i],PointList[0],PointList[end-1])
        if d>dmax:
            index = i
            dmax = d
    ResultList = []
    if dmax>epsilon:
        recResult1 = DouglasPeucker(PointList[:index+1],epsilon,threshold)
        recResult2 = DouglasPeucker(PointList[index:end],epsilon,threshold)
        ResultList = recResult1[:-1] + recResult2
    else:
        if distance(PointList[0],PointList[end-1]) > threshold:
            mid = end//2
            recResult1 = DouglasPeucker(PointList[:mid+1],epsilon,threshold)
            recResult2 = DouglasPeucker(PointList[mid:end],epsilon,threshold)
            ResultList = recResult1[:-1] + recResult2
        else:
            ResultList.append(PointList[0])
            ResultList.append(PointList[end-1])
    return ResultList

# orders the points in edges in a clockwise direction so as to be passed to douglas-peucker
def order_points_clockwise(a, b):
    if a[0] - center[0] >= 0 and b[0] - center[0] < 0 :
        return 1
    if a[0] - center[0] < 0 and b[0] - center[0] >= 0 :
        return -1
    if a[0] - center[0] == 0 and b[0] - center[0] == 0 :
        if a[1] - center[1] >= 0 or b[1] - center[1] >= 0:
            if a[1] > b[1]:
                return 1
            else:
                return -1
        if b[1] > a[1]:
            return 1
        else:
            return -1;
    det = (a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1])
    if det < 0:
        return 1
    if det > 0:
        return -1
    d1 = (a[0] - center[0]) * (a[0] - center[0]) + (a[1] - center[1]) * (a[1] - center[1])
    d2 = (b[0] - center[0]) * (b[0] - center[0]) + (b[1] - center[1]) * (b[1] - center[1])
    if d1 > d2:
        return 1
    else:
        return -1

# get the coordinates of all the points in the contour
def getContourPoints(contour):
    points = []
    for elem in contour:
        pt = np.array([elem[0][1],elem[0][0]])
        points.append(pt)
    if len(points)>1:
        points.sort(key=cmp_to_key(order_points_clockwise))
    return points

# Detects edge
# samples points from edges
# generates additional random sample points
# Performs delaunay triangulation
def constrainedTriangulation(im, gray_image, lowerThresh, upperThresh):
    cannyEdges = cv2.Canny(gray_image, lowerThresh, upperThresh)
    saveEdge = cv2.getTrackbarPos(save_edge,'trackbar')
    if saveEdge==1:
        cv2.imwrite('canny.jpg',cannyEdges)
    
    _,contours,_ = cv2.findContours(cannyEdges, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    epsilon = cv2.getTrackbarPos(douglas_epsilon,'trackbar')
    distanceThreshold = cv2.getTrackbarPos(douglas_distance,'trackbar')
    
    pts = []
    for contour in contours:
        points = getContourPoints(contour)
        pts = pts + DouglasPeucker(points,epsilon,distanceThreshold)
    
    r_max,c_max = im.shape[0],im.shape[1]
    pts = np.vstack([pts, [0, 0]])
    pts = np.vstack([pts, [0, c_max-1]])
    pts = np.vstack([pts, [r_max-1, 0]])
    pts = np.vstack([pts, [r_max-1, c_max-1]])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    dets = detector(im, 1)
    for _, d in enumerate(dets):
        shape = predictor(im, d)
        for i in range(shape.num_parts):
            pts = np.vstack([pts, [shape.part(i).x, shape.part(i).y]])
    
    samplePoints = cv2.getTrackbarPos('Sample Points','trackbar')
    samplePts = np.random.randint(0, 750, size=(samplePoints, 2))
    
    useLlyod = cv2.getTrackbarPos(use_lloyd,'trackbar')
    if useLlyod==1:
        showLloyd = cv2.getTrackbarPos(show_lloyd,'trackbar')
        field = Field(samplePts)
        if showLloyd==1:
            fig = voronoi_plot_2d(field.voronoi, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=2)
            plt.show()

        for i in range(1,10):
            field.relax()
        samplePts = field.get_points()

        if showLloyd==1:
            fig = voronoi_plot_2d(field.voronoi, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=2)
            plt.show()

    pts = np.vstack([pts, samplePts])
    tri = Delaunay(pts, incremental=True)
    tri.close()
    return tri

# resizes image if it is larger than 700x700 for fast computation
def resizeImage(image,size=700):
    if size < np.max(image.shape[:2]):
        scale = size / float(np.max(image.shape[:2]))
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale,interpolation=cv2.INTER_AREA)
    return image

# Function calls all other functions for low poly rendering
def start(inputfile, outputfile):
    print("Starting...")
    inputImage = cv2.imread(inputfile)
    inputImage = resizeImage(inputImage)
    
    center[0],center[1] = inputImage.shape[0],inputImage.shape[1]
    
    noiselessImage = cv2.fastNlMeansDenoisingColored(inputImage, None, 10, 10, 7, 21)
    
    ycbcrImage = cv2.cvtColor(noiselessImage, cv2.COLOR_RGB2YCrCb)
    ycbcrImage = ycbcrImage[:, :, 0]
    highThreshValue,_ = cv2.threshold(ycbcrImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThreshValue = 0.5 * highThreshValue
    
    grayImage = cv2.cvtColor(noiselessImage, cv2.COLOR_BGR2GRAY)
    blurImage = cv2.GaussianBlur(grayImage, (0, 0), 3)
    weightedImage = cv2.addWeighted(grayImage, 2.5, blurImage, -1, 0)
    
    tri = constrainedTriangulation(inputImage, weightedImage, lowThreshValue, highThreshValue)
    meshImage = getMeshImage(tri, inputImage)
    
    print("Complete")
    cv2.imwrite(outputfile, meshImage)
    cv2.imshow('image',meshImage)
    cv2.waitKey(0)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    input = 'Input/'+sys.argv[1]
    output = sys.argv[2]
    inputImage = cv2.imread(input)
    inputImage = resizeImage(inputImage)
    while True:
        cv2.imshow('image',inputImage)
        k = cv2.waitKey(1) & 0xFF
        if k==27:
            break
        s = cv2.getTrackbarPos(switch,'trackbar')
        if s==1:
            start(input,output)
