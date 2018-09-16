import numpy as np
import h5py
import sys
import sklearn
from sklearn.neighbors import KDTree
import pdb
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import random
from itertools import cycle
import time

distance = 0.5
pointLimit = 128
groundLevel = -1.64

def readFile(fileName):
	pointCloud = []
	inputFile = open(fileName,"r")
	for line in inputFile:
		try:
			floats=[float(x) for x in line.split(" ")]
			if floats[2] > groundLevel and floats[0] < 40 and floats[0] > -40 and floats[1] < 40 and floats[1] > -40:
				pointCloud.append(floats[0:3])
		except Exception as e:
			pass
	return pointCloud

def filterGround(pointCloud):
	temp = []
	for point in pointCloud:
		if point[2] > groundLevel and point[0] < 3500 and point[0] > -3500 and point[1] < 3500 and point[1] > -3500:
			temp.append(point)
	return temp


def getNeighbors(tree,pointCloud):
	clusters = []
	neighborhoods = tree.query_radius(pointCloud,r=distance)
	for array in neighborhoods:
		clusters.append(array.tolist())
	return clusters

def to_graph(neighbors_clusters):
	G=nx.Graph()
	for neighbors in neighbors_clusters:
		G.add_nodes_from(neighbors)
		G.add_edges_from(to_edges(neighbors))
	return G
	
def to_edges(neighbors):
	it = iter(neighbors)
	last = next(it)
	for current in it:
		yield last,current
		last = current

def formatDataSize(pointClusters,pointCloud):
	formatedClusters = []
	labels = []
	for points in pointClusters:
		if len(points) >= 128:
			formatedClusters.append(resize(points,pointCloud))
			labels.append(0)
		elif 128 > len(points) and len(points) > 20:
			formatedClusters.append(resize(points,pointCloud))
			labels.append(0)
		else:
			pass	
	pdb.set_trace()
	return formatedClusters, labels

def formatData(pointClusters,pointCloud):
	formatedCluster = []
	allClusters = []
	for points in pointClusters:
			formatedCluster = []
			for point in points:
				formatedCluster.append(pointCloud[point])
			allClusters.append(formatedCluster)
	return allClusters

def resize(model,pointCloud):
	if len(model) > 128:
		for x in range(int(len(model)/pointLimit)):
			newSet = random.sample(model,pointLimit)
	else:
		newSet = model
	formatedCloud = []
	for point in newSet:
		formatedCloud.append(pointCloud[point])
	return formatedCloud

def retrieveData(inputFile):
	pointCloud = readFile(inputFile)
	#pointCloud = filterGround(pointCloud)
	current_time = time.time()
	tree = KDTree(pointCloud)
	print("Tree build time:%f",(time.time()-current_time))
	current_time = time.time()
	Graph=to_graph(getNeighbors(tree,pointCloud))
	print("Graph build time:%f",(time.time()-current_time))
	subGraph = list(nx.connected_components(Graph))
	return formatDataSize(subGraph,pointCloud)

def getClusters(inputFile):
	pointCloud = readFile(inputFile)
	current_time = time.time()
	tree = KDTree(pointCloud)
	print("Tree build time:%f",(time.time()-current_time))
	current_time = time.time()
	Graph=to_graph(getNeighbors(tree,pointCloud))
	print("Graph build time:%f",(time.time()-current_time))
	subGraph = list(nx.connected_components(Graph))
	return formatData(subGraph,pointCloud)

def importData(inputFile):
	formatedList = []
	orderList = []
	data,labels = retrieveData(inputFile)
	for cluster in data:
		if len(cluster) == 128:
			X = (max([point[0] for point in cluster]) + min([point[0] for point in cluster])) / 2
			Y = (max([point[1] for point in cluster]) + min([point[1] for point in cluster])) /2
			Z = (max([point[2] for point in cluster]) + min([point[2] for point in cluster])) / 2
			centeredCoordinates = [(point[0]-X,point[1]-Y,point[2]-Z) for point in cluster]
			formatedList.append(centeredCoordinates)
			orderList.append(data.index(cluster))
	return np.array(formatedList), orderList, data



def printInfo():
	start_time = time.time()
	data,labels = retrieveData()
	print("Total time for retrieve:%f",(time.time()-start_time))
	a=0
	for x in data:
		a= a+ len(x)		
	print(a)
	print(len(data))

def convertToNumpy2D(data):
	array = np.array([[0,0,0]])
	objLen = []
	for x in range(len(data)):
		objLen.append(len(data[x]))
		temp=np.asarray(data[x],dtype=np.float32)
		array = np.concatenate((array,temp),axis=0)
	return array, objLen


# Create the scatter plot
def colorMaping(predLabels,data):
	
	colorMap = np.array([[1,1,1]])
	n=0	
	for obj in data:
		if len(obj) == 128:
			if predLabels[n] == 1:
				color = np.array([[1, 1, 0]] * len(obj))
				colorMap = np.concatenate((colorMap,color),axis=0)
			elif predLabels[n] == 2:
				color = np.array([[0, 1, 1]] * len(obj))
				colorMap = np.concatenate((colorMap,color),axis=0)
			elif predLabels[n] == 3:
				color = np.array([[1, 0, 1]] * len(obj))
				colorMap = np.concatenate((colorMap,color),axis=0)
			else:
				color = np.array([[1, 1, 1]] * len(obj))
				colorMap = np.concatenate((colorMap,color),axis=0)
			n += 1
		else:
			color = np.array([[1, 1, 1]] * len(obj))
			colorMap = np.concatenate((colorMap,color),axis=0)
	return colorMap

def pointCloudToFile(pointCloud,fileName):
	output = open(fileName,"w")
	output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(pointCloud)) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(pointCloud))+"\nDATA ascii\n")
	for point in pointCloud:
		output.write(str(point[0])+" "+str(point[1])+" "+str(point[2])+"\n")
	output.close()


