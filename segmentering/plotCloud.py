import numpy as np
import h5py
import sys
import sklearn
from sklearn.neighbors import KDTree
import pdb
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import matplotlib.pyplot as plt
import random
from vispy import scene, visuals, app, gloo, io
from itertools import cycle
import time
import pdb
from vispy.visuals.transforms import STTransform, NullTransform
#import joindata

groundLevel = -1.64
distance = 0.5
pointLimit = 128

class BBox(object):
	pos = 0
	def __init__(self, arg):
		super(BBox, self).__init__()
		self.arg = arg
	def set_pos(coordinats):
		self.pos = coordinats

def readFile(fileName):
	pointCloud = []
	inputFile = open(fileName,"r")
	for line in inputFile:
		try:
			floats=[float(x) for x in line.split(" ")]
			if floats[2]>groundLevel and floats[0] < 40 and floats[0] > -40 and floats[1] < 40 and floats[1] > -40 :
				pointCloud.append(floats[0:3])
		except Exception as e:
			pass
	return pointCloud


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

def formatData(pointClusters,pointCloud):
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
	return (formatedClusters, labels)

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

def retrieveData():
	pointCloud = readFile(sys.argv[1])
	tree = KDTree(pointCloud)
	Graph=to_graph(getNeighbors(tree,pointCloud))
	subGraph = list(nx.connected_components(Graph))
	return formatData(subGraph,pointCloud)

def exportData():
	formatedList = []
	orderList = []
	data,labels = retrieveData()
	for cluster in data:
		if len(cluster) == 128:
			X = (max([point[0] for point in cluster]) + min([point[0] for point in cluster])) / 2
			Y = (max([point[1] for point in cluster]) + min([point[1] for point in cluster])) /2
			Z = (max([point[2] for point in cluster]) + min([point[2] for point in cluster])) / 2
			centeredCoordinates = [(point[0]-X,point[1]-Y,point[2]-Z) for point in cluster]
			formatedList.append(centeredCoordinates)
			orderList.append(data.index(cluster))
	return np.array(formatedList), orderList






def printInfo():
	start_time = time.time()
	data,labels = retrieveData()
	print(time.time()-start_time)
	a=0
	for x in data:
		a= a+ len(x)		
	print(a)
	print(len(data))

def convertToNumpy2D(data):
	array = np.array([[0,0,0]])
	for x in range(len(data)):
		temp=np.asarray(data[x],dtype=np.float32)
		array = np.concatenate((array,temp),axis=0)
	return array



canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
fov = 60.
cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov)
cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov)
cam3 = scene.cameras.ArcballCamera(parent=view.scene, fov=fov)
view.camera = cam1

# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global opaque_cmap, translucent_cmap
    if event.text == '1':
        cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
        view.camera = cam_toggle.get(view.camera, 'fly')
    elif event.text == '2':
        methods = ['mip', 'translucent', 'iso', 'additive']
        method = methods[(methods.index(volume1.method) + 1) % 4]
        print("Volume render method: %s" % method)
        cmap = opaque_cmap if method in ['mip', 'iso'] else translucent_cmap
        volume1.method = method
        volume1.cmap = cmap
        volume2.method = method
        volume2.cmap = cmap
    elif event.text == '3':
        volume1.visible = not volume1.visible
        volume2.visible = not volume1.visible
    elif event.text == '4':
        if volume1.method in ['mip', 'iso']:
            cmap = opaque_cmap = next(opaque_cmaps)
        else:
            cmap = translucent_cmap = next(translucent_cmaps)
        volume1.cmap = cmap
        volume2.cmap = cmap
    elif event.text == '0':
        cam1.set_range()
        cam3.set_range()
    elif event.text != '' and event.text in '[]':
        s = -0.025 if event.text == '[' else 0.025
        volume1.threshold += s
        volume2.threshold += s
        th = volume1.threshold if volume1.visible else volume2.threshold
        print("Isosurface threshold: %0.3f" % th)


# Create the scatter plot
def colorMaping(predLabels,obj_length):
	
	colorMap = np.array([[1,1,1]])
	for index,label in enumerate(predLabels):
		if label == 0:
			color = np.array([[1, 0, 0]] * obj_length[index])
			colorMap = np.concatenate((colorMap,color),axis=0)
		elif label == 1:
			color = np.array([[0, 1, 0]] * obj_length[index])
			colorMap = np.concatenate((colorMap,color),axis=0)
		elif label == 2:
			color = np.array([[0, 0.7, 1]] * obj_length[index])
			colorMap = np.concatenate((colorMap,color),axis=0)
		else:
			color = np.array([[1, 1, 1]] * obj_length[index])
			colorMap = np.concatenate((colorMap,color),axis=0)
	return colorMap

def outputPCDFiles(data):
	fileName = 0
	for cluster in data:
		if len(cluster) == 128:
			fileName += 1
			output=open((str(fileName)+".pcd"),"w")
			output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(cluster)) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(cluster))+"\nDATA ascii\n")
			for node in cluster:
				output.write(str(node[0])+" "+str(node[1])+" "+str(node[2])+"\n")
			output.close()
	

# build visuals

#Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
#posBB = joindata.main().traspose()

#Plot3D(BBox, width=2.0, color='red', edge_color='w',parent=view.scene)


file = h5py.File(sys.argv[1],'r')
obj_length = list(file['obj_length'])
predLabels = list(file['label'])
all_clusters = file['all_segmented_points']
all_clusters = np.concatenate((all_clusters,np.array([[0,0,0]])),axis=0)

scatter = scene.visuals.Markers()
scatter.set_data(all_clusters[:,:3], face_color=colorMaping(predLabels,obj_length),size=2)

view.add(scatter)
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range()
app.run()

