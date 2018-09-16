import numpy as np
import seg_lib
import hdbscan
import sys
import pdb
from collections import defaultdict
import random

dictonary = defaultdict(list)

pointCloud = seg_lib.readFile(sys.argv[1])
pointCloud = seg_lib.filterGround(pointCloud)
array = np.asarray(pointCloud)
clusterer = hdbscan.HDBSCAN(min_cluster_size=200,approx_min_span_tree=True,leaf_size=50,
    gen_min_span_tree=True)
clusterer.fit(array)
for i in range(len(clusterer.labels_)):
	dictonary[clusterer.labels_[i]].append(pointCloud[i])

def outputFiles():
	fileName = 0
	for index in range(len(dictonary)):
		if len(dictonary[index]) > 30:
			fileName += 1
			output=open((str(fileName)+"_full_real.pcd"),"w")
			output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(dictonary[index])) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(dictonary[index]))+"\nDATA ascii\n")
			for node in dictonary[index]:
				output.write(str(node[0])+" "+str(node[1])+" "+str(node[2])+"\n")
			output.close()

def outputFiles_128_points():
	fileName = 0
	obj = []
	for index in range(len(dictonary)):
		obj = []
		if len(dictonary[index]) > 30:
			fileName += 1
			output=open((str(fileName)+"_128.pcd"),"w")
			output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(128) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(128)+"\nDATA ascii\n")
			for node in dictonary[index]:
				obj.append(node)

			#pdb.set_trace()
			#numbers = np.arange(len(obj))
			#pdb.set_trace()
			#w = np.exp(numbers/128.)
			#w /= w.sum()
			#point_index = np.random.choice(numbers, size=128, replace=False, p=w)
			#print(point_index)
			#obj = np.random.poisson(np.asarray(obj),128)
			#pdb.set_trace()		
			obj = random.sample(obj,128)
			#pdb.set_trace()
			for x in range(128):
				output.write(str(obj[x][0])+" "+str(obj[x][1])+" "+str(obj[x][2])+"\n")
			output.close()

outputFiles()