import numpy as np
import h5py
import seg_lib
import sys

data = seg_lib.getClusters(sys.argv[1])
def outputFiles_128_points():
	fileName = 0
	for obj in data:
		if len(obj) == 128:
			fileName += 1
			output=open((str(fileName)+"_128_NN.pcd"),"w")
			output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(obj)) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(obj))+"\nDATA ascii\n")
			for point in obj:
				output.write(str(point[0])+" "+str(point[1])+" "+str(point[2])+"\n")
			output.close()

def outputFiles():
	fileName = 0
	pdb.set_trace()
	for obj in data:
		if len(obj) > 30:
			fileName += 1
			output=open((str(fileName)+"_NN.pcd"),"w")
			output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(obj)) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(obj))+"\nDATA ascii\n")
			for node in obj:
				output.write(str(node[0])+" "+str(node[1])+" "+str(node[2])+"\n")
			output.close()

outputFiles()