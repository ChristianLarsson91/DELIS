#!/usr/bin/env python
import numpy as np
import sys
import pdb
import h5py
# Parameters


class lidar_parameters:
    vertical_layers = 64  # amount
    horizontal_layers = 5401  # amount
    max_range = 60  # meters
    data_rate = 2.2*10**6  # points/sec
    horizontal_freq = 10.0  # Hz
    horizontal_coverage = 360.0  # degrees
    vertical_coverage = 26.9  # degrees
    readings_per_degree = 12.5  # @10Hz


def read_lidar_data(lidar=lidar_parameters, lidar_filename=sys.argv[1]):
    with open(lidar_filename) as f:
        all_lidar_data = []
        for line in f:
            if line.startswith("%"):
                break
            current_lidar_data = []
            all_lidar_data.append(current_lidar_data)
            line = line.strip(' \t\r\n').split("\t")
            current_lidar_data.append(line[0])
            data = np.array([float(i) for i in line[2].split(" ")])
            current_lidar_data.append(
                    data.reshape(
                        lidar.vertical_layers, lidar.horizontal_layers))
    return all_lidar_data


def transform_2_cartesian(polar_lidar_data, lidar=lidar_parameters):

    x = 8.5
    y = -9478.03
    # z = 0.66

    lx = 8.5
    ly = -9477.15
    lz = 2.0  # 2.34

    x_pos = lx - x
    y_pos = ly - y
    z_pos = lz

    theta = np.radians(90+(lidar.vertical_coverage/lidar.vertical_layers)*np.arange(-lidar.vertical_layers/2,lidar.vertical_layers/2,1))[::-1]
    phi = np.radians(
        lidar.horizontal_coverage / 2 - np.arange(lidar.horizontal_layers) *
        lidar.horizontal_coverage / lidar.horizontal_layers)

    cartesian_lidar_data = np.zeros(
        (len(polar_lidar_data),
         lidar.vertical_layers * lidar.horizontal_layers,
         3))
    for frame_number in range(len(polar_lidar_data)):
        a = np.multiply(polar_lidar_data[frame_number][1],np.cos(phi))
        x = x_pos + np.multiply(a.transpose(),np.sin(theta)).transpose()
        x = x.reshape(lidar.vertical_layers * lidar.horizontal_layers, 1)
        cartesian_lidar_data[frame_number, :, 0] = x.squeeze()

        y = y_pos + np.multiply(
            np.multiply(polar_lidar_data[frame_number][1],np.sin(phi)).transpose(),np.sin(theta)).transpose()
        y = y.reshape(lidar.vertical_layers * lidar.horizontal_layers, 1)
        cartesian_lidar_data[frame_number, :, 1] = y.squeeze()

        z = z_pos + np.multiply(polar_lidar_data[frame_number]
                                [1].transpose(), np.cos(theta)).transpose()
        z = z.reshape(lidar.vertical_layers * lidar.horizontal_layers, 1)
        cartesian_lidar_data[frame_number, :, 2] = z.squeeze()

    return cartesian_lidar_data

def outputPCDFile(pointCloud):
    output=open(sys.argv[2],"w")
    output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(pointCloud[0])) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(pointCloud[0]))+"\nDATA ascii\n")
    for node in range(len(pointCloud[0]))  :
        output.write(str(pointCloud[0][node][0])+" "+str(pointCloud[0][node][1])+" "+str(pointCloud[0][node][2])+"\n")
    output.close()

if __name__ == '__main__':
    lidar = lidar_parameters
    polar_lidar_data = read_lidar_data(lidar)
    cartesian_lidar_data = transform_2_cartesian(polar_lidar_data, lidar)
    #sys.stdout.write(cartesian_lidar_data)
    f = h5py.File("lidar.h5","w")
    f.create_dataset("data",data=cartesian_lidar_data[0],dtype=float)
    outputPCDFile(cartesian_lidar_data)
