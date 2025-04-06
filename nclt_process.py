import os
import glob
import struct
import numpy as np


def convert_nclt(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z


def load_velodyne_binary_nclt(filename):
    f_bin = open(filename, "rb")
    hits = []
    while True:
        x_str = f_bin.read(2)
        if x_str == b'': #eof
            break
        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert_nclt(x, y, z)

        hits += [[x, y, z, i]]

    f_bin.close()

    hits = np.array(hits)

    return hits

def processing(seq):
    file_list = glob.glob(data_root + seq + '/velodyne_sync/*.bin')
    for file in file_list:
        print(file)
        scan = load_velodyne_binary_nclt(file).astype(np.float32)
        scan.tofile(data_root + seq + '/velodyne_left/' + os.path.split(file)[-1])


if __name__ == '__main__':
    seqs = ['2012-02-18', '2012-02-19', '2012-03-31', '2012-05-11'
            , '2012-02-12', '2012-03-17', '2012-03-25', '2012-05-26']
    # Please modify the ‘xx’ as the path of your dataset  e.g., /home/lw/NCLT/
    data_root = '/home/lw/NCLT/'
    for seq in seqs:
        processing(data_root, seq)