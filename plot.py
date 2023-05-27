import argparse
import os

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('filename', metavar='FILE', type=str)
args = parser.parse_args()

if not os.path.exists(args.filename):
    print("Error while reading file", args.filename)
    exit(-1)

f = open(args.filename, "r")
datalines = f.readlines()
l = list(map(float, datalines[0].split()))
f.close()

fig, ax = plt.subplots()

total_line, = ax.plot(l, range(1, len(l) + 1), label="TOTAL")

lines = [total_line]

for i in range(3):
    l = list(map(float, datalines[i+1].split()))
    f.close()

    line, = ax.plot(l, range(1, len(l) + 1), label="OUT №" + str(i+1), linestyle='--')
    lines.append(line)

leg = ax.legend(handles=lines, loc='upper left')
ax.add_artist(leg)

plt.show()