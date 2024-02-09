import sys
import pickle
file = sys.argv[1]


with open(file, "rb") as f:
    print(pickle.load(f))