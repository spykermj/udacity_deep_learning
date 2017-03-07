#!/usr/bin/env python3

import numpy as np
import pickle

pickle_file = 'notMNIST.pickle'

f = open(pickle_file, 'rb')
datasets = pickle.load(f)
f.close()

train_test_exact = 0
train_test_near = 0

train_valid_exact = 0
train_valid_near = 0

train_dataset = datasets['train_dataset']
train_labels = datasets['train_labels']

test_dataset = datasets['test_dataset']
valid_dataset = datasets['valid_dataset']

train_size = len(train_dataset)
test_size = len(test_dataset)
valid_size = len(valid_dataset)

train_hashes = [hash(bytes(x.data)) for x in train_dataset]
train_hashes_unique = set(train_hashes)
valid_hashes = [hash(bytes(x.data)) for x in valid_dataset]
valid_hashes_unique = set(valid_hashes)
test_hashes = [hash(bytes(x.data)) for x in test_dataset]
test_hashes_unique = set(test_hashes)

train_test_exact = len(train_hashes_unique.intersection(test_hashes_unique))
train_valid_exact = len(train_hashes_unique.intersection(valid_hashes_unique))

print('train_test_exact: {} {}%'.format(train_test_exact, 100.0 * train_test_exact / test_size))
print('train_valid_exact: {} {}%'.format(train_valid_exact, 100.0 * train_valid_exact / valid_size))
