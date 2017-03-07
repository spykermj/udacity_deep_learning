#!/usr/bin/env python3

import numpy as np
import pickle

pickle_file = 'notMNIST.pickle'

f = open(pickle_file, 'rb')
datasets = pickle.load(f)
f.close()

train_size = len(datasets['train_dataset'])
test_size = len(datasets['test_dataset'])
valid_size = len(datasets['valid_dataset'])

train_test_exact = 0
train_test_near = 0

train_valid_exact = 0
train_valid_near = 0

train_dataset = datasets['train_dataset']
train_labels = datasets['train_labels']

test_dataset = datasets['test_dataset']
valid_dataset = datasets['valid_dataset']

test_duplicate_indices = set()
test_all_indices = set(range(len(test_dataset)))
valid_duplicate_indices = set()
valid_all_indices = set(range(len(valid_dataset)))

for i, train_compare in enumerate(train_dataset):
    print('starting comparison for training dataset index {}'.format(i))
    for j, test_compare in enumerate(test_dataset):
        if np.array_equal(train_compare, test_compare):
            train_test_exact += 1
            test_duplicate_indices.add(j)
        elif np.allclose(train_compare, test_compare):
            train_test_near += 1
            test_duplicate_indices.add(j)
    for k, valid_compare in enumerate(valid_dataset):
        if np.array_equal(train_compare, valid_compare):
            train_valid_exact += 1
            valid_duplicate_indices.add(k)
        elif np.allclose(train_compare, valid_compare):
            train_valid_near += 1
            valid_duplicate_indices.add(k)

print('train_test_exact: {} {}%'.format(train_test_exact, 100.0 * train_test_exact / test_size))
print('train_test_near: {} {}%'.format(train_test_near, 100.0 * train_test_near / test_size))
print('train_valid_exact: {} {}%'.format(train_valid_exact, 100.0 * train_valid_exact / valid_size))
print('train_valid_near: {} {}%'.format(train_valid_near, 100.0 * train_valid_near / valid_size))


valid_keep_indices = list(set(valid_all_indices - valid_duplicate_indices))
test_keep_indices = list(set(test_all_indices - test_duplicate_indices))

valid_dataset = valid_dataset[valid_keep_indices]
valid_labels = datasets['valid_labels'][valid_keep_indices]
test_dataset = test_dataset[test_keep_indices]
test_labels = datasets['test_labels'][test_keep_indices]

save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
}

pickle_file = 'cleaned_{}'.format(pickle_file)

f = open(pickle_file, 'wb')
pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
f.close()
