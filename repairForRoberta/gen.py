import ast
import numpy as np

total = list(range(1, 301))

benign = list(range(1, 159))
malicious = list(range(159, 301))

np.random.shuffle(benign)
np.random.shuffle(malicious)

train = benign[:120] + malicious[:120]
valid = benign[120:135] + malicious[120:135]
test = benign[135:] + malicious[135:]

#write to file train.txt valid.txt test.txt
with open('train.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)

with open('valid.txt', 'w') as f:
    for item in valid:
        f.write("%s\n" % item)

with open('test.txt', 'w') as f:
    for item in test:
        f.write("%s\n" % item)
