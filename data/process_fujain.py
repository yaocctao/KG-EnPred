import sys, os
from pathlib import Path
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict

def prepare_data(args):
    entities, relations, timestamps = set(), set(), set()
    files = ['train', 'valid', 'test']
    if not os.path.exists(os.path.join(args.path, args.name)):
        os.makedirs(os.path.join(args.path, args.name))
    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(args.src_path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in tqdm(to_read.readlines()):
            lhs, rel, rhs, ts = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(ts)
            try:
                examples.append([lhs, rel, rhs, ts])
            except ValueError:
                continue
        out = open(Path(args.path) / args.name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")
    
    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(args.path) / args.name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + len(relations), ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)
            
    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in tqdm(to_skip.items()):
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(args.path) / args.name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()
    
    examples = pickle.load(open(Path(args.path) / args.name / 'train.pickle', 'rb'))
    n_entities = len(entities)
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs, _ts in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(args.path) / args.name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='fujian', help='Name of the dataset')
    parser.add_argument('--path', type=str, default='data', help='Path to save the dataset')
    parser.add_argument('--src_path', type=str, default='src_data/fujian', help='Path to the source data')
    args = parser.parse_args()
    prepare_data(args)
    print("Done")