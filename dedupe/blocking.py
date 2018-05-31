#!/usr/bin/python
# -*- coding: utf-8 -*-

from future.utils import viewvalues

from collections import defaultdict
import logging
import time

import numpy

logger = logging.getLogger(__name__)


class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''

    def __init__(self, predicates):

        self.predicates = predicates

        self.index_fields = defaultdict(lambda: defaultdict(list))

        for full_predicate in predicates:
            for predicate in full_predicate:
                if hasattr(predicate, 'index'):
                    self.index_fields[predicate.field][predicate.type].append(
                        predicate)

    def __call__(self, records, target=False):

        start_time = time.clock()
        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]

        for i, record in enumerate(records):
            record_id, instance = record

            for pred_id, predicate in predicates:
                block_keys = predicate(instance, target=target)
                for block_key in block_keys:
                    yield block_key + pred_id, record_id

            if i and i % 10000 == 0:
                logger.info('%(iteration)d, %(elapsed)f2 seconds',
                            {'iteration': i,
                             'elapsed': time.clock() - start_time})

    def resetIndices(self):
        # clear canopies to reduce memory usage
        for index_type in self.index_fields.values():
            for predicates in index_type.values():
                for predicate in predicates:
                    predicate.index = None
                    if hasattr(predicate, 'canopy'):
                        predicate.canopy = {}
                    if hasattr(predicate, '_cache'):
                        predicate._cache = {}

    def index(self, data, field):
        '''Creates TF/IDF index of a given set of data'''
        indices = extractIndices(self.index_fields[field])

        for doc in data:
            if doc:
                for _, index, preprocess in indices:
                    index.index(preprocess(doc))

        for index_type, index, _ in indices:

            index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                logger.debug("Canopy: %s", str(predicate))
                predicate.index = index

    def unindex(self, data, field):
        '''Remove index of a given set of data'''
        indices = extractIndices(self.index_fields[field])

        for doc in data:
            if doc:
                for _, index, preprocess in indices:
                    index.unindex(preprocess(doc))

        for index_type, index, _ in indices:

            index._index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                logger.debug("Canopy: %s", str(predicate))
                predicate.index = index

    def indexAll(self, data_d):
        for field in self.index_fields:
            unique_fields = {record[field]
                             for record
                             in viewvalues(data_d)
                             if record[field]}
            self.index(unique_fields, field)


def extractIndices(index_fields):

    indices = []
    for index_type, predicates in index_fields.items():
        predicate = predicates[0]
        index = predicate.index
        preprocess = predicate.preprocess
        if predicate.index is None:
            index = predicate.initIndex()
        indices.append((index_type, index, preprocess))

    return indices

def block_sizes(blocker, data_d):
    sizes = defaultdict(int)
    for k, record_id in blocker(data_d.items()):
        sizes[k] += 1

    ret = {}
    for k, n in sizes.items():
        key,_,i = k.rpartition(':')
        key = '{}:{}:{}'.format(key, i, blocker.predicates[int(i)])
        ret[key] = n
    return ret

def print_block_sizes_info(sizes, name, show_n_largest=10):
    sizes_sorted = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

    logger.info("{}:".format(name))
    logger.info("  blocks: {:,}".format(len(sizes)))
    logger.info("  mean size: {:.1f}".format(numpy.mean(list(sizes.values()))))
    for percentile in (25, 50, 75, 95, 99):
        logger.info("  {}th percentile: {}".format(
            percentile,
            numpy.percentile(list(sizes.values()), percentile)))
    logger.info("  largest {} blocks:".format(show_n_largest))
    for i in range(show_n_largest):
        k, size = sizes_sorted[i]
        logger.info("    {}: {:,}".format(repr(k), size))

def print_blocker_stats(blocker, data_d, data_d2=None):
    sizes1 = block_sizes(blocker, data_d)
    print_block_sizes_info(sizes1, 'Block stats for data1')

    pair_sizes = {}
    if data_d2 is not None:
        sizes2 = block_sizes(blocker, data_d2)
        print_block_sizes_info(sizes2, 'Block stats for data2')

        for k, n1 in sizes1.items():
            n2 = sizes2.get(k, 0)
            if not n2:
                continue
            pair_sizes[k] = n1 * n2
    else:
        pair_sizes = {k: n*n for k, n in sizes1.items()}

    print_block_sizes_info(pair_sizes, 'Blocked pairs to be scored')

    logger.info('')
    logger.info('TOTAL BLOCKED PAIRS: {:,}'.format(
        numpy.sum(list(pair_sizes.values()))))
    logger.info('')

def print_blocker_recall(blocker, training_pairs):
    labels = []
    for record_1, record_2 in training_pairs.get('match', []):
        for predicate in blocker.predicates:
                keys = predicate(record_1)
                if keys:
                    if set(predicate(record_2, target=True)) & set(keys):
                        labels.append(1)
                        break
        else:
            labels.append(0)

    logger.info('ESTIMATED BLOCKING RECALL: {}'.format(sum(labels) / len(labels)))
    logger.info('')
