import itertools

from .base import FieldType
from dedupe import predicates
from simplecosine.cosine import CosineSetSimilarity
from affinegap import normalizedAffineGapDistance as affineGap

class SetType(FieldType):
    type = "Set"

    _predicate_functions = (predicates.wholeSetPredicate,
                            predicates.commonSetElementPredicate,
                            predicates.lastSetElementPredicate,
                            predicates.commonTwoElementsPredicate,
                            predicates.commonThreeElementsPredicate,
                            predicates.magnitudeOfCardinality,
                            predicates.firstSetElementPredicate)

    _index_predicates = (predicates.TfidfSetSearchPredicate,
                         predicates.TfidfSetCanopyPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition):
        super(SetType, self).__init__(definition)

        if 'corpus' not in definition:
            definition['corpus'] = []

        self.comparator = CosineSetSimilarity(definition['corpus'])


class MinDistanceSetType(SetType):
    type = "MinDistanceSet"

    def __init__(self, definition):
        super().__init__(definition)
        if 'corpus' not in definition:
            definition['corpus'] = []

        self.comparator = affine_set_similarity


def affine_set_similarity(string1, string2):
    assert(isinstance(string1, (list, tuple, set)))
    assert(isinstance(string2, (list, tuple, set)))

    closest_distance = None
    for (w1, w2) in itertools.product(string1, string2):
        if (not w1) or (not w2):
            continue

        distance = affineGap(w1, w2)
        if closest_distance is None:
            closest_distance = distance
        else:
            closest_distance = min(distance, closest_distance)

    return closest_distance
