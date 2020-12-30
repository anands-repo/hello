# © 2019 University of Illinois Board of Trustees.  All rights reserved
# First mixture-of-experts config
configDict = {
    'readConv': 'MoEReadConvolver250FeatureMap',
    'alleleConvSingle': 'ExpertAlleleConvolver250FeatureMap',
    'graphConvSingle': 'ExpertGraphConvolver250FeatureMap',
    'graphConvHybrid': 'ExpertGraphConvolver250FeatureMap',
    'meta': 'MetaCombiner250FeatureMap',
    'convCombiner': 'ConvCombiner250FeatureMap',
    'noSiteLevelCombiner': True,
    'kwargs': {
        'useAdditive': True,
    }
};
