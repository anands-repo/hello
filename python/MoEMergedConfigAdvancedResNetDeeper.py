# First mixture-of-experts config
configDict = {
    'readConv': 'MoEReadConvolverDeeper',
    'alleleConvSingle': 'ExpertAlleleConvolverDeeper',
    'graphConvSingle': 'ExpertGraphConvolverDeeper',
    'graphConvHybrid': 'ExpertGraphConvolverDeeper',
    'meta': 'MetaCombinerDeeper',
    'convCombiner': 'ConvCombinerResNetDeeper',
    'kwargs': {
        'useAdditive': True,
    }
};
