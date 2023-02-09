from . import srp_pruner

# when new pruner implementation is added in the 'pruner' dir, update this dict to maintain minimal code change.
# key: pruning method name, value: the corresponding pruner
pruner_dict = {
    'SRP': srp_pruner,
}