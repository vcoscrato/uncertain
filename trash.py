with open(path+'fitted/R.pkl', 'rb') as f:
    R = pickle.load(f)

with open(path+'fitted/ensemble.pkl', 'rb') as f:
    ensemble = pickle.load(f)

with open(path+'fitted/resample.pkl', 'rb') as f:
    resample = pickle.load(f)

with open(path+'fitted/double.pkl', 'rb') as f:
    double = pickle.load(f)

with open(path+'fitted/linear.pkl', 'rb') as f:
    linear = pickle.load(f)