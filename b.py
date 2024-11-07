a = [1, 32, 64, 1]

for in_features, out_features in zip(a, a[1:]):
    print(in_features, out_features)
