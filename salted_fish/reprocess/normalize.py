
def fit_single_feature(feature):
    value_count = {}
    total_count = 0
    for x in feature:
        total_count += 1
        if x not in value_count:
            value_count[x] = 0
        value_count[x] += 1

    value_count_pair = sorted(value_count.items(), key=lambda p: p[0])

    cumulative_count = 0
    cdf = {}
    for value, count in value_count_pair:
        cumulative_count += count
        cdf[value] = 1.0 * cumulative_count / total_count

    return sorted(cdf.items(), key=lambda p: p[0])


def get_cdf(x, value_cdf):
    res = 0
    for value, cdf in value_cdf:
        if x < value:
            break
        res = cdf
    return res


def transform_single_feature(value_cdf, feature):
    return [get_cdf(x, value_cdf) for x in feature]


class NormalizeWithCDF(object):

    def __init__(self):
        self.value_cdfs = []

    def fit(self, features):
        self.value_cdfs = [fit_single_feature(f) for f in features]

    def transform(self, features):
        res = []
        for i in range(0, len(features)):
            f = features[i]
            cdf = self.value_cdfs[i]
            single_res = transform_single_feature(cdf, f)
            res.append(single_res)
        return np.array(res)

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)


if __name__ == "__main__":

    import numpy as np

    normal = NormalizeWithCDF()
    arr1 = np.random.uniform(0, 1, 10)
    arr2 = np.random.uniform(0, 1, 10)
    print(arr1)
    print(normal.fit_transform([arr1, arr2]))
