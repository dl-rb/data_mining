import csv
import random
import sys
import time

from sklearn.cluster import KMeans
import numpy as np


def read_csv(f_in):
    print('Reading data..')
    data = np.genfromtxt(f_in, delimiter=',')
    print('Done')
    # todo random permutation
    data = np.random.permutation(data)
    return data


def write_csv(f_out, data, header):
    with open(f_out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)


def get_size_of_each_cluster(labels):
    cluster_sizes = {}
    for label in labels:
        cluster_sizes[label] = cluster_sizes.setdefault(label, 0) + 1
    return cluster_sizes


def get_cluster_and_mask(data, n_cluster):
    """

    :param data: np array, row = [index, true label, data point..]
    :param n_cluster:
    :return: labels of each data, mask indicating that data'cluster has only 1 member
    """
    kmeans = KMeans(n_cluster).fit(data[:, 2:])
    cluster_sizes = get_size_of_each_cluster(kmeans.labels_)
    return kmeans.labels_, np.array([cluster_sizes[label] == 1 for label in kmeans.labels_])


def RS_cluster(RS, n_cluster):
    """

    :param RS: list of (np.array (ind, true label, data values...))
    :param n_cluster:
    :return: rs = list of data point, cs list of cluster
    """
    if len(RS) < n_cluster:
        return RS, []
    RS = np.array(RS)
    kmeans_labels, is_size_one = get_cluster_and_mask(RS, n_cluster)
    rs = []
    for p in RS[is_size_one]:
        rs.append(p)
    cs_dict = {}
    for i in range(len(RS)):
        if not is_size_one[i]:
            label = kmeans_labels[i]
            if label not in cs_dict:
                cs_dict[label] = [1, RS[i][2:], RS[i][2:] ** 2, 0, 0, {RS[i][0]}]
            else:
                cs_dict[label][0] += 1
                cs_dict[label][1] += RS[i][2:]
                cs_dict[label][2] += RS[i][2:] ** 2
                cs_dict[label][5].add(RS[i][0])

    return rs, cs_dict.values()


def init(data, n_cluster, dimension):
    """

    :param data:
    :param n_cluster:
    :param dimension:
    :return:
    """

    # DS = [[number of vectors, sum of vectors, sum square of vectors, centroid, std, set of members], ...]
    DS = [[0, np.zeros([dimension, ]), np.zeros([dimension, ]), 0, 0, set()] for _ in range(n_cluster)]

    # step 2
    kmeans_labels, is_size_one = get_cluster_and_mask(data, n_cluster * 5)

    # step 3
    RS = []
    for p in data[is_size_one]:
        RS.append(p)

    # step 4
    remaining_points = data[~is_size_one]
    kmeans = KMeans(n_cluster).fit(remaining_points[:, 2:])

    # step 5
    for point, label in zip(remaining_points, kmeans.labels_):
        DS[label][0] += 1
        DS[label][1] += point[2:]
        DS[label][2] += point[2:] ** 2
        DS[label][5].add(point[0])

    # step 6
    CS = []
    if len(RS) >= n_cluster * 5:
        RS, CS = RS_cluster(RS, n_cluster * 5)

    return DS, CS, RS


def update_cluster_info(cluster_set):
    """
    calculate mu (centroid) and std
    :param cluster_set:
    :return:
    """
    for c in cluster_set:
        mu = c[1] / c[0]
        temp = c[2] / c[0] - mu ** 2
        if temp.any() < 0:
            print('something is wrong here')
        std = np.sqrt(c[2] / c[0] - mu ** 2)
        c[3] = mu
        c[4] = std


def Mahalanobis_distance(point, cluster):
    val = (point - cluster[3]) / cluster[4]
    return np.sqrt(np.dot(val, val))


def add2cluster(point, cluster_set, threshold):
    """

    :param point: np.array([ind,true label, data vals...])
    :param cluster_set:
    :param threshold:
    :return:
    """
    min_dis = 10000000
    closest_cluster = None

    for cluster in cluster_set:
        d = Mahalanobis_distance(point[2:], cluster)
        if d < threshold and d < min_dis:
            min_dis = d
            closest_cluster = cluster
    if closest_cluster is not None:
        closest_cluster[0] += 1
        closest_cluster[1] += point[2:]
        closest_cluster[2] += point[2:] ** 2
        closest_cluster[5].add(point[0])
        return True
    return False

    # for i in range(n_cluster):
    #     d = Mahalanobis_distance(point, cluster_set[i])
    #     if d < 2 * cluster_set[i][4]:  # 2 std
    #         min_dis = min(d, min_dis)
    #         cluster_id = i
    # if cluster_id != -1:
    #     cluster_set[cluster_id][0] += 1
    #     cluster_set[cluster_id][1] += point
    #     cluster_set[cluster_id][2] += point ** 2
    #     return True
    # return False


def mergeCS(CS, threshold):
    n = len(CS)
    if n == 0:
        return []
    is_used = [False for _ in range(n)]

    merged_set = []
    for i in range(n - 1):
        is_used[i] = True
        merged_set.append(CS[i])
        for j in range(i + 1, n):
            if not is_used[j]:
                d = Mahalanobis_distance(CS[j][3], CS[i])
                if d < threshold:
                    is_used[j] = True
                    merge_cluster(CS[i], CS[j])
                    # CS[i][0] += CS[j][0]
                    # CS[i][1] += CS[j][1]
                    # CS[i][2] += CS[j][2]
                    # CS[i][5] = CS[i][5].union(CS[j][5])
    if not is_used[-1]:
        merged_set.append(CS[-1])
    update_cluster_info(merged_set)
    return merged_set


def BRF(data, n_cluster, DS, CS, RS, threshold):
    for p in data:
        # step 8
        isAdded = add2cluster(p, DS, threshold)
        if not isAdded:
            # step 9
            isAdded = add2cluster(p, CS, threshold)
            if not isAdded:
                # step 10
                RS.append(p)
                # step 11
                RS, CS2 = RS_cluster(RS, n_cluster * 5)
                update_cluster_info(CS2)
                CS += CS2
                CS = mergeCS(CS, threshold)

        update_cluster_info(DS)
        update_cluster_info(CS)


def to_report(f_out_stream, DS, CS, RS, round_number):
    n_discard = 0
    for cluster in DS:
        n_discard += cluster[0]
    n_compress = 0
    for cluster in CS:
        n_compress += cluster[0]

    f_out_stream.write("Round {}: {},{},{},{}\n".format(round_number, n_discard, len(CS), n_compress, len(RS)))


def merge_cluster(c1, c2):
    c1[1] += c2[1]
    c1[2] += c2[2]
    c1[5] = c1[5].union(c2[5])


def merge_DS_CS(DS, CS, threshold):
    n = len(CS)
    is_merged = [False for _ in range(n)]
    for i in range(len(DS)):
        for j in range(len(CS)):
            if not is_merged[j]:
                d = Mahalanobis_distance(CS[j][3], DS[i])
                if d < threshold:
                    is_merged = True
                    merge_cluster(DS[i], CS[j])
                    # DS[i][0] += CS[j][0]
                    # DS[i][1] += CS[j][1]
                    # DS[i][2] += CS[j][2]
                    # DS[i][5] = DS[i][5].union(CS[j][5])
    cs = []
    for i in range(len(CS)):
        if not is_merged[i]:
            cs.append(CS[i])
    return DS, cs


def get_truth(point_info):
    """

    :param point_info: np.array, row = index, true label
    :return:
    """
    clusters = {}
    for index, label in point_info:
        if label not in clusters:
            clusters[label] = set()
        clusters[label].add(index)
    return clusters


def compare_clusters(cluster1, cluster2):
    is_used2 = {k: False for k in cluster2}
    is_used2[-1] = True

    total_match = 0

    for k1 in cluster1:
        if k1 == -1:
            continue
        max_match = 0
        saved_k2 = None
        for k2 in cluster2:
            if not is_used2[k2]:
                n_match = len(cluster1[k1].intersection(cluster2[k2]))
                if n_match > max_match:
                    max_match = n_match
                    saved_k2 = k2

        if max_match == 0:
            print('somethings wrong')
        is_used2[saved_k2] = True
        total_match += max_match

    n_match_outline = len(cluster1[-1].intersection(cluster2[-1]))
    return total_match + n_match_outline


def get_predict(DS, CS, RS):
    cluster_dict = {}
    for i in range(len(DS)):
        cluster_dict[i] = DS[i][5]
    cluster_dict[-1] = set()
    for p in RS:
        cluster_dict[-1].add(p[0])
    for p in CS:
        cluster_dict[-1] = cluster_dict[-1].union(p[5])

    return cluster_dict


def to_index(predict_cluster_dict):
    l = []
    for label in predict_cluster_dict:
        for idx in predict_cluster_dict[label]:
            l.append((int(idx), label))
    l.sort()
    return l


def main():
    f_in = str(sys.argv[1])
    n_cluster = int(sys.argv[2])
    f_out = str(sys.argv[3])
    #
    # f_in = 'input.txt'
    # n_cluster = 10
    # f_out = 'out.txt'

    f_out_stream = open(f_out, 'w')
    f_out_stream.write("The intermediate results:\n")

    t = time.time()
    input_data = read_csv(f_in)
    print("time ", time.time() - t)

    n = len(input_data)
    dimension = input_data.shape[1] - 2
    threshold = 2 * np.sqrt(dimension)

    p = 0.2
    chunk_sz = int(n * p)
    DS, CS, RS = init(input_data[0:chunk_sz], n_cluster, dimension)

    update_cluster_info(DS)
    update_cluster_info(CS)

    print("time ", time.time() - t)

    round_num = 0
    for s in range(chunk_sz, n - chunk_sz, chunk_sz):
        round_num += 1
        to_report(f_out_stream, DS, CS, RS, round_num)
        BRF(input_data[s:s + chunk_sz], n_cluster, DS, CS, RS, threshold)

    DS, CS = merge_DS_CS(DS, CS, threshold)
    to_report(f_out_stream, DS, CS, RS, round_num + 1)

    predict_cluster_dict = get_predict(DS, CS, RS)

    idx_cluster = to_index(predict_cluster_dict)
    f_out_stream.write("\n")
    f_out_stream.write("The clustering results:\n")
    for r in idx_cluster:
        f_out_stream.write("{},{}\n".format(*r))

    f_out_stream.close()
    print("time ", time.time() - t)

    truth_cluster_dict = get_truth(input_data[:, 0:2])
    total_match = compare_clusters(truth_cluster_dict, predict_cluster_dict)

    print('Match = ', total_match / n * 100)

    print('Aloha')


if __name__ == '__main__':
    main()
