import copy
import numpy as np
import pandas as pd
from collections import defaultdict


def data_partition(args):
    print("==========begin partitioning data==========")
    sequences = defaultdict(list)
    contexts = defaultdict(list)

    sequences_train = {}
    sequences_valid = {}
    sequences_test = {}
    contexts_train = {}
    contexts_valid = {}
    contexts_test = {}

    # assume user/item index starting from 1
    path = args.path + args.dataset + '.txt'
    df = pd.read_csv(filepath_or_buffer=path, header=0)
    users = df['user'].unique()
    user_dict = dict(zip(users, np.arange(1, len(users) + 1)))
    df['user'] = df['user'].map(user_dict)
    items = df['item'].unique()
    item_dict = dict(zip(items, np.arange(1, len(items) + 1)))
    df['item'] = df['item'].map(item_dict)

    usernum = max(df['user'])
    itemnum = max(df['item'])
    context_attributes = ['week', 'month', 'day', 'hour']
    context_dim = np.shape(context_attributes)[0]

    grouped = df.groupby(by='user')
    for group in grouped:
        u = int(group[0])
        sequences[u] = group[1]['item'].values
        contexts[u] = group[1][context_attributes].values

    for user in sequences:
        nfeedback = len(sequences[user])
        if nfeedback < 3:
            sequences_train[user] = sequences[user]
            sequences_valid[user] = []
            sequences_test[user] = []
            contexts_train[user] = contexts[user]

        else:
            sequences_train[user] = sequences[user][:-2]
            sequences_valid[user] = [sequences[user][-2]]
            sequences_test[user] = [sequences[user][-1]]

            contexts_train[user] = contexts[user][:-2, :]
            contexts_valid[user] = [contexts[user][-2, :]]
            contexts_test[user] = [contexts[user][-1, :]]

    pad_rate = 0
    for user in sequences_train:
        pad_rate += max(args.maxlen - len(sequences_train[user]), 0)/float(args.maxlen)
    pad_rate /= len(sequences_train)

    print("#user:{}".format(usernum))
    print("#item:{}".format(itemnum))
    print("average padding rate: {}".format(pad_rate))
    print("==========finish partitioning data==========")
    return [sequences_train, sequences_valid, sequences_test,
            contexts_train, contexts_valid, contexts_test,
            usernum, itemnum, context_dim]


def prepare_test_data(dataset, args):
    print("==========begin preparing test dataset==========")
    [sequences_train, sequences_valid, sequences_test,
     contexts_train, contexts_valid, contexts_test,
     usernum, itemnum, context_dim] = copy.deepcopy(dataset)
    seqs = []
    contexts = []
    pos_contexts = []
    test_items = list()
    valid_users = []

    users = range(1, usernum + 1)
    for u in users:
        if len(sequences_train[u]) < 1 or len(sequences_test[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        context = np.zeros([args.maxlen, context_dim], dtype=np.float32)
        pos_context = np.zeros([args.maxlen, context_dim], dtype=np.float32)
        length = min(args.maxlen, len(sequences_train[u]) + 1)
        start_index = args.maxlen - length

        seq[-1] = sequences_valid[u][0]
        context[-1, :] = contexts_valid[u][0]
        pos_context[-1, :] = contexts_test[u][0]
        pos_context[-2, :] = contexts_valid[u][0]
        if length > 1:
            seq[np.arange(start_index, args.maxlen - 1)] = sequences_train[u][np.arange(len(sequences_train[u]) - length + 1, len(sequences_train[u]))]
            context[np.arange(start_index, args.maxlen - 1), :] = contexts_train[u][np.arange(len(sequences_train[u]) - length + 1, len(sequences_train[u]))]
            pos_context[np.arange(start_index, args.maxlen - 2), :] = contexts_train[u][
                np.arange(len(sequences_train[u]) - length + 2, len(sequences_train[u]))]

        seqs.append(seq)
        contexts.append(context)
        pos_contexts.append(pos_context)
        test_items.append(sequences_test[u][0])
        valid_users.append(u)

    print("Number of valid users: {}".format(len(valid_users)))
    print("==========finish preparing test dataset==========")
    return seqs, contexts, pos_contexts, test_items, valid_users


def prepare_valid_data(dataset, args):
    print("==========begin preparing valid dataset==========")
    [sequences_train, sequences_valid, sequences_test,
     contexts_train, contexts_valid, contexts_test,
     usernum, itemnum, context_dim] = copy.deepcopy(dataset)
    seqs = []
    contexts = []
    pos_contexts =[]
    test_items = list()
    valid_users = []

    users = range(1, usernum + 1)
    for u in users:
        if len(sequences_train[u]) < 1 or len(sequences_test[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        context = np.zeros([args.maxlen, context_dim], dtype=np.float32)
        pos_context = np.zeros([args.maxlen, context_dim], dtype=np.float32)
        start_index = args.maxlen - min(args.maxlen, len(sequences_train[u]))
        length = min(args.maxlen, len(sequences_train[u]))

        pos_context[-1, :] = contexts_valid[u][0]
        seq[np.arange(start_index, args.maxlen)] = sequences_train[u][np.arange(len(sequences_train[u]) - length, len(sequences_train[u]))]
        context[np.arange(start_index, args.maxlen), :] = contexts_train[u][np.arange(len(sequences_train[u]) - length, len(sequences_train[u]))]
        pos_context[np.arange(start_index, args.maxlen - 1), :] = contexts_train[u][
            np.arange(len(sequences_train[u]) - length + 1, len(sequences_train[u]))]

        seqs.append(seq)
        contexts.append(context)
        pos_contexts.append(pos_context)
        test_items.append(sequences_valid[u][0])
        valid_users.append(u)

    print("Number of valid users: {}".format(len(valid_users)))
    print("==========finish preparing valid dataset==========")
    return seqs, contexts, pos_contexts, test_items, valid_users


def evaluate(model, sess, seqs, contexts, pos_contexts, test_items, valid_users, args, itemnum):
    total_rank = np.zeros(len(valid_users))
    num_batch = int(np.ceil(float(len(valid_users))/args.test_batch_size))
    for i in range(num_batch):
        index_min = i * args.test_batch_size
        index_max = min(len(valid_users), (i + 1) * args.test_batch_size)
        predictions = model.predict(sess, valid_users[index_min: index_max], seqs[index_min: index_max],
                                                 contexts[index_min: index_max], pos_contexts[index_min: index_max],
                                                 itemnum)

        predictions = - predictions[0]
        predictions = np.reshape(predictions, [-1, itemnum])
        target = predictions[np.arange(len(predictions)), np.array(test_items[index_min: index_max]) - 1].reshape([len(predictions), 1])
        rank = np.sum((predictions < target).astype(int), axis=1)
        total_rank[index_min: index_max] = rank
    HT_5 = sum((total_rank < 5).astype(int))
    NDCG_5 = sum((total_rank < 5).astype(int) / np.log2(total_rank + 2))
    HT_10 = sum((total_rank < 10).astype(int))
    NDCG_10 = sum((total_rank < 10).astype(int) / np.log2(total_rank + 2))
    HT_20 = sum((total_rank < 20).astype(int))
    NDCG_20 = sum((total_rank < 20).astype(int) / np.log2(total_rank + 2))
    return HT_5 / len(valid_users), NDCG_5 / len(valid_users), HT_10 / len(valid_users), NDCG_10 / len(
        valid_users), HT_20 / len(valid_users), NDCG_20 / len(valid_users)




