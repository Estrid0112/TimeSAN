import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def random_neq2(candidates, s):
    t = np.random.randint(0, len(candidates))
    while candidates[t] in s:
        t = np.random.randint(0, len(candidates))
    return candidates[t]


def sample_function(user_train, context_train, usernum, itemnum, context_dim, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        context = np.zeros([maxlen, context_dim], dtype=np.float32)
        pos_context = np.zeros([maxlen, context_dim], dtype=np.float32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        length_o = min(maxlen - 1, len(user_train[user]) - 1)
        length_i = len(user_train[user])

        seq[np.arange(maxlen-length_o, maxlen)] = user_train[user][np.arange(length_i - length_o - 1, length_i - 1)]
        context[np.arange(maxlen-length_o, maxlen), :] = context_train[user][np.arange(length_i - length_o - 1, length_i - 1), :]
        pos_context[np.arange(maxlen - length_o, maxlen - 1), :] = context_train[user][
                                                           np.arange(length_i - length_o, length_i - 1), :]
        pos_context[-1, :] = context_train[user][-1, :]

        pos[np.arange(maxlen-length_o, maxlen)] = user_train[user][np.arange(length_i - length_o, length_i)]

        # negative sampling
        pos_samples = set(user_train[user])
        for i in range(maxlen-length_o, maxlen):
            neg[i] = random_neq(1, itemnum + 1, pos_samples)
        return (user, seq, context, pos_context, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, Sequences, Contexts, usernum, itemnum, context_dim, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(Sequences,
                                                      Contexts,
                                                      usernum,
                                                      itemnum,
                                                      context_dim,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
