import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tokyo', type=str)
parser.add_argument('--train_dir', default='default', type=str)
parser.add_argument('--path', default='./data/', type=str)
parser.add_argument('--write_log', default=False, type=bool)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=250, type=int)
parser.add_argument('--hidden_units', default=100, type=int)
parser.add_argument('--time_units', default=100, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1001, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--l2_emb', default=0.0000000, type=float)
parser.add_argument('--debug', default=False, type=bool)


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = data_partition(args)
    [sequences_train, sequences_valid, sequences_test,
     contexts_train, contexts_valid, contexts_test,
     usernum, itemnum, context_dim] = dataset
    num_batch = int(np.ceil(len(sequences_train) / args.batch_size))
    cc = 0.0

    os.makedirs(args.dataset + '_log', exist_ok=True)
    if args.debug:
        f = open(os.path.join(args.dataset + '_log', 'debug_log.txt'), 'w')
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
        f.write('\n')
    else:
        f = open(os.path.join(args.dataset + '_log', 'exp_log.txt'), 'w')
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
        f.write('\n')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    sampler = WarpSampler(sequences_train, contexts_train,
                          usernum, itemnum, context_dim,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=12)

    model = Model(usernum, itemnum, context_dim, args)
    sess.run(tf.initialize_all_variables())

    T = 0.0

    seqs, contexts, pos_contexts, test_items, valid_users = prepare_test_data(dataset=dataset, args=args)
    seqs_2, contexts_2, pos_contexts_2, test_items_2, valid_users_2 = prepare_valid_data(dataset=dataset, args=args)
    sample_seq = seqs[7]
    sample_context = contexts[7]
    sample_pos_context = pos_contexts[7]
    sample_test_item = test_items[7]
    sample_test_user = valid_users[7]
    i = 0
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.separate = False
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, context, pos_context, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: np.reshape(u, [-1, 1]), model.input_seq: seq,
                                     model.input_contexts: context,
                                     model.input_pos_contexts: pos_context,
                                     model.pos: pos, model.neg: neg,
                                     model.is_training: True})
            i = i + 1

        if epoch % 100 == 0:
            t1 = time.time() - t0
            T += t1
            t_test = evaluate(model, sess, seqs, contexts, pos_contexts, test_items,
                              valid_users, args, itemnum)
            t_valid = evaluate(model, sess, seqs_2, contexts_2, pos_contexts_2,
                               test_items_2, valid_users_2, args, itemnum)
            # print ''
            msg = 'epoch:%d, time: %f(s), acc: %f,' \
                  'valid (HR@5: %.4f, NDCG@5 %.4f, HR@10: %.4f, NDCG@10 %.4f, HR@20: %.4f, NDCG@20 %.4f), ' \
                  'test (HR@5: %.4f, NDCG@5 %.4f, HR@10: %.4f, NDCG@10 %.4f, HR@20: %.4f, NDCG@20 %.4f)' % (
                      epoch, T, auc,
                      t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_valid[4], t_valid[5],
                      t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5])
            msg_2 = '%d, %.4f, %.4f' % (epoch, T, t_test[2])
            print(msg)

            f.write(msg)
            f.write("\n")
            f.flush()
            t0 = time.time()
