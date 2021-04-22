from modules import *
from contextual_attention import *


def time_encoding(input, dim, args, dtype=tf.float32):
    input = tf.expand_dims(input, axis=-1)
    sin_factor = tf.convert_to_tensor(np.array([[1 / np.power(j, i / dim) for i in range(0, dim, 2)] for j in [7, 12, 31, 24]]), dtype=dtype)
    sin_vec = tf.sin(input * sin_factor)
    cos_factor = tf.convert_to_tensor(np.array([[1 / np.power(j, i / dim) for i in range(0, dim, 2)] for j in [7, 12, 31, 24]]), dtype=dtype)
    cos_vec = tf.cos(input * cos_factor)
    sin_vec = tf.reshape(sin_vec, [tf.shape(input)[0], args.maxlen, 4, -1, 1])
    cos_vec = tf.reshape(cos_vec, [tf.shape(input)[0], args.maxlen, 4, -1, 1])
    combined = tf.concat([sin_vec, cos_vec], axis=-1)
    output = tf.reshape(combined, [tf.shape(input)[0], args.maxlen, 4 * dim])

    return output


class Model():
    def __init__(self, usernum, itemnum, context_dim, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=None)
        self.u = tf.placeholder(tf.int32, shape=[None, 1])
        self.input_seq = tf.placeholder(tf.int32, shape=[None, args.maxlen])
        self.input_contexts = tf.placeholder(tf.float32, shape=[None, args.maxlen, context_dim])
        self.input_pos_contexts = tf.placeholder(tf.float32, shape=[None, args.maxlen, context_dim])

        self.pos = tf.placeholder(tf.int32, shape=[None, args.maxlen])
        self.neg = tf.placeholder(tf.int32, shape=[None, args.maxlen])
        pos = self.pos
        neg = self.neg
        input_time = tf.slice(self.input_contexts, [0, 0, 0],
                              [tf.shape(self.input_contexts)[0], args.maxlen, 4])

        input_pos_time = tf.slice(self.input_pos_contexts, [0, 0, 0],
                                  [tf.shape(self.input_pos_contexts)[0], args.maxlen, 4])

        input_time = time_encoding(input=input_time, dim=10, args=args)
        input_pos_time = time_encoding(input=input_pos_time, dim=10, args=args)

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("user/embedding"):
            self.user = embedding(self.u,
                                  vocab_size=usernum + 1,
                                  num_units=args.hidden_units,
                                  zero_pad=True,
                                  scale=True,
                                  l2_reg=args.l2_emb,
                                  scope="user_embeddings",
                                  with_t=False,
                                  reuse=reuse
                                  )

        with tf.variable_scope("sequence"):
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # # Positional Encoding
            # t, pos_emb_table = embedding(
            #     tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
            #     vocab_size=args.maxlen,
            #     num_units=args.hidden_units,
            #     zero_pad=False,
            #     scale=False,
            #     l2_reg=args.l2_emb,
            #     scope="dec_pos",
            #     reuse=reuse,
            #     with_t=True
            # )
            # self.seq += t
        with tf.variable_scope("sequence/dropout"):
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.user = tf.layers.dropout(self.user,
                                          rate=args.dropout_rate,
                                          training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

        # Parallel block
        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                with tf.variable_scope("self_attention"):
                    self.seq, _, _, _ = contextual_attention(queries=normalize(self.seq),
                                                             keys=self.seq,
                                                             values=self.seq,
                                                             temp_contexts_queries=input_pos_time,
                                                             temp_contexts_keys=input_time,
                                                             num_units=args.hidden_units,
                                                             num_time_units=args.time_units,
                                                             num_heads=args.num_heads,
                                                             dropout_rate=args.dropout_rate,
                                                             is_training=self.is_training,
                                                             causality=True,
                                                             scope="contextual_attention")

                    with tf.variable_scope("forward"):
                        self.seq = feedforward(normalize(self.seq),
                                               num_units=[args.hidden_units, args.hidden_units],
                                               dropout_rate=args.dropout_rate, is_training=self.is_training)

        self.seq = normalize(self.seq)
        seq_ = tf.layers.dense(self.seq, units=args.hidden_units, use_bias=False)
        user_ = tf.layers.dense(self.user, units=args.hidden_units, use_bias=False)
        fusion_bias = tf.get_variable(name='o_bias', shape=[args.hidden_units], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.))
        fusion = tf.sigmoid(seq_ + user_ + fusion_bias)
        self.seq = fusion * self.seq + (1 - fusion) * self.user

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)

        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=[1, itemnum])
        test_item_emb = tf.reduce_sum(tf.nn.embedding_lookup(item_emb_table, self.test_item), axis=0)

        seq_emb_last = tf.reshape(self.seq[:, -1, :], [-1, args.hidden_units])
        self.test_logits = tf.matmul(seq_emb_last, tf.transpose(test_item_emb))

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.seq_loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)

        self.loss = self.seq_loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)

            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, context, pos_context,  itemnum):
        return sess.run([self.test_logits],
                        {self.u: np.reshape(u, [-1, 1]), self.input_seq: seq, self.input_contexts: context,
                         self.input_pos_contexts: pos_context,
                         self.test_item: np.arange(1, itemnum + 1).reshape([1, itemnum]),
                         self.is_training: False})



