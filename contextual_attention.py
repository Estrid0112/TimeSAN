from __future__ import print_function
import tensorflow as tf
import numpy as np


def contextual_attention(queries,
                         keys,
                         values,
                         temp_contexts_queries,
                         temp_contexts_keys,
                         num_time_units,
                         num_units=None,
                         num_heads=8,
                         dropout_rate=0,
                         is_training=True,
                         causality=False,
                         scope="contextual_attention",
                         reuse=None,
                         with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.layers.dense(values, num_units, activation=None)  # (N, T_k, C)
        T_C_Q = tf.layers.dense(temp_contexts_queries, num_time_units, activation=None, name='time')
        T_C_K = tf.layers.dense(temp_contexts_keys, num_time_units, activation=None, name='time_key')

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        T_C_Q_ = tf.concat(tf.split(T_C_Q, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        T_C_K_ = tf.concat(tf.split(T_C_K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)


        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs_temp = tf.nn.softsign(tf.matmul(T_C_Q_, tf.transpose(T_C_K_, [0, 2, 1])))

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        outputs_temp = outputs_temp / (K_.get_shape().as_list()[-1] ** 0.5)
        att_weights = outputs
        temp_weights = outputs_temp
        outputs *= outputs_temp


        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
        if causality:
            indices = tf.convert_to_tensor(np.array(list(zip(np.arange(250 - 1), np.arange(1, 250)))))
            tril = tf.sparse_to_dense(indices, output_shape=[250, 250], default_value=1., sparse_values=0.)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)
        joint_weights = outputs

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries


        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs, joint_weights, att_weights, temp_weights


def gated_sum(context, seq, num_units):
    concatenated = tf.concat([seq, context], axis=-1)
    gate = tf.layers.dense(concatenated, num_units, activation=tf.sigmoid)
    return gate * seq + (1 - gate) * context
