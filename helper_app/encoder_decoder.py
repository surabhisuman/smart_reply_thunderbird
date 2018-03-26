import re
import collections
import tensorflow as tf
import helpers
import numpy as np

tf.reset_default_graph()
sess = tf.Session()

input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units
train_file = "data/train_2.csv"
test_file = "data/test.csv"
train_n = 1000
vocab_size = 1000

train_mails = []
pattern = re.compile("\"(.*?)\",\"(.*?)\"")
with open(train_file) as f:
    next(f)
    for i in range(train_n):
        line = next(f).strip()
        line = re.findall(pattern, line)
        train_mails.append([line[0][0], line[0][1]])

vocab = collections.Counter()
for i in train_mails:
    vocab.update(i[0].split(" "))
    vocab.update(i[1].split(" "))
vocab = [x[0] for x in vocab.most_common(vocab_size)]
word2int = {}
for i in sorted(vocab):
    word2int[i] = len(word2int)
int2word = {}
for i in word2int:
    int2word[word2int[i]] = i


encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)
del encoder_outputs

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder",
)

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
decoder_prediction = tf.argmax(decoder_logits, 2)
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.global_variables_initializer())


def encode_str(s):
    s = s.split(" ")
    ret = []
    for i in s:
        if i in word2int: ret.append(word2int[i])
    return np.array(ret).reshape(-1,1)

def decode_str(v):
    ret = []
    for i in v:
        if i in int2word: ret.append(int2word[i])
    return " ".join(ret)
def next_feed(offset = 0):
    return {
    encoder_inputs: encode_str(train_mails[offset][0]),
    decoder_targets: encode_str(train_mails[offset][1]),
    decoder_inputs: [[0]]
    }

# batch_ = [encode_str(train_mails[x][0]) for x in range(3)]
# batch_, batch_length_ = helpers.batch(batch_)
# print('batch_encoded:\n' + str("\n".join([decode_str(x) for x in batch_])))
#
# din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
#                             max_sequence_length=4)
# print('decoder inputs:\n' + str(din_))
#
# pred_ = sess.run(decoder_prediction,
#     feed_dict={
#         encoder_inputs: batch_,
#         decoder_inputs: din_,
#     })
# pred_ = pred_.swapaxes(0,1)
# for i in pred_:
#     print(decode_str(i))
# print('decoder predictions:\n' + str(pred_))

batch_size = 10
# def batch_generator():
#     while True:
#         yield [encode_str(train_mails[1][0]) for x in range(batch_size)]

batches =  helpers.random_sequences(length_from=4, length_to=8,
                                   vocab_lower=50, vocab_upper=vocab_size,
                                   batch_size=batch_size)
def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [(sequence) for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

loss_track = []
max_batches = train_n
batches_in_epoch = 100

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(decode_str(inp)))
                print('    predicted > {}'.format(decode_str(pred)))
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')
