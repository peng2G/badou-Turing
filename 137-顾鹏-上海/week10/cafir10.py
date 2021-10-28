import warnings
warnings.simplefilter("ignore")
import math
from datetime import datetime
import time
import os
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

# 定义全局变量
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'Cifar_data/logs/',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# 基本模型参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'Cifar_data/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

# 定义记录训练步数的变量
global_step = tf.train.get_or_create_global_step()  # tf.Variable(0, trainable=False)
# 从 CIFAR-10 中导入数据和标签
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# 描述模型的训练
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

class CifarData():
    def __init__(self,data_dir,batch_size,image_size = IMAGE_SIZE):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size =image_size

    class Cifar10Record(object):
        # 结果数据对象
        pass

    def get_data(self,distorted=True):
        width = 32
        heigh = 32
        channel = 3
        label_bytes = 1
        sample_obj=self._read_cifar10(width,heigh, channel,label_bytes) # 读取图像
        distorted_image = self._distorted_inputs(sample_obj.image_uint8,distorted)    # 图像增强
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_fraction_of_examples_in_queue)
        images,labels=self.generate_image_and_label_batch(  #图像打包batch
            distorted_image,
            sample_obj.label,
            min_queue_examples,
            self.batch_size
        )
        return images,labels


    def _distorted_inputs(self,image_obj,distorted):
        height = self.image_size
        width = self.image_size
        float_image = tf.cast(image_obj,tf.float32)
        distorted_image = tf.random_crop(float_image, [height, width, 3])
        if distorted:
            # 随机水平翻转图像
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # 由于这些操作是不可交换的，因此可以考虑随机化和调整操作的顺序
            # 在某范围随机调整图片亮度
            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=63)
            # 在某范围随机调整图片对比度
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.2, upper=1.8)

        # 减去平均值并除以像素的方差，白化操作：均值变为0，方差变为1
        distorted_image = tf.image.per_image_standardization(distorted_image)
        distorted_image.set_shape([height,width,3])
        return distorted_image


    def _read_cifar10(self,width = 32,heigh = 32, channel= 3,label_bytes =1):
        image_bytes = width * heigh * channel
        record_bytes = image_bytes + label_bytes

        data_obj = self.Cifar10Record()
        filenames = [os.path.join(self.data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
        for f in filenames:
            if not gfile.Exists(f):
                raise ValueError("Failed to find file:" + f)

        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

        data_obj.key, value = reader.read(filename_queue)

        # change string => uint8
        value_uint8 = tf.decode_raw(value, tf.uint8)

        # split label image
        data_obj.label = tf.cast(tf.slice(value_uint8, [0], [label_bytes]), tf.int32)


        image_stream = tf.reshape(tf.slice(value_uint8, [label_bytes], [image_bytes]), [channel, heigh, width])
        # [c,h,w]=>[h,w,c]
        data_obj.image_uint8 = tf.transpose(image_stream, [1, 2, 0])
        return data_obj


    def generate_image_and_label_batch(self,image,label,min_queue_examples,num_preprocess_threads=8):
        print('1')
        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)
        images,labels =tf.train.shuffle_batch(
            [image,label],
            batch_size=self.batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples+3*self.batch_size,
            min_after_dequeue=min_queue_examples,
        )
        return images, tf.reshape(labels, [self.batch_size])


def inference(images):
    # 定义推断网络
    with tf.variable_scope('conv1') as scope:  # 每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀
        # 5*5 的卷积核，64个
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
        # 卷积操作，步长为1，0padding SAME，不改变宽高，通道数变为64
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        # 在CPU上创建第一层卷积操作的偏置变量
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        # 加上偏置
        bias = tf.nn.bias_add(conv, biases)
        # relu非线性激活
        conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1-第一层pooling
    # 3*3 最大池化，步长为2
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
    # norm1-局部响应归一化
    # LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

    # conv2-第二层卷积
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3-全连接层，384个节点
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list(): #flatten操作
            dim *= d
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
        weights = _variable_with_weight_decay(
            'weights', shape=[dim, 384],
            stddev=0.04, wd=0.004)
        biases = tf.get_variable('biases', [384],initializer= tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)         # relu激活


    # local4-全连接层，192个节点
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[384, 192],
            stddev=0.04, wd=0.004)
        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    # softmax, i.e. softmax(WX + b)
    # 输出层
    with tf.variable_scope('softmax_linear') as scope:

        weights = _variable_with_weight_decay(
            'weights', [192, NUM_CLASSES],
            stddev=1 / 192.0, wd=0.0)        # 权重
        biases = tf.get_variable('biases', [NUM_CLASSES],initializer=tf.constant_initializer(0.0))
        # 输出层的线性操作
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    return softmax_linear


def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    创建带正则化的权重
    Args:
    name: 变量的名称
    shape: 整数列表
    stddev: 截断高斯的标准差
    wd: 加L2Loss权重衰减乘以这个浮点数.如果没有，此变量不会添加权重衰减.

    Returns:
    变量张量
    '''
    var = tf.get_variable(
        name,
        shape,
        initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd, name='weight_loss'
        )
        tf.add_to_collection('losses', weight_decay)
    return var

# 描述损失函数，往inference图中添加生成损失（loss）所需要的操作（ops）
def loss(logits, labels):
    '''
    ARGS：
    logits：来自inference（）的Logits
    labels：来自distorted_inputs或输入（）的标签.一维张量形状[batch_size]
    返回： float类型的损失张量
    '''
    labels = tf.cast(labels, tf.int64)
    # 计算这个batch的平均交叉熵损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # 总损失定义为交叉熵损失加上所有的权重衰减项（L2损失）
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step):
    '''
    ARGS：
     total_loss：loss()的全部损失
     global_step：记录训练步数的整数变量
    返回：
     train_op：训练的op
    '''
    # 影响学习率的变量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # 根据步骤数以指数方式衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    train_op=tf.train.AdamOptimizer(lr).minimize(total_loss)
    # 跟踪所有可训练变量的移动平均值
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([train_op]):  #执行完train_op后才能执行variables_averages_op
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op



# 训练之前的初始化工作，删除之前训练过程中产生的一些临时文件，并重新生成目录
if gfile.Exists(FLAGS.train_dir):
    gfile.DeleteRecursively(FLAGS.train_dir)
gfile.MakeDirs(FLAGS.train_dir)

# 获取数据op
cifar_data= CifarData(FLAGS.data_dir,FLAGS.batch_size,image_size = IMAGE_SIZE)      # 实例化数据集对象
images_train, labels_train = cifar_data.get_data()  # 训练数据op
# images_test, labels_test = cifar_data.get_data(distorted=False)     # 测试数据op
# 数据占位
x=tf.placeholder(tf.float32,[FLAGS.batch_size,24,24,3])
y_=tf.placeholder(tf.int32,[FLAGS.batch_size])
# 组合训练，测试op
logits=inference(x)  #推理op
loss = loss(logits, y_)  #损失op
train_op = train(loss, global_step)     # 训练op
top_k_op = tf.nn.in_top_k(logits,y_,1)  #统计推断正确个数
saver = tf.train.Saver(tf.all_variables())  # 用tf.train.Saver()创建一个Saver来管理模型中的所有变量
###### 训练测试Session()
# 初始化所有的变量.
init = tf.initialize_all_variables()
# 运行计算图中的所有操作.

with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as sess:
    sess.run(init)
    # 调用run或者eval去执行read之前，必须调用tf.train.start_queue_runners来将文件名填充到队列.否则read操作会被阻塞到文件名队列中有值为止
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess,coord=coord)
    for step in range(FLAGS.max_steps):
        # 记录运行计算图一次的时间
        start_time = time.time()

        images_batch,labels_batch = sess.run([images_train,labels_train])
        _, loss_value = sess.run([train_op, loss],feed_dict={x:images_batch,y_:labels_batch})
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        if step % 100 == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, sec_per_batch))
        # 定期保存模型检查点
        if step % 10000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


'''
file_path =r'Cifar_data/logs/model.ckpt-%s.meta'%str(global_step-1)
saver = tf.train.import_meta_graph(file_path)
with tf.Session() as sess:
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    saver.restore(sess,tf.train.latest_checkpoint(r'Cifar_data/logs/'))
    num_batch = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size))
    total_sample_count = num_batch * FLAGS.batch_size
    true_count = 0
    print(num_batch)

    for j in range(num_batch):
        image_batch_t, label_batch_t = sess.run([images_test, labels_test])
        predicts = sess.run([top_k_op], feed_dict={x: image_batch_t, y_: label_batch_t})
        true_count += np.sum(predicts)
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
'''