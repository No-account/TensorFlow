import tensorflow as tf
import utils
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

conv1_1_filter = tf.Variable(tf.truncated_normal([3, 3, 3, 64], 0.0, 0.1))
conv1_1_biases = tf.Variable(tf.truncated_normal([64], .0, 0.1))
conv1_2_filter = tf.Variable(tf.truncated_normal([3, 3, 64, 64], 0.0, 0.1))
conv1_2_biases = tf.Variable(tf.truncated_normal([64], .0, 0.1))

conv2_1_filter = tf.Variable(tf.truncated_normal([3, 3, 64, 128], 0.0, 0.1))
conv2_1_biases = tf.Variable(tf.truncated_normal([128], .0, 0.1))
conv2_2_filter = tf.Variable(tf.truncated_normal([3, 3, 128, 128], 0.0, 0.1))
conv2_2_biases = tf.Variable(tf.truncated_normal([128], .0, 0.1))

conv3_1_filter = tf.Variable(tf.truncated_normal([3, 3, 128, 256], 0.0, 0.1))
conv3_1_biases = tf.Variable(tf.truncated_normal([256], .0, 0.1))
conv3_2_filter = tf.Variable(tf.truncated_normal([3, 3, 256, 256], 0.0, 0.1))
conv3_2_biases = tf.Variable(tf.truncated_normal([256], .0, 0.1))

conv4_1_filter = tf.Variable(tf.truncated_normal([3, 3, 256, 512], 0.0, 0.1))
conv4_1_biases = tf.Variable(tf.truncated_normal([512], .0, 0.1))
conv4_2_filter = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.1))
conv4_2_biases = tf.Variable(tf.truncated_normal([512], .0, 0.1))

conv5_1_filter = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.1))
conv5_1_biases = tf.Variable(tf.truncated_normal([512], .0, 0.1))
conv5_2_filter = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.1))
conv5_2_biases = tf.Variable(tf.truncated_normal([512], .0, 0.1))

fc_weights1 = tf.Variable(tf.truncated_normal([25088, 4096], 0.0, 0.1))
fc_biases1 = tf.Variable(tf.truncated_normal([4096], .0, 0.1))
fc_weights2 = tf.Variable(tf.truncated_normal([4096, 1000], 0.0, 0.1))
fc_biases2 = tf.Variable(tf.truncated_normal([1000], .0, 0.1))
fc_weights3 = tf.Variable(tf.truncated_normal([1000, 128], 0.0, 0.1))
fc_biases3 = tf.Variable(tf.truncated_normal([128], .0, 0.1))


def conv_pool(img):
    img = img * 255.0
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=img)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    conv1_1 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(bgr, conv1_1_filter, [1, 1, 1, 1], padding="SAME"), conv1_1_biases))
    conv1_2 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(conv1_1, conv1_2_filter, [1, 1, 1, 1], padding="SAME"), conv1_2_biases))
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2_1 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(pool1, conv2_1_filter, [1, 1, 1, 1], padding="SAME"), conv2_1_biases))
    conv2_2 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(conv2_1, conv2_2_filter, [1, 1, 1, 1], padding="SAME"), conv2_2_biases))
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3_1 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(pool2, conv3_1_filter, [1, 1, 1, 1], padding="SAME"),
                       conv3_1_biases))
    conv3_2 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(conv3_1, conv3_2_filter, [1, 1, 1, 1], padding="SAME"),
                       conv3_2_biases))
    pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv4_1 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(pool3, conv4_1_filter, [1, 1, 1, 1], padding="SAME"),
                       conv4_1_biases))
    conv4_2 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(conv4_1, conv4_2_filter, [1, 1, 1, 1], padding="SAME"),
                       conv4_2_biases))
    pool4 = tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv5_1 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(pool4, conv5_1_filter, [1, 1, 1, 1], padding="SAME"),
                       conv5_1_biases))
    conv5_2 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(conv5_1, conv5_2_filter, [1, 1, 1, 1], padding="SAME"),
                       conv5_2_biases))
    pool5 = tf.nn.max_pool(conv5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return pool5


def fc_layer(img):
    x = tf.reshape(img, [-1, 25088])
    fc1 = tf.nn.bias_add(tf.matmul(x, fc_weights1), fc_biases1)
    relu1 = tf.nn.dropout(tf.nn.relu(fc1), 0.5)
    fc2 = tf.nn.bias_add(tf.matmul(relu1, fc_weights2), fc_biases2)
    relu2 = tf.nn.dropout(tf.nn.relu(fc2), 0.5)
    fc3 = tf.nn.bias_add(tf.matmul(relu2, fc_weights3), fc_biases3)
    relu=tf.nn.relu(fc3)
    return relu


def save_npy(sess, npy_path="./my_vgg.npy"):
    var_dict={'conv1_1_filter':conv1_1_filter,'conv1_1_biases':conv1_1_biases,
              'conv1_2_filter':conv1_2_filter,'conv1_2_biases':conv1_2_biases,
              'conv2_1_filter':conv2_1_filter,'conv2_1_biases':conv2_1_biases,
              'conv2_2_filter':conv2_2_filter,'conv2_2_biases':conv2_2_biases,
              'conv3_1_filter':conv3_1_filter,'conv3_1_biases':conv3_1_biases,
              'conv3_2_filter':conv3_2_filter,'conv3_2_biases':conv3_2_biases,
              'conv4_1_filter':conv4_1_filter,'conv4_1_biases':conv4_1_biases,
              'conv4_2_filter':conv4_2_filter,'conv4_2_biases':conv4_2_biases,
              'conv5_1_filter':conv5_1_filter,'conv5_1_biases':conv5_1_biases,
              'conv5_2_filter':conv5_2_filter,'conv5_2_biases':conv5_2_biases,
              'fc_weights1': fc_weights1, 'fc_biases1': fc_biases1,
              'fc_weights2': fc_weights2, 'fc_biases2': fc_biases2,
              'fc_weights3': fc_weights3, 'fc_biases3': fc_biases3,}
    for (name), var in list(var_dict.items()):
        var_out = sess.run(var)
        if name not in var_dict:
            var_dict[name] = {}
        var_dict[name] = var_out

    np.save(npy_path, var_dict)
    print("saved",npy_path)
    return npy_path


image1 = tf.placeholder(tf.float32, [1, 224, 224, 3])
image2 = tf.placeholder(tf.float32, [1, 224, 224, 3])
image3 = tf.placeholder(tf.float32, [1, 224, 224, 3])

image_conv_pool1 = conv_pool(image1)
image_conv_pool2 = conv_pool(image2)
image_conv_pool3 = conv_pool(image3)
image_fc1 = fc_layer(image_conv_pool1)
image_fc2 = fc_layer(image_conv_pool2)
image_fc3 = fc_layer(image_conv_pool3)

loss = tf.reduce_sum(tf.square(image_fc1 - image_fc2)) / tf.reduce_sum(tf.square(image_fc1 - image_fc3))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for j in range(100):
    m = 0
    n = 0
    while (True):
        m = np.random.randint(1, 9)
        n = np.random.randint(1, 9)
        if (m != n):
            break
    for i in range(50):
        m_ = 1
        n_ = 1
        k_ = 1
        while (True):
            m_ = np.random.randint(1, 259)
            n_ = np.random.randint(1, 259)
            K_ = np.random.randint(1, 259)
            if (m_ != n_):
                break
        img1 = utils.load_image("./test_data/" + str(m) + "/saveImage" + str(m_) + ".jpg")
        img2 = utils.load_image("./test_data/" + str(m) + "/saveImage" + str(n_) + ".jpg")
        img3 = utils.load_image("./test_data/" + str(n) + "/saveImage" + str(k_) + ".jpg")
        batch1 = img1.reshape((1, 224, 224, 3))
        batch2 = img2.reshape((1, 224, 224, 3))
        batch3 = img3.reshape((1, 224, 224, 3))
        print(j,"  ",i,"   ",sess.run([loss, train_step], feed_dict={image1: batch1, image2: batch2, image3: batch3}))
save_npy(sess)
sess.close()
