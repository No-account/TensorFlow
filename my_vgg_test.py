import tensorflow as tf
import utils
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

var_dict=np.load("./my_vgg.npy")


conv1_1_filter = tf.constant(var_dict['conv1_1_filter'])
conv1_1_biases = tf.constant(var_dict['conv1_1_biases'])
conv1_2_filter = tf.constant(var_dict['conv1_2_filter'])
conv1_2_biases = tf.constant(var_dict['conv1_2_biases'])

conv2_1_filter = tf.constant(var_dict['conv2_1_filter'])
conv2_1_biases = tf.constant(var_dict['conv2_1_biases'])
conv2_2_filter = tf.constant(var_dict['conv2_2_filter'])
conv2_2_biases = tf.constant(var_dict['conv2_2_biases'])

conv3_1_filter = tf.constant(var_dict['conv3_1_filter'])
conv3_1_biases = tf.constant(var_dict['conv3_1_biases'])
conv3_2_filter = tf.constant(var_dict['conv3_2_filter'])
conv3_2_biases = tf.constant(var_dict['conv3_2_biases'])

conv4_1_filter = tf.constant(var_dict['conv4_1_filter'])
conv4_1_biases = tf.constant(var_dict['conv4_1_biases'])
conv4_2_filter = tf.constant(var_dict['conv4_2_filter'])
conv4_2_biases = tf.constant(var_dict['conv4_2_biases'])

conv5_1_filter = tf.constant(var_dict['conv5_1_filter'])
conv5_1_biases = tf.constant(var_dict['conv5_1_biases'])
conv5_2_filter = tf.constant(var_dict['conv5_2_filter'])
conv5_2_biases = tf.constant(var_dict['conv5_2_biases'])

fc_weights1 = tf.constant(var_dict['fc_weights1'])
fc_biases1 = tf.constant(var_dict['fc_biases1'])
fc_weights2 = tf.constant(var_dict['fc_weights2'])
fc_biases2 = tf.constant(var_dict['fc_biases2'])
fc_weights3 = tf.constant(var_dict['fc_weights3'])
fc_biases3 = tf.constant(var_dict['fc_biases3'])

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
    return fc3

image1 = tf.placeholder(tf.float32, [1, 224, 224, 3])
image2 = tf.placeholder(tf.float32, [1, 224, 224, 3])

image_conv_pool1 = conv_pool(image1)
image_conv_pool2 = conv_pool(image2)
image_fc1 = fc_layer(image_conv_pool1)
image_fc2 = fc_layer(image_conv_pool2)

loss = tf.reduce_sum(tf.square(image_fc1 - image_fc2))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

m=np.random.randint(1,9)
m_=0
n_=0
i=1
while(True):
    m_=np.random.randint(1,259)
    n_=np.random.randint(1,259)
    img1=utils.load_image("./test_data/" + str(m) + "/saveImage" + str(m_) + ".jpg")
    img2=utils.load_image("./test_data/" + str(i) + "/saveImage" + str(n_) + ".jpg")
    batch1=img1.reshape(1,224,224,3)
    batch2=img2.reshape(1,224,224,3)
    print(i," ",m,"  ",sess.run(loss,feed_dict={image1:batch1,image2:batch2}))
    i+=1
    if(i==10):
        break

print("\n")
i=1
while(True):
    m_ = np.random.randint(1, 72)
    n_ = np.random.randint(1, 259)
    img1 = utils.load_image("./test_data/" + str(10) + "/saveImage" + str(m_) + ".jpg")
    img2 = utils.load_image("./test_data/" + str(i) + "/saveImage" + str(n_) + ".jpg")
    batch1=img1.reshape(1,224,224,3)
    batch2=img2.reshape(1,224,224,3)
    print(i," ","10","  ",sess.run(loss,feed_dict={image1:batch1,image2:batch2}))
    i+=1
    if(i==10):
        break
