import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
from skimage import io
from skimage.transform import resize

# First, pass the path of the image
dir_path = 'C:/Users/hits/Anaconda3/envs/SlideSeg/SlideSeg-master/SlideSeg-master/output/image_chips'
#image_path=sys.argv[1]
#filename = dir_path +'/' +image_path
image_size=128
num_channels=3
classes = ['benign','malignant']
num_classes = len(classes)
count = 0
for image_path in os.listdir(dir_path):
    filename = os.path.join(dir_path, image_path).replace('\\','/')
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    #print(filename)
    #print(image)
    #print(image.size)
    #image = io.imread(filename)
    if image.size == 0:
	    print('image is empty:'+image_path)
	    continue
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    flag = False
    try:
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    except:
        flag = True

    if flag:
        print('cannot predict for image: '+ image_path)
        continue

    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('./cancer-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    print(result)
    #new code
    #get the index which has maximum value of results
    max = -1
    output_name = 0
    for index, item in enumerate(result[0]):
        if (item >= max):
            max = item
            output_name = index

    output_name = classes[output_name]
    outpath = os.path.join('C:/Users/hits/Anaconda3/envs/SlideSeg/SlideSeg-master/SlideSeg-master/output/classified/'+output_name , image_path).replace('\\','/')
    write = cv2.imwrite(outpath,cv2.imread(filename))
    if write:
        count = count+1
    #print(outpath)
    print(count)
