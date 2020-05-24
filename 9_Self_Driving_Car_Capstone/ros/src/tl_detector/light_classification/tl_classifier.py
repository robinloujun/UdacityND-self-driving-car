from styx_msgs.msg import TrafficLight
import tensorflow as tf
import rospy
import cv2
import numpy as np
import os


def filter_boxes(min_score, boxes, scores, classes):
    """
    Return boxes with a confidence >= `min_score`
    """
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes


class TLClassifier(object):
    def __init__(self):
        # load classifier
        self.light = TrafficLight.UNKNOWN
        
        path = os.getcwd()
        self.graph_file = path + '/light_classification/frozen_inference_graph.pb'
        rospy.loginfo("the current path is %s", path)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.category_index = {1: {'id': 1, 'name': 'green'}, 2: {'id': 2, 'name': 'red'},
                               3: {'id': 3, 'name': 'yellow'}, 4: {'id': 4, 'name': 'none'}}

        # create tensorflow session for detection
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args: image (cv::Mat): image containing the traffic light

        Returns: int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        t_start = rospy.get_time()

        # light color prediction
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply the detection.
        with self.detection_graph.as_default():
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            (boxes, scores, classes) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes],
                feed_dict={self.image_tensor: np.expand_dims(image_rgb, 0)})

        # remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        confidence_cutoff = 0.8
        count_red = 0

        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)
        count_sure = len(classes)

        t_end = rospy.get_time()

        delta_t = t_end - t_start

        rospy.loginfo('The classification takes {}s'.format(delta_t)) # average 1.3s

        if count_sure:
            for i in range(len(classes)):
                class_name = self.category_index[classes[i]]['name']

                if class_name == 'red':
                    count_red += 1
        else:
            self.light = TrafficLight.UNKNOWN

        if count_red:
            self.light = TrafficLight.RED
        else:
            self.light = TrafficLight.GREEN

        return self.light
