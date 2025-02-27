import tensorflow as tf
from tensorflow.keras import backend as K


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:

        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def DecodeBox(outputs,
            num_classes,
            input_shape,
            max_boxes       = 200,
            confidence      = 0.2,
            nms_iou         = 0.3,
            letterbox_image = True):
            
    image_shape = K.reshape(outputs[-1],[-1])
    outputs     = outputs[:-1]

    bs      = K.shape(outputs[0])[0]

    grids   = []
    strides = []

    hw      = [K.shape(x)[1:3] for x in outputs]

    outputs = tf.concat([tf.reshape(x, [bs, -1, 5 + num_classes]) for x in outputs], axis = 1)
    for i in range(len(hw)):

        grid_x, grid_y  = tf.meshgrid(tf.range(hw[i][1]), tf.range(hw[i][0]))
        grid            = tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2))
        shape           = tf.shape(grid)[:2]

        grids.append(tf.cast(grid, K.dtype(outputs)))
        strides.append(tf.ones((shape[0], shape[1], 1)) * input_shape[0] / tf.cast(hw[i][0], K.dtype(outputs)))

    grids               = tf.concat(grids, axis=1)
    strides             = tf.concat(strides, axis=1)

    box_xy = (outputs[..., :2] + grids) * strides / K.cast(input_shape[::-1], K.dtype(outputs))
    box_wh = tf.exp(outputs[..., 2:4]) * strides / K.cast(input_shape[::-1], K.dtype(outputs))

    box_confidence  = K.sigmoid(outputs[..., 4:5])
    box_class_probs = K.sigmoid(outputs[..., 5: ])

    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    box_scores  = box_confidence * box_class_probs
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):

        class_boxes      = tf.boolean_mask(boxes, mask[..., c])
        class_box_scores = tf.boolean_mask(box_scores[..., c], mask[..., c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def decode_for_vision(output):

        bs, hw = np.shape(output)[0], np.shape(output)[1:3]
        # batch_size, 400, 4 + 1 + num_classes
        output          = np.reshape(output, [bs, hw[0] * hw[1], -1])

        grid_x, grid_y  = np.meshgrid(np.arange(hw[1]), np.arange(hw[0]))

        grid            = np.reshape(np.stack((grid_x, grid_y), 2), (1, -1, 2))

        box_xy  = (output[..., :2] + grid)
        box_wh  = np.exp(output[..., 2:4])

        fig = plt.figure()
        ax  = fig.add_subplot(121)
        plt.ylim(-2,22)
        plt.xlim(-2,22)
        plt.scatter(grid_x,grid_y)
        plt.scatter(0,0,c='black')
        plt.scatter(1,0,c='black')
        plt.scatter(2,0,c='black')
        plt.gca().invert_yaxis()

        ax  = fig.add_subplot(122)
        plt.ylim(-2,22)
        plt.xlim(-2,22)
        plt.scatter(grid_x,grid_y)
        plt.scatter(0,0,c='black')
        plt.scatter(1,0,c='black')
        plt.scatter(2,0,c='black')

        plt.scatter(box_xy[0,0,0], box_xy[0,0,1],c='r')
        plt.scatter(box_xy[0,1,0], box_xy[0,1,1],c='r')
        plt.scatter(box_xy[0,2,0], box_xy[0,2,1],c='r')
        plt.gca().invert_yaxis()

        pre_left    = box_xy[...,0] - box_wh[...,0]/2 
        pre_top     = box_xy[...,1] - box_wh[...,1]/2 

        rect1   = plt.Rectangle([pre_left[0,0],pre_top[0,0]],box_wh[0,0,0],box_wh[0,0,1],color="r",fill=False)
        rect2   = plt.Rectangle([pre_left[0,1],pre_top[0,1]],box_wh[0,1,0],box_wh[0,1,1],color="r",fill=False)
        rect3   = plt.Rectangle([pre_left[0,2],pre_top[0,2]],box_wh[0,2,0],box_wh[0,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()

    feat = np.concatenate([np.random.uniform(-1, 1, [4, 20, 20, 2]), np.random.uniform(1, 3, [4, 20, 20, 2]), np.random.uniform(1, 3, [4, 20, 20, 81])], -1)
    decode_for_vision(feat)
