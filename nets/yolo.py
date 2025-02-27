from tensorflow.keras.layers import (Concatenate, Input, Lambda, UpSampling2D,
                          ZeroPadding2D)
from tensorflow.keras.models import Model

from nets.CSPdarknet import (CSPLayer, DarknetConv2D, DarknetConv2D_BN_SiLU,cbam_block,eca_block,
                               darknet_body)
from nets.yolo_training import get_yolo_loss



def yolo_body(input_shape, num_classes, phi, weight_decay=5e-4):
    depth_dict      = {'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict      = {'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    depth, width    = depth_dict[phi], width_dict[phi]
    in_channels     = [256, 512, 1024]
    
    inputs      = Input(input_shape)

    feat1, feat2, feat3 = darknet_body(inputs, depth, width, weight_decay=weight_decay)


    P5          = DarknetConv2D_BN_SiLU(int(in_channels[1] * width), (1, 1), weight_decay=weight_decay, name = 'backbone.lateral_conv0')(feat3)  
    P5_upsample = UpSampling2D()(P5)
    P5_upsample = cbam_block(P5_upsample, name='P5_upsample')
    P5_upsample = Concatenate(axis = -1)([P5_upsample, feat2])
    P5_upsample = CSPLayer(P5_upsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_p4')


    P4          = DarknetConv2D_BN_SiLU(int(in_channels[0] * width), (1, 1), weight_decay=weight_decay, name = 'backbone.reduce_conv1')(P5_upsample)
    P4_upsample = UpSampling2D()(P4)
    P4_upsample = cbam_block(P4_upsample, name='P4_upsample')
    P4_upsample = Concatenate(axis = -1)([P4_upsample, feat1])
    P3_out      = CSPLayer(P4_upsample, int(in_channels[0] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_p3')

    P3_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample   = DarknetConv2D_BN_SiLU(int(in_channels[0] * width), (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.bu_conv2')(P3_downsample)
    P3_downsample   = Concatenate(axis = -1)([P3_downsample, P4])
    P4_out          = CSPLayer(P3_downsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_n3')

    P4_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    P4_downsample   = DarknetConv2D_BN_SiLU(int(in_channels[1] * width), (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.bu_conv1')(P4_downsample)
    P4_downsample   = Concatenate(axis = -1)([P4_downsample, P5])
    P5_out          = CSPLayer(P4_downsample, int(in_channels[2] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_n4')

    fpn_outs    = [P3_out, P4_out, P5_out]
    yolo_outs   = []
    for i, out in enumerate(fpn_outs):

        stem    = DarknetConv2D_BN_SiLU(int(256 * width), (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.stems.' + str(i))(out)

        cls_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_convs.' + str(i) + '.0')(stem)
        cls_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_convs.' + str(i) + '.1')(cls_conv)

        cls_pred = DarknetConv2D(num_classes, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_preds.' + str(i))(cls_conv)

        reg_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_convs.' + str(i) + '.0')(stem)
        reg_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_convs.' + str(i) + '.1')(reg_conv)

        reg_pred = DarknetConv2D(4, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_preds.' + str(i))(reg_conv)
        #---------------------------------------------------#

        obj_pred = DarknetConv2D(1, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.obj_preds.' + str(i))(reg_conv)
        output   = Concatenate(axis = -1)([reg_pred, obj_pred, cls_pred])
        yolo_outs.append(output)
    return Model(inputs, yolo_outs)

def get_train_model(model_body, input_shape, num_classes):
    y_true = [Input(shape = (None, 5))]
    model_loss  = Lambda(
        get_yolo_loss(input_shape, len(model_body.output), num_classes), 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
    )([*model_body.output, *y_true])
    
    model       = Model([model_body.input, *y_true], model_loss)
    return model
