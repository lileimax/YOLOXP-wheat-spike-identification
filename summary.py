#   该部分代码用于看网络结构
from nets.yolo import yolo_body
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape = [640, 640]
    num_classes = 80
    phi         = 'l'

    model = yolo_body([input_shape[0], input_shape[1], 3], num_classes, phi)

    model.summary()

    net_flops(model, table=False)
    

