import datetime
import os
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.yolo import get_train_model, yolo_body
from nets.yolo_training import get_lr_scheduler, get_yolo_loss
from utils.callbacks import (EvalCallback, ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint, WarmUpCosineDecayScheduler)
from utils.dataloader import YoloDatasets
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    eager           = False
    train_gpu       = [0,]
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'logs/voc_weights.h5'
    input_shape     = [960, 960]
    phi             = 's'

    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 4
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Freeze_Train        = False
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = 'cos'
    save_period         = 10
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 1
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")
        
    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))
    class_names, num_classes = get_classes(classes_path)
    if ngpus_per_node > 1:
        with strategy.scope():

            model_body  = yolo_body([None, None, 3], num_classes = num_classes, phi = phi, weight_decay=weight_decay)
            if model_path != '':
                print('Load weights {}.'.format(model_path))
                model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
            if not eager:
                model       = get_train_model(model_body, input_shape, num_classes)
            else:
                yolo_loss   = get_yolo_loss(input_shape, len(model_body.output), num_classes)
    else:
        model_body  = yolo_body([None, None, 3], num_classes = num_classes, phi = phi, weight_decay=weight_decay)
        if model_path != '':
            print('Load weights {}.'.format(model_path))
            model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
        if not eager:
            model       = get_train_model(model_body, input_shape, num_classes)
        else:
            yolo_loss   = get_yolo_loss(input_shape, len(model_body.output), num_classes)
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    if True:
        if Freeze_Train:
            freeze_layers = {'tiny': 125, 's': 125, 'm': 179, 'l': 234, 'x': 290}[phi]
            for i in range(freeze_layers): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，请扩充数据集。')

        train_dataloader    = YoloDatasets(train_lines, input_shape, batch_size, num_classes, Init_Epoch, UnFreeze_Epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataloader      = YoloDatasets(val_lines, input_shape, batch_size, num_classes, Init_Epoch, UnFreeze_Epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        if eager:
            start_epoch     = Init_Epoch
            end_epoch       = UnFreeze_Epoch
            UnFreeze_flag   = False

            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            
            if ngpus_per_node > 1:
                gen     = strategy.experimental_distribute_dataset(gen)
                gen_val = strategy.experimental_distribute_dataset(gen_val)

            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history    = LossHistory(log_dir)
            eval_callback   = EvalCallback(model_body, input_shape, class_names, num_classes, val_lines, log_dir, \
                                            eval_flag=eval_flag, period=eval_period)

            for epoch in range(start_epoch, end_epoch):

                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size      = Unfreeze_batch_size

                    nbs             = 64
                    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                    lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                    for i in range(len(model_body.layers)): 
                        model_body.layers[i].trainable = True
                    epoch_step      = num_train // batch_size
                    epoch_step_val  = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，请扩充数据集。")

                    train_dataloader.batch_size    = batch_size
                    val_dataloader.batch_size      = batch_size

                    gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
                    gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

                    gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
                    if ngpus_per_node > 1:
                        gen     = strategy.experimental_distribute_dataset(gen)
                        gen_val = strategy.experimental_distribute_dataset(gen_val)
                    
                    UnFreeze_flag = True

                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)

                fit_one_epoch(model_body, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                                            end_epoch, input_shape, num_classes, save_period, save_dir, strategy)
                            
                train_dataloader.on_epoch_end()
                val_dataloader.on_epoch_end()
        else:
            start_epoch = Init_Epoch
            end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch

            if ngpus_per_node > 1:
                with strategy.scope():
                    model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            else:
                model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            logging         = TensorBoard(log_dir)
            loss_history    = LossHistory(log_dir)
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
            early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            eval_callback   = EvalCallback(model_body, input_shape, class_names, num_classes, val_lines, log_dir, \
                                            eval_flag=eval_flag, period=eval_period)
            callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]

            if start_epoch < end_epoch:
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit(
                    x                   = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
            if Freeze_Train:
                batch_size  = Unfreeze_batch_size
                start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                end_epoch   = UnFreeze_Epoch
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
                callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]
                    
                for i in range(len(model_body.layers)): 
                    model_body.layers[i].trainable = True
                if ngpus_per_node > 1:
                    with strategy.scope():
                        model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
                else:
                    model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，请扩充数据集。")

                train_dataloader.batch_size    = Unfreeze_batch_size
                val_dataloader.batch_size      = Unfreeze_batch_size

                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit(
                    x                   = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
