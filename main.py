import model
import data_create
import argparse
import os
import cv2
import glob
import keras
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    train_height = 64  #HR・LRのサイズ
    train_width = 64
    test_height = 720  #HR・LRのサイズ
    test_width = 1280

    train_dataset_num = 10000 #生成する学習データの数
    test_dataset_num = 10     #生成するテストデータの数
    train_cut_num = 10        #一組の動画から生成するデータの数
    test_cut_num = 1

    train_path = "../../dataset/DIV2K_train_HR"  #画像が入っているパス
    test_path = "../../dataset/DIV2K_valid_HR"

    model_depth = 20  #モデルの層の数

    BATCH_SIZE = 64
    EPOCHS = 80
    
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_model', help='train_datacreate, test_datacreate, train_model, evaluate')

    args = parser.parse_args()

    if args.mode == 'train_datacreate': #学習用データセットの生成
        datacreate = data_create.datacreate()
        train_x, train_y = datacreate.datacreate(train_path,       #切り取る動画のpath
                                            train_dataset_num,     #データセットの生成数
                                            train_cut_num,         #1枚の画像から生成するデータの数
                                            train_height,          #保存サイズ
                                            train_width)   
        path = "train_data_list"
        np.savez(path, train_x, train_y)

    elif args.mode == 'test_datacreate': #評価用データセットの生成
        datacreate = data_create.datacreate()
        test_x, test_y = datacreate.datacreate(test_path,
                                            test_dataset_num,
                                            test_cut_num,
                                            test_height,
                                            test_width)

        path = "test_data_list"
        np.savez(path, test_x, test_y)

    elif args.mode == "train_model": #学習
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")

        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"]
        train_y = npz["arr_1"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.VDSR(model_depth)

        for i in range(EPOCHS // 20):
            optimizers = tf.keras.optimizers.SGD(lr=0.01 * (0.1 ** i), momentum=0.9, decay=1e-4, nesterov=False)
            train_model.compile(loss = "mean_squared_error",
                            optimizer = optimizers,
                            metrics = [psnr])

            train_model.fit(train_x,
                        train_y,
                        epochs = 20,
                        verbose = 2,
                        batch_size = BATCH_SIZE)

        train_model.save("VDSR_model.h5")

    elif args.mode == "evaluate": #評価
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")

        result_path = "result"
        os.makedirs(result_path, exist_ok = True)

        npz = np.load("test_data_list.npz", allow_pickle = True)

        test_x = npz["arr_0"]
        test_y = npz["arr_1"]

        test_x = tf.convert_to_tensor(test_x, np.float32)
        test_y = tf.convert_to_tensor(test_y, np.float32)

        test_x /= 255
        test_y /= 255
            
        path = "VDSR_model.h5"

        if os.path.exists(path):
            model = tf.keras.models.load_model(path, custom_objects={'psnr':psnr})
            pred = model.predict(test_x, batch_size = 1)

            ps_pred_ave = 0

            for p in range(len(test_y)):
                pred[p][pred[p] > 1] = 1
                pred[p][pred[p] < 0] = 0
                ps_pred = psnr(tf.reshape(test_y[p], [test_height, test_width, 1]), pred[p])
                   
                ps_pred_ave += ps_pred

                if True:
                    low_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_x[p] * 255, [test_height, test_width]))
                    cv2.imwrite(result_path + "/" + str(p) + "_low" + ".jpg", low_img) #LR

                    high_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_y[p] * 255, [test_height, test_width]))
                    cv2.imwrite(result_path + "/" + str(p) + "_high" + ".jpg", high_img)   #HR

                    pred_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(pred[p] * 255, [test_height, test_width]))
                    cv2.imwrite(result_path + "/" + str(p) + "_pred" + ".jpg", pred_img) #pred

                    print("num:{}".format(p))
                    print("psnr_pred:{}".format(ps_pred))

            print("psnr_pred_average:{}".format(ps_pred_ave / len(test_y)))


  
 
