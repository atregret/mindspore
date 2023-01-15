# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train launch."""
import os
import cv2
import time
import datetime
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig, Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from src.optimizers import get_param_groups
from src.losses.crossentropy import CrossEntropy
import numpy as np
from src.lr_scheduler import MultiStepLR, CosineAnnealingLR
from src.utils.logging import get_logger
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id

set_seed(1)


def modelarts_pre_process():
    pass




def predict_image(img_path):       
    img = cv2.imread(img_path)        
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                                 # 选择文件路径
    img = cv2.resize(img, (224, 224))
    print(img.shape)
    img = img.astype(np.float32)
    img = img / 255
    mean = np.array([0.485 , 0.456 , 0.406 ])
    std = np.array([0.229 , 0.224 , 0.225])
    img = (img - mean) / std
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)
    print(img.shape)
    img = img.reshape(1, 3, 224, 224)

    return img


# 定义网络并加载参数，对验证集进行预测
def predict_model(best_ckpt_path,path="dataset/test"):
    from src.network.densenet import DenseNet121 as DenseNet
    #使用的设备
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False)
    net = DenseNet(config.num_classes)#实例化网络
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net,param_dict)
   # loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
    criterion = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)
    model = Model(net, criterion,metrics={"Accuracy":nn.Accuracy()})
    test_name = os.listdir(path)
    test_path = [os.path.join(path, k) for k in test_name]
    test_path.sort(key=lambda x: int(x.split(".")[0].split("_")[1])) 
    cnt = len(test_path)
    print(cnt)
    class_name = {0:"冰淇淋",1:"鸡蛋布丁",2:"烤冷面",3:"芒果班",4:"三明治",5:"松鼠鱼",6:"甜甜圈",7:"土豆泥",8:"小米粥",9:"玉米饼"}
    #class_name = {0:"冰淇淋",1:"jisanbuding",2:"kaolenmian",3:"mangguoban",4:"sanmingzhi",5:"songshuyu",6:"tiantianquan",7:"tudouni",8:"xiaomizou",9:"yumibing"}
    label=[]
    f=open("result.txt",'a')
    for k in test_path:

        data = predict_image(k)
        print(k)
        output = model.predict(Tensor(data))
        pred = np.argmax(output.asnumpy(), axis=1)
        f.write(str(pred[0]))
       
        f.write("\n")
        label.append(pred[0])
    f.close()   
    print(len(label))



if __name__ == "__main__":
    predict_model("./weight/best.ckpt")
