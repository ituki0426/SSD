import torch.nn as nn


def make_vgg():
    layers = []
    in_channels = 3
    # vggに配置する畳み込み層のフィルター数
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]
    # vgg1~vgg5の畳み込み層までを生成
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,  # ウィンドウサイズ 2x2
                                    stride=2)]  # ストライドサイズ 2
        elif v == 'MC':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # vgg5のプーリング層を実装
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # vgg6の畳み込み層1を実装
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # vgg7の畳み込層2を実装
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5,
        conv6, nn.ReLU(inplace=True), #畳み込み層の活性化はReLU
        conv7, nn.ReLU(inplace=True)] #畳み込み層の活性化はReLU
    # リストlayersをnn.ModuleListに格納してReturnする
    return nn.ModuleList(layers)
