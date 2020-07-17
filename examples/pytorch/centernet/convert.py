import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
from torch import nn
# from .utils import load_state_dict_from_url

# model_urls = {
#     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
# }

# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     :param v:
#     :param divisor:
#     :param min_value:
#     :return:
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v

# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
#         padding = (kernel_size - 1) // 2
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
#             norm_layer(out_planes),
#             nn.ReLU6(inplace=True)
#         )

# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup

#         layers = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             norm_layer(oup),
#         ])
#         self.conv = nn.Sequential(*layers)

#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)

# class SeparableConv2d(nn.Module):
#     # borrow from https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
#     def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
#         super(SeparableConv2d,self).__init__()

#         self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
#         self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
#     def forward(self,x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

# class MobileNetV2(nn.Module):
#     def __init__(self,
#                  width_mult=1.0,
#                  inverted_residual_setting=None,
#                  round_nearest=8,
#                  block=None,
#                  norm_layer=None):
#         """
#         MobileNet V2 main class
#         Args:
#             width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
#             inverted_residual_setting: Network structure
#             round_nearest (int): Round the number of channels in each layer to be a multiple of this number
#             Set to 1 to turn off rounding
#             block: Module specifying inverted residual building block for mobilenet
#             norm_layer: Module specifying the normalization layer to use
#         """
#         super(MobileNetV2, self).__init__()

#         if block is None:
#             block = InvertedResidual

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         input_channel = 32
#         last_channel = 1280

#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 # t, c, n, s
#                 [1, 16, 1, 1],
#                 [6, 24, 2, 2],
#                 [6, 32, 3, 2],
#                 [6, 64, 4, 2],
#                 [6, 96, 3, 1],
#                 [6, 160, 3, 2],
#                 [6, 320, 1, 1],
#             ]

#         # only check the first element, assuming user knows t,c,n,s are required
#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element list, got {}".format(inverted_residual_setting))

#         # building first layer
#         input_channel = _make_divisible(input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
#         features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
#         # building inverted residual blocks
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
#                 input_channel = output_channel
#         # building last several layers
#         print('self.last_channel', self.last_channel)
#         features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*features)

#         # # building classifier
#         # self.classifier = nn.Sequential(
#         #     nn.Dropout(0.2),
#         #     nn.Linear(self.last_channel, 1000),
#         # )
#         self.deconv_layers = self._make_deconv_layer(
#             3,
#             [256, 256, 256],
#             [4, 4, 4],
#         )
#         # weight initialization
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
#         #         if m.bias is not None:
#         #             nn.init.zeros_(m.bias)
#         #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#         #         nn.init.ones_(m.weight)
#         #         nn.init.zeros_(m.bias)
#         #     elif isinstance(m, nn.Linear):
#         #         nn.init.normal_(m.weight, 0, 0.01)
#         #         nn.init.zeros_(m.bias)


#     def _forward_impl(self, x):
#         # This exists since TorchScript doesn't support inheritance, so the superclass method
#         # (this one) needs to have a name other than `forward` that can be accessed in a subclass
#         x = self.features(x)

#         x = self.deconv_layers(x)
#         # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
#         # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
#         # x = self.classifier(x)
#         return x

#     def forward(self, x):
#         return self._forward_impl(x)

#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'

#         inplanes = 1280
#         layers = []
#         for i in range(num_layers):
#             # kernel, padding, output_padding = \
#             #     self._get_deconv_cfg(num_kernels[i], i)
#             # print('kernel, padding, output_padding', kernel, padding, output_padding) # 4 1 0
#             planes = num_filters[i]

#             # layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
#             # # litehead
#             # if i == 0:
#             #     layers.append(SeparableConv2d(inplanes, planes, kernel_size=3, padding=1, bias=True))
#             # else:
#             #     assert inplanes == planes
#             #     layers.append(nn.Conv2d(inplanes
#             #                             , planes
#             #                             , kernel_size=3
#             #                             , padding=1
#             #                             , groups=inplanes
#             #                             , bias=True))

#             # layers.append(SeparableConv2d(inplanes, planes, kernel_size=3, padding=1, bias=True))
#             layers.append(nn.ConvTranspose2d(
#                                     in_channels=inplanes,
#                                     out_channels=planes,
#                                     kernel_size=4,
#                                     stride=2,
#                                     padding=1,
#                                     output_padding=0,
#                                     bias=False))
#             layers.append(nn.BatchNorm2d(planes, momentum=True))
#             layers.append(nn.ReLU(inplace=True))
#             inplanes = planes
#         return nn.Sequential(*layers)


if __name__ == '__main__':
    # net = MobileNetV2()
    for netstr, F in {"resnet18": models.resnet18
                    , "squeezenet1_0":models.squeezenet1_0
                    , "mobilenet_v2":models.mobilenet_v2
                    , "mobilenet_v2_batch8":models.mobilenet_v2
                    , "resnet50":models.resnet50
                    , "resnet50_batch8":models.resnet50
                    , "mnasnet1_0":models.mnasnet1_0
                    , "wide_resnet50_2":models.wide_resnet50_2
                    , "centernet_mbv2": None
                    , "centernet_res50": None
                    , "centernet_hourglass": None
                    , "dilation": None
                    }.items():
        if netstr != "mobilenet_v2_batch8":
            continue

        print(netstr)
        if netstr == "centernet_mbv2":
            import sys
            sys.path.append('/workspace/centernet/src/lib')
            from models.model import create_model, load_model
            net = create_model('mobilenetv2liteheadtrans', {'hm': 29, 'wh': 2, 'reg': 2}, 256)
            net = load_model(net, '/workspace/centernet/exp/ctdet/mobilenetv2liteheadtrans_288x384/model_200.pth')
            inputT = torch.Tensor(1,3,288,384)
            input_size_list = [[3,288,384]]
            dataset_path = './baiguang-val-dataset.txt'
            analysis_path = './baiguang_anaysis.txt'
        elif netstr == "centernet_res50":
            import sys
            sys.path.append('/workspace/centernet/src/lib')
            from models.model import create_model, load_model
            net = create_model('res_50', {'hm': 29, 'wh': 2, 'reg': 2}, 256)
            inputT = torch.Tensor(1,3,288,384)
            input_size_list = [[3,288,384]]
            dataset_path = './dataset.txt'
        elif netstr == "centernet_hourglass":
            import sys
            sys.path.append('/workspace/centernet/src/lib')
            from models.model import create_model, load_model
            net = create_model('hourglass', {'hm': 29, 'wh': 2, 'reg': 2}, 256)
            inputT = torch.Tensor(1,3,512,512)
            input_size_list = [[3,512,512]]
            dataset_path = './dataset.txt'
        elif netstr == "dilation":
            class dilation(nn.Module):
                def __init__(self,):
                    super(dilation,self).__init__()
                    self.dia = nn.Conv2d(3,16,3,1,1,dilation=2)
                def forward(self,x):
                    x = self.dia(x)
                    return x
            net = dilation()
            inputT = torch.Tensor(1,3,224,224)
            input_size_list = [[3,224,224]]
            dataset_path = './dataset.txt'
        elif netstr == "resnet50_batch8" or netstr == "mobilenet_v2_batch8":
            net = F(pretrained=False)
            inputT = torch.Tensor(8,3,224,224)
            input_size_list = [[3,224,224]]
            dataset_path = './dataset.txt'
        else:
            net = F(pretrained=False)
            inputT = torch.Tensor(1,3,224,224)
            input_size_list = [[3,224,224]]
            dataset_path = './dataset.txt'

        net.eval()
        trace_model = torch.jit.trace(net, inputT)
        trace_model.save('./{}.pt'.format(netstr))

        model = './{}.pt'.format(netstr)
        

        # Create RKNN object
        rknn = RKNN()

        # pre-process config
        print('--> config model')
        if netstr == "centernet_mbv2":
            rknn.config(channel_mean_value='123.675 116.28 103.53 58.395'
                        , reorder_channel='0 1 2'
                        , target_platform='rv1126'
                        , batch_size = 200
                        # , quantized_dtype='asymmetric_quantized-u8'
                        # , quantized_dtype='dynamic_fixed_point-8'
                        , quantized_dtype='dynamic_fixed_point-16'
                        )
            # rknn.config(channel_mean_value='123.675 116.28 103.53 58.395'
            #             , reorder_channel='0 1 2'
            #             , target_platform='rv1126'
            #             )
        else:
            rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2', target_platform='rv1126')
        print('done')

        # Load pytorch model
        print('--> Loading model')
        ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
        if ret != 0:
            print('Load pytorch model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        
        if netstr == "centernet_mbv2":
            ret = rknn.build(do_quantization=True, dataset=dataset_path, pre_compile=True)
            # ret = rknn.build(do_quantization=False, pre_compile=True)
        elif netstr == "resnet50_batch8" or netstr == "mobilenet_v2_batch8":
            ret = rknn.build(do_quantization=True, dataset=dataset_path, pre_compile=True, rknn_batch_size=8)
        else:
            # ret = rknn.build(do_quantization=False, pre_compile=True)
            ret = rknn.build(do_quantization=True, dataset=dataset_path, pre_compile=True)
        if ret != 0:
            print('Build pytorch failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn('./{}.rknn'.format(netstr))
        if ret != 0:
            print('Export {}.rknn failed!'.format(netstr))
            exit(ret)
        print('done')

        # analysis
        if netstr == "centernet_mbv2":
            print('--> Analysis')
            rknn.accuracy_analysis(analysis_path, output_dir='./snapshot', calc_qnt_error=True)


        print('done')
