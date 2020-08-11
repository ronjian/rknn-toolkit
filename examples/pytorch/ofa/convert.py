# /workspace/once-for-all/note10_lat_64ms_top1_80.2_finetune_75.jit ********************
# WARNING: Token 'COMMENT' defined, but not used
# WARNING: There is 1 unused token
# E Catch exception when loading pytorch model: /workspace/once-for-all/note10_lat_64ms_top1_80.2_finetune_75.jit!
# E Traceback (most recent call last):
# E   File "rknn/api/rknn_base.py", line 611, in rknn.api.rknn_base.RKNNBase.load_pytorch
# E   File "rknn/base/RKNNlib/app/importer/import_pytorch.py", line 97, in rknn.base.RKNNlib.app.importer.import_pytorch.ImportPytorch.run
# E   File "rknn/base/RKNNlib/converter/convert_pytorch.py", line 570, in rknn.base.RKNNlib.converter.convert_pytorch.convert_pytorch.__init__
# E   File "rknn/base/RKNNlib/converter/convert_pytorch.py", line 654, in rknn.base.RKNNlib.converter.convert_pytorch.convert_pytorch.model_simplify
# E   File "rknn/base/RKNNlib/converter/convert_pytorch.py", line 113, in rknn.base.RKNNlib.converter.convert_pytorch.torch_inference_engine.shape_pick
# E   File "rknn/base/RKNNlib/converter/convert_pytorch.py", line 148, in rknn.base.RKNNlib.converter.convert_pytorch.torch_inference_engine.__ir_shape_inference
# E   File "rknn/base/RKNNlib/converter/convert_pytorch.py", line 220, in rknn.base.RKNNlib.converter.convert_pytorch.torch_inference_engine.elementwise_boardcast_shape
# E   File "rknn/base/RKNNlib/converter/convert_pytorch.py", line 113, in rknn.base.RKNNlib.converter.convert_pytorch.torch_inference_engine.shape_pick
# E   File "rknn/base/RKNNlib/converter/convert_pytorch.py", line 148, in rknn.base.RKNNlib.converter.convert_pytorch.torch_inference_engine.__ir_shape_inference
# E KeyError: 'aten::matmul'
# Load pytorch model failed!

from rknn.api import RKNN

# Create RKNN object
rknn = RKNN()
input_size_list = [[3,208,208]]

# pre-process config
print('--> config model')
rknn.config(channel_mean_value='0. 0. 0. 255.'
            , reorder_channel='0 1 2'
            , target_platform='rv1126'
            , batch_size = 10 # for quantize
            )
print('done')

# Load pytorch model
print('--> Loading model')
ret = rknn.load_pytorch(model='/workspace/once-for-all/cpu_lat_17ms_top1_75.7_finetune_25.jit', input_size_list=input_size_list)
if ret != 0:
    print('Load pytorch model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=True
            , dataset='../yolov5s/dataset.txt'
            , pre_compile=True
            , rknn_batch_size=1)
if ret != 0:
    print('Build pytorch failed!')
    exit(ret)
print('done')

# Export rknn model
print('--> Export RKNN model')
ret = rknn.export_rknn('./cpu_lat_17ms_top1_75.7_finetune_25.rknn')
if ret != 0:
    print('Export failed!')
    exit(ret)
print('done')
