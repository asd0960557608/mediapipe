# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/gpu/gl_scaler_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2
from mediapipe.gpu import scale_mode_pb2 as mediapipe_dot_gpu_dot_scale__mode__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n(mediapipe/gpu/gl_scaler_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x1emediapipe/gpu/scale_mode.proto\"\xa6\x02\n\x19GlScalerCalculatorOptions\x12\x14\n\x0coutput_width\x18\x01 \x01(\x05\x12\x15\n\routput_height\x18\x02 \x01(\x05\x12\x17\n\x0coutput_scale\x18\x07 \x01(\x02:\x01\x31\x12\x10\n\x08rotation\x18\x03 \x01(\x05\x12\x15\n\rflip_vertical\x18\x04 \x01(\x08\x12\x17\n\x0f\x66lip_horizontal\x18\x05 \x01(\x08\x12-\n\nscale_mode\x18\x06 \x01(\x0e\x32\x19.mediapipe.ScaleMode.Mode2R\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x96\xcd\xaaO \x01(\x0b\x32$.mediapipe.GlScalerCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.gpu.gl_scaler_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_GLSCALERCALCULATOROPTIONS']._serialized_start=126
  _globals['_GLSCALERCALCULATOROPTIONS']._serialized_end=420
# @@protoc_insertion_point(module_scope)
