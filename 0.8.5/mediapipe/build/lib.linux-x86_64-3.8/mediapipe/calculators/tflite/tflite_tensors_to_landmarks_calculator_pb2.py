# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nImediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xa6\x04\n)TfLiteTensorsToLandmarksCalculatorOptions\x12\x15\n\rnum_landmarks\x18\x01 \x02(\x05\x12\x19\n\x11input_image_width\x18\x02 \x01(\x05\x12\x1a\n\x12input_image_height\x18\x03 \x01(\x05\x12\x1e\n\x0f\x66lip_vertically\x18\x04 \x01(\x08:\x05\x66\x61lse\x12 \n\x11\x66lip_horizontally\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x16\n\x0bnormalize_z\x18\x05 \x01(\x02:\x01\x31\x12\x64\n\x15visibility_activation\x18\x07 \x01(\x0e\x32?.mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.Activation:\x04NONE\x12\x62\n\x13presence_activation\x18\x08 \x01(\x0e\x32?.mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.Activation:\x04NONE\"#\n\nActivation\x12\x08\n\x04NONE\x10\x00\x12\x0b\n\x07SIGMOID\x10\x01\x32\x62\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xca\xe0\xdez \x01(\x0b\x32\x34.mediapipe.TfLiteTensorsToLandmarksCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.tflite.tflite_tensors_to_landmarks_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS']._serialized_start=127
  _globals['_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS']._serialized_end=677
  _globals['_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION']._serialized_start=542
  _globals['_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION']._serialized_end=577
# @@protoc_insertion_point(module_scope)