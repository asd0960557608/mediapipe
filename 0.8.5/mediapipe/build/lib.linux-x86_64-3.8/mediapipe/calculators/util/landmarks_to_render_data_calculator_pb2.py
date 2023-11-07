# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/landmarks_to_render_data_calculator.proto
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
from mediapipe.util import color_pb2 as mediapipe_dot_util_dot_color__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nDmediapipe/calculators/util/landmarks_to_render_data_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x1amediapipe/util/color.proto\"\xee\x04\n&LandmarksToRenderDataCalculatorOptions\x12\x1c\n\x14landmark_connections\x18\x01 \x03(\x05\x12(\n\x0elandmark_color\x18\x02 \x01(\x0b\x32\x10.mediapipe.Color\x12*\n\x10\x63onnection_color\x18\x03 \x01(\x0b\x32\x10.mediapipe.Color\x12\x14\n\tthickness\x18\x04 \x01(\x01:\x01\x31\x12&\n\x18visualize_landmark_depth\x18\x05 \x01(\x08:\x04true\x12!\n\x12utilize_visibility\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x1f\n\x14visibility_threshold\x18\x07 \x01(\x01:\x01\x30\x12\x1f\n\x10utilize_presence\x18\x08 \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x12presence_threshold\x18\t \x01(\x01:\x01\x30\x12%\n\x1amin_depth_circle_thickness\x18\n \x01(\x01:\x01\x30\x12&\n\x1amax_depth_circle_thickness\x18\x0b \x01(\x01:\x02\x31\x38\x12.\n\x14min_depth_line_color\x18\x0c \x01(\x0b\x32\x10.mediapipe.Color\x12.\n\x14max_depth_line_color\x18\r \x01(\x0b\x32\x10.mediapipe.Color2_\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xbd\xd2\x9d{ \x01(\x0b\x32\x31.mediapipe.LandmarksToRenderDataCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.landmarks_to_render_data_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_LANDMARKSTORENDERDATACALCULATOROPTIONS']._serialized_start=150
  _globals['_LANDMARKSTORENDERDATACALCULATOROPTIONS']._serialized_end=772
# @@protoc_insertion_point(module_scope)
