# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/detections_to_render_data_calculator.proto
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
from mediapipe.util import render_data_pb2 as mediapipe_dot_util_dot_render__data__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nEmediapipe/calculators/util/detections_to_render_data_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x1amediapipe/util/color.proto\x1a mediapipe/util/render_data.proto\"\x98\x03\n\'DetectionsToRenderDataCalculatorOptions\x12\"\n\x14produce_empty_packet\x18\x01 \x01(\x08:\x04true\x12\x19\n\x0etext_delimiter\x18\x02 \x01(\t:\x01,\x12!\n\x12one_label_per_line\x18\x03 \x01(\x08:\x05\x66\x61lse\x12.\n\x04text\x18\x04 \x01(\x0b\x32 .mediapipe.RenderAnnotation.Text\x12\x14\n\tthickness\x18\x05 \x01(\x01:\x01\x31\x12\x1f\n\x05\x63olor\x18\x06 \x01(\x0b\x32\x10.mediapipe.Color\x12\x1e\n\x0bscene_class\x18\x07 \x01(\t:\tDETECTION\x12\"\n\x13render_detection_id\x18\x08 \x01(\x08:\x05\x66\x61lse2`\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xe6\xde\xb6v \x01(\x0b\x32\x32.mediapipe.DetectionsToRenderDataCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.detections_to_render_data_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_DETECTIONSTORENDERDATACALCULATOROPTIONS']._serialized_start=185
  _globals['_DETECTIONSTORENDERDATACALCULATOROPTIONS']._serialized_end=593
# @@protoc_insertion_point(module_scope)
