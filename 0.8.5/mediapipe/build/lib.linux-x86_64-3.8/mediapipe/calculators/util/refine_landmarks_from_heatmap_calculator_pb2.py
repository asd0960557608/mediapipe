# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nImediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\x95\x02\n+RefineLandmarksFromHeatmapCalculatorOptions\x12\x16\n\x0bkernel_size\x18\x01 \x01(\x05:\x01\x39\x12%\n\x18min_confidence_to_refine\x18\x02 \x01(\x02:\x03\x30.5\x12\x1e\n\x0frefine_presence\x18\x03 \x01(\x08:\x05\x66\x61lse\x12 \n\x11refine_visibility\x18\x04 \x01(\x08:\x05\x66\x61lse2e\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xb5\xf5\xdf\xac\x01 \x01(\x0b\x32\x36.mediapipe.RefineLandmarksFromHeatmapCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.refine_landmarks_from_heatmap_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REFINELANDMARKSFROMHEATMAPCALCULATOROPTIONS']._serialized_start=127
  _globals['_REFINELANDMARKSFROMHEATMAPCALCULATOROPTIONS']._serialized_end=404
# @@protoc_insertion_point(module_scope)
