# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/core/graph_profile_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n9mediapipe/calculators/core/graph_profile_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\x9b\x01\n\x1dGraphProfileCalculatorOptions\x12!\n\x10profile_interval\x18\x01 \x01(\x03:\x07\x31\x30\x30\x30\x30\x30\x30\x32W\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xd7\xa7\x9d\xaf\x01 \x01(\x0b\x32(.mediapipe.GraphProfileCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.core.graph_profile_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_GRAPHPROFILECALCULATOROPTIONS']._serialized_start=111
  _globals['_GRAPHPROFILECALCULATOROPTIONS']._serialized_end=266
# @@protoc_insertion_point(module_scope)
