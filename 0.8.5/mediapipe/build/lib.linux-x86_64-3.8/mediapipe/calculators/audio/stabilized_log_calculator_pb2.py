# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/audio/stabilized_log_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n;mediapipe/calculators/audio/stabilized_log_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xd0\x01\n\x1eStabilizedLogCalculatorOptions\x12\x19\n\nstabilizer\x18\x01 \x01(\x02:\x05\x31\x65-05\x12!\n\x13\x63heck_nonnegativity\x18\x02 \x01(\x08:\x04true\x12\x17\n\x0coutput_scale\x18\x03 \x01(\x01:\x01\x31\x32W\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xe3\xa1\xd0\x30 \x01(\x0b\x32).mediapipe.StabilizedLogCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.audio.stabilized_log_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_STABILIZEDLOGCALCULATOROPTIONS']._serialized_start=113
  _globals['_STABILIZEDLOGCALCULATOROPTIONS']._serialized_end=321
# @@protoc_insertion_point(module_scope)
