# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/audio/time_series_framer_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n?mediapipe/calculators/audio/time_series_framer_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xc5\x03\n!TimeSeriesFramerCalculatorOptions\x12\x1e\n\x16\x66rame_duration_seconds\x18\x01 \x01(\x01\x12 \n\x15\x66rame_overlap_seconds\x18\x02 \x01(\x01:\x01\x30\x12/\n emulate_fractional_frame_overlap\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x1e\n\x10pad_final_packet\x18\x03 \x01(\x08:\x04true\x12Z\n\x0fwindow_function\x18\x04 \x01(\x0e\x32;.mediapipe.TimeSeriesFramerCalculatorOptions.WindowFunction:\x04NONE\x12\"\n\x13use_local_timestamp\x18\x06 \x01(\x08:\x05\x66\x61lse\"1\n\x0eWindowFunction\x12\x08\n\x04NONE\x10\x00\x12\x0b\n\x07HAMMING\x10\x01\x12\x08\n\x04HANN\x10\x02\x32Z\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xc5\xa7\x92\x18 \x01(\x0b\x32,.mediapipe.TimeSeriesFramerCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.audio.time_series_framer_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_TIMESERIESFRAMERCALCULATOROPTIONS']._serialized_start=117
  _globals['_TIMESERIESFRAMERCALCULATOROPTIONS']._serialized_end=570
  _globals['_TIMESERIESFRAMERCALCULATOROPTIONS_WINDOWFUNCTION']._serialized_start=429
  _globals['_TIMESERIESFRAMERCALCULATOROPTIONS_WINDOWFUNCTION']._serialized_end=478
# @@protoc_insertion_point(module_scope)
