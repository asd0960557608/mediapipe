# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/latency.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n(mediapipe/calculators/util/latency.proto\x12\tmediapipe\"\xca\x01\n\rPacketLatency\x12\x1c\n\x14\x63urrent_latency_usec\x18\x08 \x01(\x03\x12\x0e\n\x06\x63ounts\x18\t \x03(\x03\x12\x19\n\rnum_intervals\x18\n \x01(\x03:\x02\x31\x30\x12!\n\x12interval_size_usec\x18\x0b \x01(\x03:\x05\x31\x30\x30\x30\x30\x12\x18\n\x10\x61vg_latency_usec\x18\x02 \x01(\x03\x12\r\n\x05label\x18\x07 \x01(\t\x12\x18\n\x10sum_latency_usec\x18\x0c \x01(\x03J\x04\x08\x01\x10\x02J\x04\x08\x03\x10\x07')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.latency_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_PACKETLATENCY']._serialized_start=56
  _globals['_PACKETLATENCY']._serialized_end=258
# @@protoc_insertion_point(module_scope)
