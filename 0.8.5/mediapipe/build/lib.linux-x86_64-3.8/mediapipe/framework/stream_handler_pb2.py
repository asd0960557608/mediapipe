# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/framework/stream_handler.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import mediapipe_options_pb2 as mediapipe_dot_framework_dot_mediapipe__options__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n(mediapipe/framework/stream_handler.proto\x12\tmediapipe\x1a+mediapipe/framework/mediapipe_options.proto\"\x81\x01\n\x18InputStreamHandlerConfig\x12\x37\n\x14input_stream_handler\x18\x01 \x01(\t:\x19\x44\x65\x66\x61ultInputStreamHandler\x12,\n\x07options\x18\x03 \x01(\x0b\x32\x1b.mediapipe.MediaPipeOptions\"\x9f\x01\n\x19OutputStreamHandlerConfig\x12\x39\n\x15output_stream_handler\x18\x01 \x01(\t:\x1aInOrderOutputStreamHandler\x12\x19\n\x11input_side_packet\x18\x02 \x03(\t\x12,\n\x07options\x18\x03 \x01(\x0b\x32\x1b.mediapipe.MediaPipeOptionsB<\n\x1a\x63om.google.mediapipe.protoB\x12StreamHandlerProto\xa2\x02\tMediaPipe')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.framework.stream_handler_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\032com.google.mediapipe.protoB\022StreamHandlerProto\242\002\tMediaPipe'
  _globals['_INPUTSTREAMHANDLERCONFIG']._serialized_start=101
  _globals['_INPUTSTREAMHANDLERCONFIG']._serialized_end=230
  _globals['_OUTPUTSTREAMHANDLERCONFIG']._serialized_start=233
  _globals['_OUTPUTSTREAMHANDLERCONFIG']._serialized_end=392
# @@protoc_insertion_point(module_scope)
