# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/framework/formats/affine_transform_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n7mediapipe/framework/formats/affine_transform_data.proto\x12\tmediapipe\"#\n\x0bVector2Data\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\"\xa2\x01\n\x13\x41\x66\x66ineTransformData\x12+\n\x0btranslation\x18\x01 \x01(\x0b\x32\x16.mediapipe.Vector2Data\x12%\n\x05scale\x18\x02 \x01(\x0b\x32\x16.mediapipe.Vector2Data\x12%\n\x05shear\x18\x03 \x01(\x0b\x32\x16.mediapipe.Vector2Data\x12\x10\n\x08rotation\x18\x04 \x01(\x02')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.framework.formats.affine_transform_data_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_VECTOR2DATA']._serialized_start=70
  _globals['_VECTOR2DATA']._serialized_end=105
  _globals['_AFFINETRANSFORMDATA']._serialized_start=108
  _globals['_AFFINETRANSFORMDATA']._serialized_end=270
# @@protoc_insertion_point(module_scope)
