# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/modules/objectron/calculators/object.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n4mediapipe/modules/objectron/calculators/object.proto\x12\tmediapipe\"d\n\x08KeyPoint\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\x19\n\x11\x63onfidence_radius\x18\x04 \x01(\x02\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x0e\n\x06hidden\x18\x06 \x01(\x08\"\xd0\x02\n\x06Object\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x10\n\x08\x63\x61tegory\x18\x02 \x01(\t\x12$\n\x04type\x18\x03 \x01(\x0e\x32\x16.mediapipe.Object.Type\x12\x10\n\x08rotation\x18\x04 \x03(\x02\x12\x13\n\x0btranslation\x18\x05 \x03(\x02\x12\r\n\x05scale\x18\x06 \x03(\x02\x12&\n\tkeypoints\x18\x07 \x03(\x0b\x32\x13.mediapipe.KeyPoint\x12(\n\x06method\x18\x08 \x01(\x0e\x32\x18.mediapipe.Object.Method\":\n\x04Type\x12\x12\n\x0eUNDEFINED_TYPE\x10\x00\x12\x10\n\x0c\x42OUNDING_BOX\x10\x01\x12\x0c\n\x08SKELETON\x10\x02\">\n\x06Method\x12\x12\n\x0eUNKNOWN_METHOD\x10\x00\x12\x0e\n\nANNOTATION\x10\x01\x12\x10\n\x0c\x41UGMENTATION\x10\x02\"$\n\x04\x45\x64ge\x12\x0e\n\x06source\x18\x01 \x01(\x05\x12\x0c\n\x04sink\x18\x02 \x01(\x05\"\x80\x01\n\x08Skeleton\x12\x1a\n\x12reference_keypoint\x18\x01 \x01(\x05\x12\x10\n\x08\x63\x61tegory\x18\x02 \x01(\t\x12&\n\tkeypoints\x18\x03 \x03(\x0b\x32\x13.mediapipe.KeyPoint\x12\x1e\n\x05\x65\x64ges\x18\x04 \x03(\x0b\x32\x0f.mediapipe.Edge\"0\n\tSkeletons\x12#\n\x06object\x18\x01 \x03(\x0b\x32\x13.mediapipe.Skeletonb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.modules.objectron.calculators.object_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_KEYPOINT']._serialized_start=67
  _globals['_KEYPOINT']._serialized_end=167
  _globals['_OBJECT']._serialized_start=170
  _globals['_OBJECT']._serialized_end=506
  _globals['_OBJECT_TYPE']._serialized_start=384
  _globals['_OBJECT_TYPE']._serialized_end=442
  _globals['_OBJECT_METHOD']._serialized_start=444
  _globals['_OBJECT_METHOD']._serialized_end=506
  _globals['_EDGE']._serialized_start=508
  _globals['_EDGE']._serialized_end=544
  _globals['_SKELETON']._serialized_start=547
  _globals['_SKELETON']._serialized_end=675
  _globals['_SKELETONS']._serialized_start=677
  _globals['_SKELETONS']._serialized_end=725
# @@protoc_insertion_point(module_scope)