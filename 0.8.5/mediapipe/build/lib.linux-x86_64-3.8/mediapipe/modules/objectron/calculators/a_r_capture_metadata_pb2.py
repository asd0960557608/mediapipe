# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/modules/objectron/calculators/a_r_capture_metadata.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nBmediapipe/modules/objectron/calculators/a_r_capture_metadata.proto\x12\tmediapipe\"\xf4\x02\n\x17\x41VCameraCalibrationData\x12\x1c\n\x10intrinsic_matrix\x18\x01 \x03(\x02\x42\x02\x10\x01\x12\x32\n*intrinsic_matrix_reference_dimension_width\x18\x02 \x01(\x02\x12\x33\n+intrinsic_matrix_reference_dimension_height\x18\x03 \x01(\x02\x12\x1c\n\x10\x65xtrinsic_matrix\x18\x04 \x03(\x02\x42\x02\x10\x01\x12\x12\n\npixel_size\x18\x05 \x01(\x02\x12)\n\x1dlens_distortion_lookup_values\x18\x06 \x03(\x02\x42\x02\x10\x01\x12\x31\n%inverse_lens_distortion_lookup_values\x18\x07 \x03(\x02\x42\x02\x10\x01\x12 \n\x18lens_distortion_center_x\x18\x08 \x01(\x02\x12 \n\x18lens_distortion_center_y\x18\t \x01(\x02\"\xd7\x04\n\x0b\x41VDepthData\x12\x16\n\x0e\x64\x65pth_data_map\x18\x01 \x01(\x0c\x12\x17\n\x0f\x64\x65pth_data_type\x18\x02 \x01(\t\x12\x46\n\x13\x64\x65pth_data_accuracy\x18\x03 \x01(\x0e\x32\x1f.mediapipe.AVDepthData.Accuracy:\x08RELATIVE\x12\x1b\n\x13\x64\x65pth_data_filtered\x18\x04 \x01(\x08\x12:\n\x12\x64\x65pth_data_quality\x18\x05 \x01(\x0e\x32\x1e.mediapipe.AVDepthData.Quality\x12\x43\n\x17\x63\x61mera_calibration_data\x18\x06 \x01(\x0b\x32\".mediapipe.AVCameraCalibrationData\x12-\n%depth_data_map_original_minimum_value\x18\x07 \x01(\x02\x12-\n%depth_data_map_original_maximum_value\x18\x08 \x01(\x02\x12\x1c\n\x14\x64\x65pth_data_map_width\x18\t \x01(\x05\x12\x1d\n\x15\x64\x65pth_data_map_height\x18\n \x01(\x05\x12!\n\x19\x64\x65pth_data_map_raw_values\x18\x0b \x01(\x0c\">\n\x08\x41\x63\x63uracy\x12\x16\n\x12UNDEFINED_ACCURACY\x10\x00\x12\x0c\n\x08RELATIVE\x10\x01\x12\x0c\n\x08\x41\x42SOLUTE\x10\x02\"3\n\x07Quality\x12\x15\n\x11UNDEFINED_QUALITY\x10\x00\x12\x08\n\x04HIGH\x10\x01\x12\x07\n\x03LOW\x10\x02\"\x9f\x02\n\x0f\x41RLightEstimate\x12\x19\n\x11\x61mbient_intensity\x18\x01 \x01(\x01\x12!\n\x19\x61mbient_color_temperature\x18\x02 \x01(\x01\x12,\n spherical_harmonics_coefficients\x18\x03 \x03(\x02\x42\x02\x10\x01\x12K\n\x17primary_light_direction\x18\x04 \x01(\x0b\x32*.mediapipe.ARLightEstimate.DirectionVector\x12\x1f\n\x17primary_light_intensity\x18\x05 \x01(\x02\x1a\x32\n\x0f\x44irectionVector\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"\xad\x05\n\x08\x41RCamera\x12\x46\n\x0etracking_state\x18\x01 \x01(\x0e\x32!.mediapipe.ARCamera.TrackingState:\x0bUNAVAILABLE\x12L\n\x15tracking_state_reason\x18\x02 \x01(\x0e\x32\'.mediapipe.ARCamera.TrackingStateReason:\x04NONE\x12\x15\n\ttransform\x18\x03 \x03(\x02\x42\x02\x10\x01\x12\x35\n\x0c\x65uler_angles\x18\x04 \x01(\x0b\x32\x1f.mediapipe.ARCamera.EulerAngles\x12\x1e\n\x16image_resolution_width\x18\x05 \x01(\x05\x12\x1f\n\x17image_resolution_height\x18\x06 \x01(\x05\x12\x16\n\nintrinsics\x18\x07 \x03(\x02\x42\x02\x10\x01\x12\x1d\n\x11projection_matrix\x18\x08 \x03(\x02\x42\x02\x10\x01\x12\x17\n\x0bview_matrix\x18\t \x03(\x02\x42\x02\x10\x01\x1a\x37\n\x0b\x45ulerAngles\x12\x0c\n\x04roll\x18\x01 \x01(\x02\x12\r\n\x05pitch\x18\x02 \x01(\x02\x12\x0b\n\x03yaw\x18\x03 \x01(\x02\"W\n\rTrackingState\x12\x1c\n\x18UNDEFINED_TRACKING_STATE\x10\x00\x12\x0f\n\x0bUNAVAILABLE\x10\x01\x12\x0b\n\x07LIMITED\x10\x02\x12\n\n\x06NORMAL\x10\x03\"\x99\x01\n\x13TrackingStateReason\x12#\n\x1fUNDEFINED_TRACKING_STATE_REASON\x10\x00\x12\x08\n\x04NONE\x10\x01\x12\x10\n\x0cINITIALIZING\x10\x02\x12\x14\n\x10\x45XCESSIVE_MOTION\x10\x03\x12\x19\n\x15INSUFFICIENT_FEATURES\x10\x04\x12\x10\n\x0cRELOCALIZING\x10\x05\"\xd2\x02\n\x0e\x41RFaceGeometry\x12\x32\n\x08vertices\x18\x01 \x03(\x0b\x32 .mediapipe.ARFaceGeometry.Vertex\x12\x14\n\x0cvertex_count\x18\x02 \x01(\x05\x12H\n\x13texture_coordinates\x18\x03 \x03(\x0b\x32+.mediapipe.ARFaceGeometry.TextureCoordinate\x12 \n\x18texture_coordinate_count\x18\x04 \x01(\x05\x12\x1c\n\x10triangle_indices\x18\x05 \x03(\x05\x42\x02\x10\x01\x12\x16\n\x0etriangle_count\x18\x06 \x01(\x05\x1a)\n\x06Vertex\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x1a)\n\x11TextureCoordinate\x12\t\n\x01u\x18\x01 \x01(\x02\x12\t\n\x01v\x18\x02 \x01(\x02\"\x92\x01\n\x0f\x41RBlendShapeMap\x12\x34\n\x07\x65ntries\x18\x01 \x03(\x0b\x32#.mediapipe.ARBlendShapeMap.MapEntry\x1aI\n\x08MapEntry\x12\x1c\n\x14\x62lend_shape_location\x18\x01 \x01(\t\x12\x1f\n\x17\x62lend_shape_coefficient\x18\x02 \x01(\x02\"\x94\x01\n\x0c\x41RFaceAnchor\x12+\n\x08geometry\x18\x01 \x01(\x0b\x32\x19.mediapipe.ARFaceGeometry\x12\x30\n\x0c\x62lend_shapes\x18\x02 \x01(\x0b\x32\x1a.mediapipe.ARBlendShapeMap\x12\x11\n\ttransform\x18\x03 \x03(\x02\x12\x12\n\nis_tracked\x18\x04 \x01(\x08\"\xb2\x03\n\x0f\x41RPlaneGeometry\x12\x33\n\x08vertices\x18\x01 \x03(\x0b\x32!.mediapipe.ARPlaneGeometry.Vertex\x12\x14\n\x0cvertex_count\x18\x02 \x01(\x05\x12I\n\x13texture_coordinates\x18\x03 \x03(\x0b\x32,.mediapipe.ARPlaneGeometry.TextureCoordinate\x12 \n\x18texture_coordinate_count\x18\x04 \x01(\x05\x12\x1c\n\x10triangle_indices\x18\x05 \x03(\x05\x42\x02\x10\x01\x12\x16\n\x0etriangle_count\x18\x06 \x01(\x05\x12<\n\x11\x62oundary_vertices\x18\x07 \x03(\x0b\x32!.mediapipe.ARPlaneGeometry.Vertex\x12\x1d\n\x15\x62oundary_vertex_count\x18\x08 \x01(\x05\x1a)\n\x06Vertex\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x1a)\n\x11TextureCoordinate\x12\t\n\x01u\x18\x01 \x01(\x02\x12\t\n\x01v\x18\x02 \x01(\x02\"\xdc\x05\n\rARPlaneAnchor\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x11\n\ttransform\x18\x02 \x03(\x02\x12\x35\n\talignment\x18\x03 \x01(\x0e\x32\".mediapipe.ARPlaneAnchor.Alignment\x12,\n\x08geometry\x18\x04 \x01(\x0b\x32\x1a.mediapipe.ARPlaneGeometry\x12\x34\n\x06\x63\x65nter\x18\x05 \x01(\x0b\x32$.mediapipe.ARPlaneAnchor.PlaneVector\x12\x34\n\x06\x65xtent\x18\x06 \x01(\x0b\x32$.mediapipe.ARPlaneAnchor.PlaneVector\x12 \n\x18\x63lassification_supported\x18\x07 \x01(\x08\x12\x44\n\x0e\x63lassification\x18\x08 \x01(\x0e\x32,.mediapipe.ARPlaneAnchor.PlaneClassification\x12Q\n\x15\x63lassification_status\x18\t \x01(\x0e\x32\x32.mediapipe.ARPlaneAnchor.PlaneClassificationStatus\x1a.\n\x0bPlaneVector\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"8\n\tAlignment\x12\r\n\tUNDEFINED\x10\x00\x12\x0e\n\nHORIZONTAL\x10\x01\x12\x0c\n\x08VERTICAL\x10\x02\"V\n\x13PlaneClassification\x12\x08\n\x04NONE\x10\x00\x12\x08\n\x04WALL\x10\x01\x12\t\n\x05\x46LOOR\x10\x02\x12\x0b\n\x07\x43\x45ILING\x10\x03\x12\t\n\x05TABLE\x10\x04\x12\x08\n\x04SEAT\x10\x05\"V\n\x19PlaneClassificationStatus\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0f\n\x0bUNAVAILABLE\x10\x01\x12\x10\n\x0cUNDETERMINED\x10\x02\x12\t\n\x05KNOWN\x10\x03\"\x8d\x01\n\x0c\x41RPointCloud\x12\r\n\x05\x63ount\x18\x01 \x01(\x05\x12,\n\x05point\x18\x02 \x03(\x0b\x32\x1d.mediapipe.ARPointCloud.Point\x12\x16\n\nidentifier\x18\x03 \x03(\x03\x42\x02\x10\x01\x1a(\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"\xd2\x02\n\x07\x41RFrame\x12\x11\n\ttimestamp\x18\x01 \x01(\x01\x12*\n\ndepth_data\x18\x02 \x01(\x0b\x32\x16.mediapipe.AVDepthData\x12\x1c\n\x14\x64\x65pth_data_timestamp\x18\x03 \x01(\x01\x12#\n\x06\x63\x61mera\x18\x04 \x01(\x0b\x32\x13.mediapipe.ARCamera\x12\x32\n\x0elight_estimate\x18\x05 \x01(\x0b\x32\x1a.mediapipe.ARLightEstimate\x12,\n\x0b\x66\x61\x63\x65_anchor\x18\x06 \x01(\x0b\x32\x17.mediapipe.ARFaceAnchor\x12.\n\x0cplane_anchor\x18\x07 \x03(\x0b\x32\x18.mediapipe.ARPlaneAnchor\x12\x33\n\x12raw_feature_points\x18\x08 \x01(\x0b\x32\x17.mediapipe.ARPointCloud')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.modules.objectron.calculators.a_r_capture_metadata_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_AVCAMERACALIBRATIONDATA'].fields_by_name['intrinsic_matrix']._options = None
  _globals['_AVCAMERACALIBRATIONDATA'].fields_by_name['intrinsic_matrix']._serialized_options = b'\020\001'
  _globals['_AVCAMERACALIBRATIONDATA'].fields_by_name['extrinsic_matrix']._options = None
  _globals['_AVCAMERACALIBRATIONDATA'].fields_by_name['extrinsic_matrix']._serialized_options = b'\020\001'
  _globals['_AVCAMERACALIBRATIONDATA'].fields_by_name['lens_distortion_lookup_values']._options = None
  _globals['_AVCAMERACALIBRATIONDATA'].fields_by_name['lens_distortion_lookup_values']._serialized_options = b'\020\001'
  _globals['_AVCAMERACALIBRATIONDATA'].fields_by_name['inverse_lens_distortion_lookup_values']._options = None
  _globals['_AVCAMERACALIBRATIONDATA'].fields_by_name['inverse_lens_distortion_lookup_values']._serialized_options = b'\020\001'
  _globals['_ARLIGHTESTIMATE'].fields_by_name['spherical_harmonics_coefficients']._options = None
  _globals['_ARLIGHTESTIMATE'].fields_by_name['spherical_harmonics_coefficients']._serialized_options = b'\020\001'
  _globals['_ARCAMERA'].fields_by_name['transform']._options = None
  _globals['_ARCAMERA'].fields_by_name['transform']._serialized_options = b'\020\001'
  _globals['_ARCAMERA'].fields_by_name['intrinsics']._options = None
  _globals['_ARCAMERA'].fields_by_name['intrinsics']._serialized_options = b'\020\001'
  _globals['_ARCAMERA'].fields_by_name['projection_matrix']._options = None
  _globals['_ARCAMERA'].fields_by_name['projection_matrix']._serialized_options = b'\020\001'
  _globals['_ARCAMERA'].fields_by_name['view_matrix']._options = None
  _globals['_ARCAMERA'].fields_by_name['view_matrix']._serialized_options = b'\020\001'
  _globals['_ARFACEGEOMETRY'].fields_by_name['triangle_indices']._options = None
  _globals['_ARFACEGEOMETRY'].fields_by_name['triangle_indices']._serialized_options = b'\020\001'
  _globals['_ARPLANEGEOMETRY'].fields_by_name['triangle_indices']._options = None
  _globals['_ARPLANEGEOMETRY'].fields_by_name['triangle_indices']._serialized_options = b'\020\001'
  _globals['_ARPOINTCLOUD'].fields_by_name['identifier']._options = None
  _globals['_ARPOINTCLOUD'].fields_by_name['identifier']._serialized_options = b'\020\001'
  _globals['_AVCAMERACALIBRATIONDATA']._serialized_start=82
  _globals['_AVCAMERACALIBRATIONDATA']._serialized_end=454
  _globals['_AVDEPTHDATA']._serialized_start=457
  _globals['_AVDEPTHDATA']._serialized_end=1056
  _globals['_AVDEPTHDATA_ACCURACY']._serialized_start=941
  _globals['_AVDEPTHDATA_ACCURACY']._serialized_end=1003
  _globals['_AVDEPTHDATA_QUALITY']._serialized_start=1005
  _globals['_AVDEPTHDATA_QUALITY']._serialized_end=1056
  _globals['_ARLIGHTESTIMATE']._serialized_start=1059
  _globals['_ARLIGHTESTIMATE']._serialized_end=1346
  _globals['_ARLIGHTESTIMATE_DIRECTIONVECTOR']._serialized_start=1296
  _globals['_ARLIGHTESTIMATE_DIRECTIONVECTOR']._serialized_end=1346
  _globals['_ARCAMERA']._serialized_start=1349
  _globals['_ARCAMERA']._serialized_end=2034
  _globals['_ARCAMERA_EULERANGLES']._serialized_start=1734
  _globals['_ARCAMERA_EULERANGLES']._serialized_end=1789
  _globals['_ARCAMERA_TRACKINGSTATE']._serialized_start=1791
  _globals['_ARCAMERA_TRACKINGSTATE']._serialized_end=1878
  _globals['_ARCAMERA_TRACKINGSTATEREASON']._serialized_start=1881
  _globals['_ARCAMERA_TRACKINGSTATEREASON']._serialized_end=2034
  _globals['_ARFACEGEOMETRY']._serialized_start=2037
  _globals['_ARFACEGEOMETRY']._serialized_end=2375
  _globals['_ARFACEGEOMETRY_VERTEX']._serialized_start=2291
  _globals['_ARFACEGEOMETRY_VERTEX']._serialized_end=2332
  _globals['_ARFACEGEOMETRY_TEXTURECOORDINATE']._serialized_start=2334
  _globals['_ARFACEGEOMETRY_TEXTURECOORDINATE']._serialized_end=2375
  _globals['_ARBLENDSHAPEMAP']._serialized_start=2378
  _globals['_ARBLENDSHAPEMAP']._serialized_end=2524
  _globals['_ARBLENDSHAPEMAP_MAPENTRY']._serialized_start=2451
  _globals['_ARBLENDSHAPEMAP_MAPENTRY']._serialized_end=2524
  _globals['_ARFACEANCHOR']._serialized_start=2527
  _globals['_ARFACEANCHOR']._serialized_end=2675
  _globals['_ARPLANEGEOMETRY']._serialized_start=2678
  _globals['_ARPLANEGEOMETRY']._serialized_end=3112
  _globals['_ARPLANEGEOMETRY_VERTEX']._serialized_start=2291
  _globals['_ARPLANEGEOMETRY_VERTEX']._serialized_end=2332
  _globals['_ARPLANEGEOMETRY_TEXTURECOORDINATE']._serialized_start=2334
  _globals['_ARPLANEGEOMETRY_TEXTURECOORDINATE']._serialized_end=2375
  _globals['_ARPLANEANCHOR']._serialized_start=3115
  _globals['_ARPLANEANCHOR']._serialized_end=3847
  _globals['_ARPLANEANCHOR_PLANEVECTOR']._serialized_start=3567
  _globals['_ARPLANEANCHOR_PLANEVECTOR']._serialized_end=3613
  _globals['_ARPLANEANCHOR_ALIGNMENT']._serialized_start=3615
  _globals['_ARPLANEANCHOR_ALIGNMENT']._serialized_end=3671
  _globals['_ARPLANEANCHOR_PLANECLASSIFICATION']._serialized_start=3673
  _globals['_ARPLANEANCHOR_PLANECLASSIFICATION']._serialized_end=3759
  _globals['_ARPLANEANCHOR_PLANECLASSIFICATIONSTATUS']._serialized_start=3761
  _globals['_ARPLANEANCHOR_PLANECLASSIFICATIONSTATUS']._serialized_end=3847
  _globals['_ARPOINTCLOUD']._serialized_start=3850
  _globals['_ARPOINTCLOUD']._serialized_end=3991
  _globals['_ARPOINTCLOUD_POINT']._serialized_start=3951
  _globals['_ARPOINTCLOUD_POINT']._serialized_end=3991
  _globals['_ARFRAME']._serialized_start=3994
  _globals['_ARFRAME']._serialized_end=4332
# @@protoc_insertion_point(module_scope)
