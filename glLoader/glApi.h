//
//   Copyright 2020 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#ifndef OPENSUBDIV_GLLOADER_GLAPILOADER_H
#define OPENSUBDIV_GLLOADER_GLAPILOADER_H

#if defined(__gl_h_) || defined(__gl3_h_)
    #error platform OpenGL header included before this header
#endif
#define __gl_h_
#define __gl3_h_

#include "khrplatform.h"

#ifdef _WIN32
#define GLAPIENTRY __stdcall
#else
#define GLAPIENTRY
#endif

#define GLAPIENTRYP GLAPIENTRY*

typedef unsigned int GLenum;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef void GLvoid;
typedef khronos_int8_t GLbyte;
typedef khronos_uint8_t GLubyte;
typedef khronos_int16_t GLshort;
typedef khronos_uint16_t GLushort;
typedef int GLint;
typedef unsigned int GLuint;
typedef khronos_int32_t GLclampx;
typedef int GLsizei;
typedef khronos_float_t GLfloat;
typedef khronos_float_t GLclampf;
typedef double GLdouble;
typedef double GLclampd;
typedef void *GLeglClientBufferEXT;
typedef void *GLeglImageOES;
typedef char GLchar;
typedef char GLcharARB;
#ifdef __APPLE__
typedef void *GLhandleARB;
#else
typedef unsigned int GLhandleARB;
#endif
typedef khronos_uint16_t GLhalf;
typedef khronos_uint16_t GLhalfARB;
typedef khronos_int32_t GLfixed;
typedef khronos_intptr_t GLintptr;
typedef khronos_intptr_t GLintptrARB;
typedef khronos_ssize_t GLsizeiptr;
typedef khronos_ssize_t GLsizeiptrARB;
typedef khronos_int64_t GLint64;
typedef khronos_int64_t GLint64EXT;
typedef khronos_uint64_t GLuint64;
typedef khronos_uint64_t GLuint64EXT;
typedef struct __GLsync *GLsync;
struct _cl_context;
struct _cl_event;
typedef void (GLAPIENTRY  *GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
typedef void (GLAPIENTRY  *GLDEBUGPROCARB)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
typedef void (GLAPIENTRY  *GLDEBUGPROCKHR)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
typedef void (GLAPIENTRY  *GLDEBUGPROCAMD)(GLuint id,GLenum category,GLenum severity,GLsizei length,const GLchar *message,void *userParam);
typedef unsigned short GLhalfNV;
typedef GLintptr GLvdpauSurfaceNV;
typedef void (GLAPIENTRY  *GLVULKANPROCNV)(void);


#define GL_VERSION_1_0 1
#define GL_VERSION_1_1 1
#define GL_VERSION_1_2 1
#define GL_VERSION_1_3 1
#define GL_VERSION_1_4 1
#define GL_VERSION_1_5 1
#define GL_VERSION_2_0 1
#define GL_VERSION_2_1 1
#define GL_VERSION_3_0 1
#define GL_VERSION_3_1 1
#define GL_VERSION_3_2 1
#define GL_VERSION_3_3 1
#define GL_VERSION_4_0 1
#define GL_VERSION_4_1 1
#define GL_VERSION_4_2 1
#define GL_VERSION_4_3 1
#define GL_VERSION_4_4 1
#define GL_VERSION_4_5 1
#define GL_VERSION_4_6 1


#define GL_AMD_blend_minmax_factor 1
#define GL_AMD_conservative_depth 1
#define GL_AMD_debug_output 1
#define GL_AMD_depth_clamp_separate 1
#define GL_AMD_draw_buffers_blend 1
#define GL_AMD_framebuffer_multisample_advanced 1
#define GL_AMD_framebuffer_sample_positions 1
#define GL_AMD_gcn_shader 1
#define GL_AMD_gpu_shader_half_float 1
#define GL_AMD_gpu_shader_int16 1
#define GL_AMD_gpu_shader_int64 1
#define GL_AMD_interleaved_elements 1
#define GL_AMD_multi_draw_indirect 1
#define GL_AMD_name_gen_delete 1
#define GL_AMD_occlusion_query_event 1
#define GL_AMD_performance_monitor 1
#define GL_AMD_pinned_memory 1
#define GL_AMD_query_buffer_object 1
#define GL_AMD_sample_positions 1
#define GL_AMD_seamless_cubemap_per_texture 1
#define GL_AMD_shader_atomic_counter_ops 1
#define GL_AMD_shader_ballot 1
#define GL_AMD_shader_gpu_shader_half_float_fetch 1
#define GL_AMD_shader_image_load_store_lod 1
#define GL_AMD_shader_stencil_export 1
#define GL_AMD_shader_trinary_minmax 1
#define GL_AMD_shader_explicit_vertex_parameter 1
#define GL_AMD_sparse_texture 1
#define GL_AMD_stencil_operation_extended 1
#define GL_AMD_texture_gather_bias_lod 1
#define GL_AMD_texture_texture4 1
#define GL_AMD_transform_feedback3_lines_triangles 1
#define GL_AMD_transform_feedback4 1
#define GL_AMD_vertex_shader_layer 1
#define GL_AMD_vertex_shader_tessellator 1
#define GL_AMD_vertex_shader_viewport_index 1
#define GL_APPLE_aux_depth_stencil 1
#define GL_APPLE_client_storage 1
#define GL_APPLE_element_array 1
#define GL_APPLE_fence 1
#define GL_APPLE_float_pixels 1
#define GL_APPLE_flush_buffer_range 1
#define GL_APPLE_object_purgeable 1
#define GL_APPLE_rgb_422 1
#define GL_APPLE_row_bytes 1
#define GL_APPLE_specular_vector 1
#define GL_APPLE_texture_range 1
#define GL_APPLE_transform_hint 1
#define GL_APPLE_vertex_array_object 1
#define GL_APPLE_vertex_array_range 1
#define GL_APPLE_vertex_program_evaluators 1
#define GL_APPLE_ycbcr_422 1
#define GL_ARB_ES2_compatibility 1
#define GL_ARB_ES3_1_compatibility 1
#define GL_ARB_ES3_2_compatibility 1
#define GL_ARB_ES3_compatibility 1
#define GL_ARB_arrays_of_arrays 1
#define GL_ARB_base_instance 1
#define GL_ARB_bindless_texture 1
#define GL_ARB_blend_func_extended 1
#define GL_ARB_buffer_storage 1
#define GL_ARB_cl_event 1
#define GL_ARB_clear_buffer_object 1
#define GL_ARB_clear_texture 1
#define GL_ARB_clip_control 1
#define GL_ARB_color_buffer_float 1
#define GL_ARB_compatibility 1
#define GL_ARB_compressed_texture_pixel_storage 1
#define GL_ARB_compute_shader 1
#define GL_ARB_compute_variable_group_size 1
#define GL_ARB_conditional_render_inverted 1
#define GL_ARB_conservative_depth 1
#define GL_ARB_copy_buffer 1
#define GL_ARB_copy_image 1
#define GL_ARB_cull_distance 1
#define GL_ARB_debug_output 1
#define GL_ARB_depth_buffer_float 1
#define GL_ARB_depth_clamp 1
#define GL_ARB_depth_texture 1
#define GL_ARB_derivative_control 1
#define GL_ARB_direct_state_access 1
#define GL_ARB_draw_buffers 1
#define GL_ARB_draw_buffers_blend 1
#define GL_ARB_draw_elements_base_vertex 1
#define GL_ARB_draw_indirect 1
#define GL_ARB_draw_instanced 1
#define GL_ARB_enhanced_layouts 1
#define GL_ARB_explicit_attrib_location 1
#define GL_ARB_explicit_uniform_location 1
#define GL_ARB_fragment_coord_conventions 1
#define GL_ARB_fragment_layer_viewport 1
#define GL_ARB_fragment_program 1
#define GL_ARB_fragment_program_shadow 1
#define GL_ARB_fragment_shader 1
#define GL_ARB_fragment_shader_interlock 1
#define GL_ARB_framebuffer_no_attachments 1
#define GL_ARB_framebuffer_object 1
#define GL_ARB_framebuffer_sRGB 1
#define GL_ARB_geometry_shader4 1
#define GL_ARB_get_program_binary 1
#define GL_ARB_get_texture_sub_image 1
#define GL_ARB_gl_spirv 1
#define GL_ARB_gpu_shader5 1
#define GL_ARB_gpu_shader_fp64 1
#define GL_ARB_gpu_shader_int64 1
#define GL_ARB_half_float_pixel 1
#define GL_ARB_half_float_vertex 1
#define GL_ARB_imaging 1
#define GL_ARB_indirect_parameters 1
#define GL_ARB_instanced_arrays 1
#define GL_ARB_internalformat_query 1
#define GL_ARB_internalformat_query2 1
#define GL_ARB_invalidate_subdata 1
#define GL_ARB_map_buffer_alignment 1
#define GL_ARB_map_buffer_range 1
#define GL_ARB_matrix_palette 1
#define GL_ARB_multi_bind 1
#define GL_ARB_multi_draw_indirect 1
#define GL_ARB_multisample 1
#define GL_ARB_multitexture 1
#define GL_ARB_occlusion_query 1
#define GL_ARB_occlusion_query2 1
#define GL_ARB_parallel_shader_compile 1
#define GL_ARB_pipeline_statistics_query 1
#define GL_ARB_pixel_buffer_object 1
#define GL_ARB_point_parameters 1
#define GL_ARB_point_sprite 1
#define GL_ARB_polygon_offset_clamp 1
#define GL_ARB_post_depth_coverage 1
#define GL_ARB_program_interface_query 1
#define GL_ARB_provoking_vertex 1
#define GL_ARB_query_buffer_object 1
#define GL_ARB_robust_buffer_access_behavior 1
#define GL_ARB_robustness 1
#define GL_ARB_robustness_isolation 1
#define GL_ARB_sample_locations 1
#define GL_ARB_sample_shading 1
#define GL_ARB_sampler_objects 1
#define GL_ARB_seamless_cube_map 1
#define GL_ARB_seamless_cubemap_per_texture 1
#define GL_ARB_separate_shader_objects 1
#define GL_ARB_shader_atomic_counter_ops 1
#define GL_ARB_shader_atomic_counters 1
#define GL_ARB_shader_ballot 1
#define GL_ARB_shader_bit_encoding 1
#define GL_ARB_shader_clock 1
#define GL_ARB_shader_draw_parameters 1
#define GL_ARB_shader_group_vote 1
#define GL_ARB_shader_image_load_store 1
#define GL_ARB_shader_image_size 1
#define GL_ARB_shader_objects 1
#define GL_ARB_shader_precision 1
#define GL_ARB_shader_stencil_export 1
#define GL_ARB_shader_storage_buffer_object 1
#define GL_ARB_shader_subroutine 1
#define GL_ARB_shader_texture_image_samples 1
#define GL_ARB_shader_texture_lod 1
#define GL_ARB_shader_viewport_layer_array 1
#define GL_ARB_shading_language_100 1
#define GL_ARB_shading_language_420pack 1
#define GL_ARB_shading_language_include 1
#define GL_ARB_shading_language_packing 1
#define GL_ARB_shadow 1
#define GL_ARB_shadow_ambient 1
#define GL_ARB_sparse_buffer 1
#define GL_ARB_sparse_texture 1
#define GL_ARB_sparse_texture2 1
#define GL_ARB_sparse_texture_clamp 1
#define GL_ARB_spirv_extensions 1
#define GL_ARB_stencil_texturing 1
#define GL_ARB_sync 1
#define GL_ARB_tessellation_shader 1
#define GL_ARB_texture_barrier 1
#define GL_ARB_texture_border_clamp 1
#define GL_ARB_texture_buffer_object 1
#define GL_ARB_texture_buffer_object_rgb32 1
#define GL_ARB_texture_buffer_range 1
#define GL_ARB_texture_compression 1
#define GL_ARB_texture_compression_bptc 1
#define GL_ARB_texture_compression_rgtc 1
#define GL_ARB_texture_cube_map 1
#define GL_ARB_texture_cube_map_array 1
#define GL_ARB_texture_env_add 1
#define GL_ARB_texture_env_combine 1
#define GL_ARB_texture_env_crossbar 1
#define GL_ARB_texture_env_dot3 1
#define GL_ARB_texture_filter_anisotropic 1
#define GL_ARB_texture_filter_minmax 1
#define GL_ARB_texture_float 1
#define GL_ARB_texture_gather 1
#define GL_ARB_texture_mirror_clamp_to_edge 1
#define GL_ARB_texture_mirrored_repeat 1
#define GL_ARB_texture_multisample 1
#define GL_ARB_texture_non_power_of_two 1
#define GL_ARB_texture_query_levels 1
#define GL_ARB_texture_query_lod 1
#define GL_ARB_texture_rectangle 1
#define GL_ARB_texture_rg 1
#define GL_ARB_texture_rgb10_a2ui 1
#define GL_ARB_texture_stencil8 1
#define GL_ARB_texture_storage 1
#define GL_ARB_texture_storage_multisample 1
#define GL_ARB_texture_swizzle 1
#define GL_ARB_texture_view 1
#define GL_ARB_timer_query 1
#define GL_ARB_transform_feedback2 1
#define GL_ARB_transform_feedback3 1
#define GL_ARB_transform_feedback_instanced 1
#define GL_ARB_transform_feedback_overflow_query 1
#define GL_ARB_transpose_matrix 1
#define GL_ARB_uniform_buffer_object 1
#define GL_ARB_vertex_array_bgra 1
#define GL_ARB_vertex_array_object 1
#define GL_ARB_vertex_attrib_64bit 1
#define GL_ARB_vertex_attrib_binding 1
#define GL_ARB_vertex_blend 1
#define GL_ARB_vertex_buffer_object 1
#define GL_ARB_vertex_program 1
#define GL_ARB_vertex_shader 1
#define GL_ARB_vertex_type_10f_11f_11f_rev 1
#define GL_ARB_vertex_type_2_10_10_10_rev 1
#define GL_ARB_viewport_array 1
#define GL_ARB_window_pos 1
#define GL_EXT_422_pixels 1
#define GL_EXT_EGL_image_storage 1
#define GL_EXT_EGL_sync 1
#define GL_EXT_abgr 1
#define GL_EXT_bgra 1
#define GL_EXT_bindable_uniform 1
#define GL_EXT_blend_color 1
#define GL_EXT_blend_equation_separate 1
#define GL_EXT_blend_func_separate 1
#define GL_EXT_blend_logic_op 1
#define GL_EXT_blend_minmax 1
#define GL_EXT_blend_subtract 1
#define GL_EXT_clip_volume_hint 1
#define GL_EXT_cmyka 1
#define GL_EXT_color_subtable 1
#define GL_EXT_compiled_vertex_array 1
#define GL_EXT_convolution 1
#define GL_EXT_coordinate_frame 1
#define GL_EXT_copy_texture 1
#define GL_EXT_cull_vertex 1
#define GL_EXT_debug_label 1
#define GL_EXT_debug_marker 1
#define GL_EXT_depth_bounds_test 1
#define GL_EXT_direct_state_access 1
#define GL_EXT_draw_buffers2 1
#define GL_EXT_draw_instanced 1
#define GL_EXT_draw_range_elements 1
#define GL_EXT_external_buffer 1
#define GL_EXT_fog_coord 1
#define GL_EXT_framebuffer_blit 1
#define GL_EXT_framebuffer_multisample 1
#define GL_EXT_framebuffer_multisample_blit_scaled 1
#define GL_EXT_framebuffer_object 1
#define GL_EXT_framebuffer_sRGB 1
#define GL_EXT_geometry_shader4 1
#define GL_EXT_gpu_program_parameters 1
#define GL_EXT_gpu_shader4 1
#define GL_EXT_histogram 1
#define GL_EXT_index_array_formats 1
#define GL_EXT_index_func 1
#define GL_EXT_index_material 1
#define GL_EXT_index_texture 1
#define GL_EXT_light_texture 1
#define GL_EXT_memory_object 1
#define GL_EXT_memory_object_fd 1
#define GL_EXT_memory_object_win32 1
#define GL_EXT_misc_attribute 1
#define GL_EXT_multi_draw_arrays 1
#define GL_EXT_multisample 1
#define GL_EXT_multiview_tessellation_geometry_shader 1
#define GL_EXT_multiview_texture_multisample 1
#define GL_EXT_multiview_timer_query 1
#define GL_EXT_packed_depth_stencil 1
#define GL_EXT_packed_float 1
#define GL_EXT_packed_pixels 1
#define GL_EXT_paletted_texture 1
#define GL_EXT_pixel_buffer_object 1
#define GL_EXT_pixel_transform 1
#define GL_EXT_pixel_transform_color_table 1
#define GL_EXT_point_parameters 1
#define GL_EXT_polygon_offset 1
#define GL_EXT_polygon_offset_clamp 1
#define GL_EXT_post_depth_coverage 1
#define GL_EXT_provoking_vertex 1
#define GL_EXT_raster_multisample 1
#define GL_EXT_rescale_normal 1
#define GL_EXT_semaphore 1
#define GL_EXT_semaphore_fd 1
#define GL_EXT_semaphore_win32 1
#define GL_EXT_secondary_color 1
#define GL_EXT_separate_shader_objects 1
#define GL_EXT_separate_specular_color 1
#define GL_EXT_shader_framebuffer_fetch 1
#define GL_EXT_shader_framebuffer_fetch_non_coherent 1
#define GL_EXT_shader_image_load_formatted 1
#define GL_EXT_shader_image_load_store 1
#define GL_EXT_shader_integer_mix 1
#define GL_EXT_shadow_funcs 1
#define GL_EXT_shared_texture_palette 1
#define GL_EXT_sparse_texture2 1
#define GL_EXT_stencil_clear_tag 1
#define GL_EXT_stencil_two_side 1
#define GL_EXT_stencil_wrap 1
#define GL_EXT_subtexture 1
#define GL_EXT_texture 1
#define GL_EXT_texture3D 1
#define GL_EXT_texture_array 1
#define GL_EXT_texture_buffer_object 1
#define GL_EXT_texture_compression_latc 1
#define GL_EXT_texture_compression_rgtc 1
#define GL_EXT_texture_compression_s3tc 1
#define GL_EXT_texture_cube_map 1
#define GL_EXT_texture_env_add 1
#define GL_EXT_texture_env_combine 1
#define GL_EXT_texture_env_dot3 1
#define GL_EXT_texture_filter_anisotropic 1
#define GL_EXT_texture_filter_minmax 1
#define GL_EXT_texture_integer 1
#define GL_EXT_texture_lod_bias 1
#define GL_EXT_texture_mirror_clamp 1
#define GL_EXT_texture_object 1
#define GL_EXT_texture_perturb_normal 1
#define GL_EXT_texture_sRGB 1
#define GL_EXT_texture_sRGB_R8 1
#define GL_EXT_texture_sRGB_decode 1
#define GL_EXT_texture_shared_exponent 1
#define GL_EXT_texture_snorm 1
#define GL_EXT_texture_swizzle 1
#define GL_EXT_timer_query 1
#define GL_EXT_transform_feedback 1
#define GL_EXT_vertex_array 1
#define GL_EXT_vertex_array_bgra 1
#define GL_EXT_vertex_attrib_64bit 1
#define GL_EXT_vertex_shader 1
#define GL_EXT_vertex_weighting 1
#define GL_EXT_win32_keyed_mutex 1
#define GL_EXT_window_rectangles 1
#define GL_EXT_x11_sync_object 1
#define GL_INTEL_conservative_rasterization 1
#define GL_INTEL_fragment_shader_ordering 1
#define GL_INTEL_framebuffer_CMAA 1
#define GL_INTEL_map_texture 1
#define GL_INTEL_blackhole_render 1
#define GL_INTEL_parallel_arrays 1
#define GL_INTEL_performance_query 1
#define GL_KHR_blend_equation_advanced 1
#define GL_KHR_blend_equation_advanced_coherent 1
#define GL_KHR_context_flush_control 1
#define GL_KHR_debug 1
#define GL_KHR_no_error 1
#define GL_KHR_robust_buffer_access_behavior 1
#define GL_KHR_robustness 1
#define GL_KHR_shader_subgroup 1
#define GL_KHR_texture_compression_astc_hdr 1
#define GL_KHR_texture_compression_astc_ldr 1
#define GL_KHR_texture_compression_astc_sliced_3d 1
#define GL_KHR_parallel_shader_compile 1
#define GL_NV_alpha_to_coverage_dither_control 1
#define GL_NV_bindless_multi_draw_indirect 1
#define GL_NV_bindless_multi_draw_indirect_count 1
#define GL_NV_bindless_texture 1
#define GL_NV_blend_equation_advanced 1
#define GL_NV_blend_equation_advanced_coherent 1
#define GL_NV_blend_minmax_factor 1
#define GL_NV_blend_square 1
#define GL_NV_clip_space_w_scaling 1
#define GL_NV_command_list 1
#define GL_NV_compute_program5 1
#define GL_NV_compute_shader_derivatives 1
#define GL_NV_conditional_render 1
#define GL_NV_conservative_raster 1
#define GL_NV_conservative_raster_dilate 1
#define GL_NV_conservative_raster_pre_snap 1
#define GL_NV_conservative_raster_pre_snap_triangles 1
#define GL_NV_conservative_raster_underestimation 1
#define GL_NV_copy_depth_to_color 1
#define GL_NV_copy_image 1
#define GL_NV_deep_texture3D 1
#define GL_NV_depth_buffer_float 1
#define GL_NV_depth_clamp 1
#define GL_NV_draw_texture 1
#define GL_NV_draw_vulkan_image 1
#define GL_NV_evaluators 1
#define GL_NV_explicit_multisample 1
#define GL_NV_fence 1
#define GL_NV_fill_rectangle 1
#define GL_NV_float_buffer 1
#define GL_NV_fog_distance 1
#define GL_NV_fragment_coverage_to_color 1
#define GL_NV_fragment_program 1
#define GL_NV_fragment_program2 1
#define GL_NV_fragment_program4 1
#define GL_NV_fragment_program_option 1
#define GL_NV_fragment_shader_barycentric 1
#define GL_NV_fragment_shader_interlock 1
#define GL_NV_framebuffer_mixed_samples 1
#define GL_NV_framebuffer_multisample_coverage 1
#define GL_NV_geometry_program4 1
#define GL_NV_geometry_shader4 1
#define GL_NV_geometry_shader_passthrough 1
#define GL_NV_gpu_program4 1
#define GL_NV_gpu_program5 1
#define GL_NV_gpu_program5_mem_extended 1
#define GL_NV_gpu_shader5 1
#define GL_NV_half_float 1
#define GL_NV_internalformat_sample_query 1
#define GL_NV_light_max_exponent 1
#define GL_NV_gpu_multicast 1
#define GL_NV_memory_attachment 1
#define GL_NV_mesh_shader 1
#define GL_NV_multisample_coverage 1
#define GL_NV_multisample_filter_hint 1
#define GL_NV_occlusion_query 1
#define GL_NV_packed_depth_stencil 1
#define GL_NV_parameter_buffer_object 1
#define GL_NV_parameter_buffer_object2 1
#define GL_NV_path_rendering 1
#define GL_NV_path_rendering_shared_edge 1
#define GL_NV_pixel_data_range 1
#define GL_NV_point_sprite 1
#define GL_NV_present_video 1
#define GL_NV_primitive_restart 1
#define GL_NV_query_resource 1
#define GL_NV_query_resource_tag 1
#define GL_NV_register_combiners 1
#define GL_NV_register_combiners2 1
#define GL_NV_representative_fragment_test 1
#define GL_NV_robustness_video_memory_purge 1
#define GL_NV_sample_locations 1
#define GL_NV_sample_mask_override_coverage 1
#define GL_NV_scissor_exclusive 1
#define GL_NV_shader_atomic_counters 1
#define GL_NV_shader_atomic_float 1
#define GL_NV_shader_atomic_float64 1
#define GL_NV_shader_atomic_fp16_vector 1
#define GL_NV_shader_atomic_int64 1
#define GL_NV_shader_buffer_load 1
#define GL_NV_shader_buffer_store 1
#define GL_NV_shader_storage_buffer_object 1
#define GL_NV_shader_subgroup_partitioned 1
#define GL_NV_shader_texture_footprint 1
#define GL_NV_shader_thread_group 1
#define GL_NV_shader_thread_shuffle 1
#define GL_NV_shading_rate_image 1
#define GL_NV_stereo_view_rendering 1
#define GL_NV_tessellation_program5 1
#define GL_NV_texgen_emboss 1
#define GL_NV_texgen_reflection 1
#define GL_NV_texture_barrier 1
#define GL_NV_texture_compression_vtc 1
#define GL_NV_texture_env_combine4 1
#define GL_NV_texture_expand_normal 1
#define GL_NV_texture_multisample 1
#define GL_NV_texture_rectangle 1
#define GL_NV_texture_rectangle_compressed 1
#define GL_NV_texture_shader 1
#define GL_NV_texture_shader2 1
#define GL_NV_texture_shader3 1
#define GL_NV_transform_feedback 1
#define GL_NV_transform_feedback2 1
#define GL_NV_uniform_buffer_unified_memory 1
#define GL_NV_vdpau_interop 1
#define GL_NV_vdpau_interop2 1
#define GL_NV_vertex_array_range 1
#define GL_NV_vertex_array_range2 1
#define GL_NV_vertex_attrib_integer_64bit 1
#define GL_NV_vertex_buffer_unified_memory 1
#define GL_NV_vertex_program 1
#define GL_NV_vertex_program1_1 1
#define GL_NV_vertex_program2 1
#define GL_NV_vertex_program2_option 1
#define GL_NV_vertex_program3 1
#define GL_NV_vertex_program4 1
#define GL_NV_video_capture 1
#define GL_NV_viewport_array2 1
#define GL_NV_viewport_swizzle 1
#define GL_EXT_texture_shadow_lod 1


#define GL_CURRENT_BIT                                                0x00000001
#define GL_POINT_BIT                                                  0x00000002
#define GL_LINE_BIT                                                   0x00000004
#define GL_POLYGON_BIT                                                0x00000008
#define GL_POLYGON_STIPPLE_BIT                                        0x00000010
#define GL_PIXEL_MODE_BIT                                             0x00000020
#define GL_LIGHTING_BIT                                               0x00000040
#define GL_FOG_BIT                                                    0x00000080
#define GL_DEPTH_BUFFER_BIT                                           0x00000100
#define GL_ACCUM_BUFFER_BIT                                           0x00000200
#define GL_STENCIL_BUFFER_BIT                                         0x00000400
#define GL_VIEWPORT_BIT                                               0x00000800
#define GL_TRANSFORM_BIT                                              0x00001000
#define GL_ENABLE_BIT                                                 0x00002000
#define GL_COLOR_BUFFER_BIT                                           0x00004000
#define GL_HINT_BIT                                                   0x00008000
#define GL_EVAL_BIT                                                   0x00010000
#define GL_LIST_BIT                                                   0x00020000
#define GL_TEXTURE_BIT                                                0x00040000
#define GL_SCISSOR_BIT                                                0x00080000
#define GL_MULTISAMPLE_BIT                                            0x20000000
#define GL_MULTISAMPLE_BIT_ARB                                        0x20000000
#define GL_MULTISAMPLE_BIT_EXT                                        0x20000000
#define GL_ALL_ATTRIB_BITS                                            0xFFFFFFFF
#define GL_DYNAMIC_STORAGE_BIT                                        0x0100
#define GL_CLIENT_STORAGE_BIT                                         0x0200
#define GL_SPARSE_STORAGE_BIT_ARB                                     0x0400
#define GL_PER_GPU_STORAGE_BIT_NV                                     0x0800
#define GL_CLIENT_PIXEL_STORE_BIT                                     0x00000001
#define GL_CLIENT_VERTEX_ARRAY_BIT                                    0x00000002
#define GL_CLIENT_ALL_ATTRIB_BITS                                     0xFFFFFFFF
#define GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT                        0x00000001
#define GL_CONTEXT_FLAG_DEBUG_BIT                                     0x00000002
#define GL_CONTEXT_FLAG_DEBUG_BIT_KHR                                 0x00000002
#define GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT                             0x00000004
#define GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT_ARB                         0x00000004
#define GL_CONTEXT_FLAG_NO_ERROR_BIT                                  0x00000008
#define GL_CONTEXT_FLAG_NO_ERROR_BIT_KHR                              0x00000008
#define GL_CONTEXT_CORE_PROFILE_BIT                                   0x00000001
#define GL_CONTEXT_COMPATIBILITY_PROFILE_BIT                          0x00000002
#define GL_MAP_READ_BIT                                               0x0001
#define GL_MAP_WRITE_BIT                                              0x0002
#define GL_MAP_INVALIDATE_RANGE_BIT                                   0x0004
#define GL_MAP_INVALIDATE_BUFFER_BIT                                  0x0008
#define GL_MAP_FLUSH_EXPLICIT_BIT                                     0x0010
#define GL_MAP_UNSYNCHRONIZED_BIT                                     0x0020
#define GL_MAP_PERSISTENT_BIT                                         0x0040
#define GL_MAP_COHERENT_BIT                                           0x0080
#define GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT                            0x00000001
#define GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT_EXT                        0x00000001
#define GL_ELEMENT_ARRAY_BARRIER_BIT                                  0x00000002
#define GL_ELEMENT_ARRAY_BARRIER_BIT_EXT                              0x00000002
#define GL_UNIFORM_BARRIER_BIT                                        0x00000004
#define GL_UNIFORM_BARRIER_BIT_EXT                                    0x00000004
#define GL_TEXTURE_FETCH_BARRIER_BIT                                  0x00000008
#define GL_TEXTURE_FETCH_BARRIER_BIT_EXT                              0x00000008
#define GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV                        0x00000010
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT                            0x00000020
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT                        0x00000020
#define GL_COMMAND_BARRIER_BIT                                        0x00000040
#define GL_COMMAND_BARRIER_BIT_EXT                                    0x00000040
#define GL_PIXEL_BUFFER_BARRIER_BIT                                   0x00000080
#define GL_PIXEL_BUFFER_BARRIER_BIT_EXT                               0x00000080
#define GL_TEXTURE_UPDATE_BARRIER_BIT                                 0x00000100
#define GL_TEXTURE_UPDATE_BARRIER_BIT_EXT                             0x00000100
#define GL_BUFFER_UPDATE_BARRIER_BIT                                  0x00000200
#define GL_BUFFER_UPDATE_BARRIER_BIT_EXT                              0x00000200
#define GL_FRAMEBUFFER_BARRIER_BIT                                    0x00000400
#define GL_FRAMEBUFFER_BARRIER_BIT_EXT                                0x00000400
#define GL_TRANSFORM_FEEDBACK_BARRIER_BIT                             0x00000800
#define GL_TRANSFORM_FEEDBACK_BARRIER_BIT_EXT                         0x00000800
#define GL_ATOMIC_COUNTER_BARRIER_BIT                                 0x00001000
#define GL_ATOMIC_COUNTER_BARRIER_BIT_EXT                             0x00001000
#define GL_SHADER_STORAGE_BARRIER_BIT                                 0x00002000
#define GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT                           0x00004000
#define GL_QUERY_BUFFER_BARRIER_BIT                                   0x00008000
#define GL_ALL_BARRIER_BITS                                           0xFFFFFFFF
#define GL_ALL_BARRIER_BITS_EXT                                       0xFFFFFFFF
#define GL_QUERY_DEPTH_PASS_EVENT_BIT_AMD                             0x00000001
#define GL_QUERY_DEPTH_FAIL_EVENT_BIT_AMD                             0x00000002
#define GL_QUERY_STENCIL_FAIL_EVENT_BIT_AMD                           0x00000004
#define GL_QUERY_DEPTH_BOUNDS_FAIL_EVENT_BIT_AMD                      0x00000008
#define GL_QUERY_ALL_EVENT_BITS_AMD                                   0xFFFFFFFF
#define GL_SYNC_FLUSH_COMMANDS_BIT                                    0x00000001
#define GL_VERTEX_SHADER_BIT                                          0x00000001
#define GL_VERTEX_SHADER_BIT_EXT                                      0x00000001
#define GL_FRAGMENT_SHADER_BIT                                        0x00000002
#define GL_FRAGMENT_SHADER_BIT_EXT                                    0x00000002
#define GL_GEOMETRY_SHADER_BIT                                        0x00000004
#define GL_TESS_CONTROL_SHADER_BIT                                    0x00000008
#define GL_TESS_EVALUATION_SHADER_BIT                                 0x00000010
#define GL_COMPUTE_SHADER_BIT                                         0x00000020
#define GL_MESH_SHADER_BIT_NV                                         0x00000040
#define GL_TASK_SHADER_BIT_NV                                         0x00000080
#define GL_ALL_SHADER_BITS                                            0xFFFFFFFF
#define GL_ALL_SHADER_BITS_EXT                                        0xFFFFFFFF
#define GL_SUBGROUP_FEATURE_BASIC_BIT_KHR                             0x00000001
#define GL_SUBGROUP_FEATURE_VOTE_BIT_KHR                              0x00000002
#define GL_SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR                        0x00000004
#define GL_SUBGROUP_FEATURE_BALLOT_BIT_KHR                            0x00000008
#define GL_SUBGROUP_FEATURE_SHUFFLE_BIT_KHR                           0x00000010
#define GL_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR                  0x00000020
#define GL_SUBGROUP_FEATURE_CLUSTERED_BIT_KHR                         0x00000040
#define GL_SUBGROUP_FEATURE_QUAD_BIT_KHR                              0x00000080
#define GL_SUBGROUP_FEATURE_PARTITIONED_BIT_NV                        0x00000100
#define GL_TEXTURE_STORAGE_SPARSE_BIT_AMD                             0x00000001
#define GL_BOLD_BIT_NV                                                0x01
#define GL_ITALIC_BIT_NV                                              0x02
#define GL_GLYPH_WIDTH_BIT_NV                                         0x01
#define GL_GLYPH_HEIGHT_BIT_NV                                        0x02
#define GL_GLYPH_HORIZONTAL_BEARING_X_BIT_NV                          0x04
#define GL_GLYPH_HORIZONTAL_BEARING_Y_BIT_NV                          0x08
#define GL_GLYPH_HORIZONTAL_BEARING_ADVANCE_BIT_NV                    0x10
#define GL_GLYPH_VERTICAL_BEARING_X_BIT_NV                            0x20
#define GL_GLYPH_VERTICAL_BEARING_Y_BIT_NV                            0x40
#define GL_GLYPH_VERTICAL_BEARING_ADVANCE_BIT_NV                      0x80
#define GL_GLYPH_HAS_KERNING_BIT_NV                                   0x100
#define GL_FONT_X_MIN_BOUNDS_BIT_NV                                   0x00010000
#define GL_FONT_Y_MIN_BOUNDS_BIT_NV                                   0x00020000
#define GL_FONT_X_MAX_BOUNDS_BIT_NV                                   0x00040000
#define GL_FONT_Y_MAX_BOUNDS_BIT_NV                                   0x00080000
#define GL_FONT_UNITS_PER_EM_BIT_NV                                   0x00100000
#define GL_FONT_ASCENDER_BIT_NV                                       0x00200000
#define GL_FONT_DESCENDER_BIT_NV                                      0x00400000
#define GL_FONT_HEIGHT_BIT_NV                                         0x00800000
#define GL_FONT_MAX_ADVANCE_WIDTH_BIT_NV                              0x01000000
#define GL_FONT_MAX_ADVANCE_HEIGHT_BIT_NV                             0x02000000
#define GL_FONT_UNDERLINE_POSITION_BIT_NV                             0x04000000
#define GL_FONT_UNDERLINE_THICKNESS_BIT_NV                            0x08000000
#define GL_FONT_HAS_KERNING_BIT_NV                                    0x10000000
#define GL_FONT_NUM_GLYPH_INDICES_BIT_NV                              0x20000000
#define GL_PERFQUERY_SINGLE_CONTEXT_INTEL                             0x00000000
#define GL_PERFQUERY_GLOBAL_CONTEXT_INTEL                             0x00000001
#define GL_TERMINATE_SEQUENCE_COMMAND_NV                              0x0000
#define GL_NOP_COMMAND_NV                                             0x0001
#define GL_DRAW_ELEMENTS_COMMAND_NV                                   0x0002
#define GL_DRAW_ARRAYS_COMMAND_NV                                     0x0003
#define GL_DRAW_ELEMENTS_STRIP_COMMAND_NV                             0x0004
#define GL_DRAW_ARRAYS_STRIP_COMMAND_NV                               0x0005
#define GL_DRAW_ELEMENTS_INSTANCED_COMMAND_NV                         0x0006
#define GL_DRAW_ARRAYS_INSTANCED_COMMAND_NV                           0x0007
#define GL_ELEMENT_ADDRESS_COMMAND_NV                                 0x0008
#define GL_ATTRIBUTE_ADDRESS_COMMAND_NV                               0x0009
#define GL_UNIFORM_ADDRESS_COMMAND_NV                                 0x000A
#define GL_BLEND_COLOR_COMMAND_NV                                     0x000B
#define GL_STENCIL_REF_COMMAND_NV                                     0x000C
#define GL_LINE_WIDTH_COMMAND_NV                                      0x000D
#define GL_POLYGON_OFFSET_COMMAND_NV                                  0x000E
#define GL_ALPHA_REF_COMMAND_NV                                       0x000F
#define GL_VIEWPORT_COMMAND_NV                                        0x0010
#define GL_SCISSOR_COMMAND_NV                                         0x0011
#define GL_FRONT_FACE_COMMAND_NV                                      0x0012
#define GL_LAYOUT_DEFAULT_INTEL                                       0
#define GL_LAYOUT_LINEAR_INTEL                                        1
#define GL_LAYOUT_LINEAR_CPU_CACHED_INTEL                             2
#define GL_CLOSE_PATH_NV                                              0x00
#define GL_MOVE_TO_NV                                                 0x02
#define GL_RELATIVE_MOVE_TO_NV                                        0x03
#define GL_LINE_TO_NV                                                 0x04
#define GL_RELATIVE_LINE_TO_NV                                        0x05
#define GL_HORIZONTAL_LINE_TO_NV                                      0x06
#define GL_RELATIVE_HORIZONTAL_LINE_TO_NV                             0x07
#define GL_VERTICAL_LINE_TO_NV                                        0x08
#define GL_RELATIVE_VERTICAL_LINE_TO_NV                               0x09
#define GL_QUADRATIC_CURVE_TO_NV                                      0x0A
#define GL_RELATIVE_QUADRATIC_CURVE_TO_NV                             0x0B
#define GL_CUBIC_CURVE_TO_NV                                          0x0C
#define GL_RELATIVE_CUBIC_CURVE_TO_NV                                 0x0D
#define GL_SMOOTH_QUADRATIC_CURVE_TO_NV                               0x0E
#define GL_RELATIVE_SMOOTH_QUADRATIC_CURVE_TO_NV                      0x0F
#define GL_SMOOTH_CUBIC_CURVE_TO_NV                                   0x10
#define GL_RELATIVE_SMOOTH_CUBIC_CURVE_TO_NV                          0x11
#define GL_SMALL_CCW_ARC_TO_NV                                        0x12
#define GL_RELATIVE_SMALL_CCW_ARC_TO_NV                               0x13
#define GL_SMALL_CW_ARC_TO_NV                                         0x14
#define GL_RELATIVE_SMALL_CW_ARC_TO_NV                                0x15
#define GL_LARGE_CCW_ARC_TO_NV                                        0x16
#define GL_RELATIVE_LARGE_CCW_ARC_TO_NV                               0x17
#define GL_LARGE_CW_ARC_TO_NV                                         0x18
#define GL_RELATIVE_LARGE_CW_ARC_TO_NV                                0x19
#define GL_CONIC_CURVE_TO_NV                                          0x1A
#define GL_RELATIVE_CONIC_CURVE_TO_NV                                 0x1B
#define GL_SHARED_EDGE_NV                                             0xC0
#define GL_ROUNDED_RECT_NV                                            0xE8
#define GL_RELATIVE_ROUNDED_RECT_NV                                   0xE9
#define GL_ROUNDED_RECT2_NV                                           0xEA
#define GL_RELATIVE_ROUNDED_RECT2_NV                                  0xEB
#define GL_ROUNDED_RECT4_NV                                           0xEC
#define GL_RELATIVE_ROUNDED_RECT4_NV                                  0xED
#define GL_ROUNDED_RECT8_NV                                           0xEE
#define GL_RELATIVE_ROUNDED_RECT8_NV                                  0xEF
#define GL_RESTART_PATH_NV                                            0xF0
#define GL_DUP_FIRST_CUBIC_CURVE_TO_NV                                0xF2
#define GL_DUP_LAST_CUBIC_CURVE_TO_NV                                 0xF4
#define GL_RECT_NV                                                    0xF6
#define GL_RELATIVE_RECT_NV                                           0xF7
#define GL_CIRCULAR_CCW_ARC_TO_NV                                     0xF8
#define GL_CIRCULAR_CW_ARC_TO_NV                                      0xFA
#define GL_CIRCULAR_TANGENT_ARC_TO_NV                                 0xFC
#define GL_ARC_TO_NV                                                  0xFE
#define GL_RELATIVE_ARC_TO_NV                                         0xFF
#define GL_NEXT_BUFFER_NV                                             -2
#define GL_SKIP_COMPONENTS4_NV                                        -3
#define GL_SKIP_COMPONENTS3_NV                                        -4
#define GL_SKIP_COMPONENTS2_NV                                        -5
#define GL_SKIP_COMPONENTS1_NV                                        -6
#define GL_FALSE                                                      0
#define GL_NO_ERROR                                                   0
#define GL_ZERO                                                       0
#define GL_NONE                                                       0
#define GL_TRUE                                                       1
#define GL_ONE                                                        1
#define GL_INVALID_INDEX                                              0xFFFFFFFF
#define GL_ALL_PIXELS_AMD                                             0xFFFFFFFF
#define GL_TIMEOUT_IGNORED                                            0xFFFFFFFFFFFFFFFF
#define GL_UUID_SIZE_EXT                                              16
#define GL_LUID_SIZE_EXT                                              8
#define GL_POINTS                                                     0x0000
#define GL_LINES                                                      0x0001
#define GL_LINE_LOOP                                                  0x0002
#define GL_LINE_STRIP                                                 0x0003
#define GL_TRIANGLES                                                  0x0004
#define GL_TRIANGLE_STRIP                                             0x0005
#define GL_TRIANGLE_FAN                                               0x0006
#define GL_QUADS                                                      0x0007
#define GL_QUAD_STRIP                                                 0x0008
#define GL_POLYGON                                                    0x0009
#define GL_LINES_ADJACENCY                                            0x000A
#define GL_LINES_ADJACENCY_ARB                                        0x000A
#define GL_LINES_ADJACENCY_EXT                                        0x000A
#define GL_LINE_STRIP_ADJACENCY                                       0x000B
#define GL_LINE_STRIP_ADJACENCY_ARB                                   0x000B
#define GL_LINE_STRIP_ADJACENCY_EXT                                   0x000B
#define GL_TRIANGLES_ADJACENCY                                        0x000C
#define GL_TRIANGLES_ADJACENCY_ARB                                    0x000C
#define GL_TRIANGLES_ADJACENCY_EXT                                    0x000C
#define GL_TRIANGLE_STRIP_ADJACENCY                                   0x000D
#define GL_TRIANGLE_STRIP_ADJACENCY_ARB                               0x000D
#define GL_TRIANGLE_STRIP_ADJACENCY_EXT                               0x000D
#define GL_PATCHES                                                    0x000E
#define GL_ACCUM                                                      0x0100
#define GL_LOAD                                                       0x0101
#define GL_RETURN                                                     0x0102
#define GL_MULT                                                       0x0103
#define GL_ADD                                                        0x0104
#define GL_NEVER                                                      0x0200
#define GL_LESS                                                       0x0201
#define GL_EQUAL                                                      0x0202
#define GL_LEQUAL                                                     0x0203
#define GL_GREATER                                                    0x0204
#define GL_NOTEQUAL                                                   0x0205
#define GL_GEQUAL                                                     0x0206
#define GL_ALWAYS                                                     0x0207
#define GL_SRC_COLOR                                                  0x0300
#define GL_ONE_MINUS_SRC_COLOR                                        0x0301
#define GL_SRC_ALPHA                                                  0x0302
#define GL_ONE_MINUS_SRC_ALPHA                                        0x0303
#define GL_DST_ALPHA                                                  0x0304
#define GL_ONE_MINUS_DST_ALPHA                                        0x0305
#define GL_DST_COLOR                                                  0x0306
#define GL_ONE_MINUS_DST_COLOR                                        0x0307
#define GL_SRC_ALPHA_SATURATE                                         0x0308
#define GL_FRONT_LEFT                                                 0x0400
#define GL_FRONT_RIGHT                                                0x0401
#define GL_BACK_LEFT                                                  0x0402
#define GL_BACK_RIGHT                                                 0x0403
#define GL_FRONT                                                      0x0404
#define GL_BACK                                                       0x0405
#define GL_LEFT                                                       0x0406
#define GL_RIGHT                                                      0x0407
#define GL_FRONT_AND_BACK                                             0x0408
#define GL_AUX0                                                       0x0409
#define GL_AUX1                                                       0x040A
#define GL_AUX2                                                       0x040B
#define GL_AUX3                                                       0x040C
#define GL_INVALID_ENUM                                               0x0500
#define GL_INVALID_VALUE                                              0x0501
#define GL_INVALID_OPERATION                                          0x0502
#define GL_STACK_OVERFLOW                                             0x0503
#define GL_STACK_OVERFLOW_KHR                                         0x0503
#define GL_STACK_UNDERFLOW                                            0x0504
#define GL_STACK_UNDERFLOW_KHR                                        0x0504
#define GL_OUT_OF_MEMORY                                              0x0505
#define GL_INVALID_FRAMEBUFFER_OPERATION                              0x0506
#define GL_INVALID_FRAMEBUFFER_OPERATION_EXT                          0x0506
#define GL_CONTEXT_LOST                                               0x0507
#define GL_CONTEXT_LOST_KHR                                           0x0507
#define GL_2D                                                         0x0600
#define GL_3D                                                         0x0601
#define GL_3D_COLOR                                                   0x0602
#define GL_3D_COLOR_TEXTURE                                           0x0603
#define GL_4D_COLOR_TEXTURE                                           0x0604
#define GL_PASS_THROUGH_TOKEN                                         0x0700
#define GL_POINT_TOKEN                                                0x0701
#define GL_LINE_TOKEN                                                 0x0702
#define GL_POLYGON_TOKEN                                              0x0703
#define GL_BITMAP_TOKEN                                               0x0704
#define GL_DRAW_PIXEL_TOKEN                                           0x0705
#define GL_COPY_PIXEL_TOKEN                                           0x0706
#define GL_LINE_RESET_TOKEN                                           0x0707
#define GL_EXP                                                        0x0800
#define GL_EXP2                                                       0x0801
#define GL_CW                                                         0x0900
#define GL_CCW                                                        0x0901
#define GL_COEFF                                                      0x0A00
#define GL_ORDER                                                      0x0A01
#define GL_DOMAIN                                                     0x0A02
#define GL_CURRENT_COLOR                                              0x0B00
#define GL_CURRENT_INDEX                                              0x0B01
#define GL_CURRENT_NORMAL                                             0x0B02
#define GL_CURRENT_TEXTURE_COORDS                                     0x0B03
#define GL_CURRENT_RASTER_COLOR                                       0x0B04
#define GL_CURRENT_RASTER_INDEX                                       0x0B05
#define GL_CURRENT_RASTER_TEXTURE_COORDS                              0x0B06
#define GL_CURRENT_RASTER_POSITION                                    0x0B07
#define GL_CURRENT_RASTER_POSITION_VALID                              0x0B08
#define GL_CURRENT_RASTER_DISTANCE                                    0x0B09
#define GL_POINT_SMOOTH                                               0x0B10
#define GL_POINT_SIZE                                                 0x0B11
#define GL_POINT_SIZE_RANGE                                           0x0B12
#define GL_SMOOTH_POINT_SIZE_RANGE                                    0x0B12
#define GL_POINT_SIZE_GRANULARITY                                     0x0B13
#define GL_SMOOTH_POINT_SIZE_GRANULARITY                              0x0B13
#define GL_LINE_SMOOTH                                                0x0B20
#define GL_LINE_WIDTH                                                 0x0B21
#define GL_LINE_WIDTH_RANGE                                           0x0B22
#define GL_SMOOTH_LINE_WIDTH_RANGE                                    0x0B22
#define GL_LINE_WIDTH_GRANULARITY                                     0x0B23
#define GL_SMOOTH_LINE_WIDTH_GRANULARITY                              0x0B23
#define GL_LINE_STIPPLE                                               0x0B24
#define GL_LINE_STIPPLE_PATTERN                                       0x0B25
#define GL_LINE_STIPPLE_REPEAT                                        0x0B26
#define GL_LIST_MODE                                                  0x0B30
#define GL_MAX_LIST_NESTING                                           0x0B31
#define GL_LIST_BASE                                                  0x0B32
#define GL_LIST_INDEX                                                 0x0B33
#define GL_POLYGON_MODE                                               0x0B40
#define GL_POLYGON_SMOOTH                                             0x0B41
#define GL_POLYGON_STIPPLE                                            0x0B42
#define GL_EDGE_FLAG                                                  0x0B43
#define GL_CULL_FACE                                                  0x0B44
#define GL_CULL_FACE_MODE                                             0x0B45
#define GL_FRONT_FACE                                                 0x0B46
#define GL_LIGHTING                                                   0x0B50
#define GL_LIGHT_MODEL_LOCAL_VIEWER                                   0x0B51
#define GL_LIGHT_MODEL_TWO_SIDE                                       0x0B52
#define GL_LIGHT_MODEL_AMBIENT                                        0x0B53
#define GL_SHADE_MODEL                                                0x0B54
#define GL_COLOR_MATERIAL_FACE                                        0x0B55
#define GL_COLOR_MATERIAL_PARAMETER                                   0x0B56
#define GL_COLOR_MATERIAL                                             0x0B57
#define GL_FOG                                                        0x0B60
#define GL_FOG_INDEX                                                  0x0B61
#define GL_FOG_DENSITY                                                0x0B62
#define GL_FOG_START                                                  0x0B63
#define GL_FOG_END                                                    0x0B64
#define GL_FOG_MODE                                                   0x0B65
#define GL_FOG_COLOR                                                  0x0B66
#define GL_DEPTH_RANGE                                                0x0B70
#define GL_DEPTH_TEST                                                 0x0B71
#define GL_DEPTH_WRITEMASK                                            0x0B72
#define GL_DEPTH_CLEAR_VALUE                                          0x0B73
#define GL_DEPTH_FUNC                                                 0x0B74
#define GL_ACCUM_CLEAR_VALUE                                          0x0B80
#define GL_STENCIL_TEST                                               0x0B90
#define GL_STENCIL_CLEAR_VALUE                                        0x0B91
#define GL_STENCIL_FUNC                                               0x0B92
#define GL_STENCIL_VALUE_MASK                                         0x0B93
#define GL_STENCIL_FAIL                                               0x0B94
#define GL_STENCIL_PASS_DEPTH_FAIL                                    0x0B95
#define GL_STENCIL_PASS_DEPTH_PASS                                    0x0B96
#define GL_STENCIL_REF                                                0x0B97
#define GL_STENCIL_WRITEMASK                                          0x0B98
#define GL_MATRIX_MODE                                                0x0BA0
#define GL_NORMALIZE                                                  0x0BA1
#define GL_VIEWPORT                                                   0x0BA2
#define GL_MODELVIEW_STACK_DEPTH                                      0x0BA3
#define GL_MODELVIEW0_STACK_DEPTH_EXT                                 0x0BA3
#define GL_PATH_MODELVIEW_STACK_DEPTH_NV                              0x0BA3
#define GL_PROJECTION_STACK_DEPTH                                     0x0BA4
#define GL_PATH_PROJECTION_STACK_DEPTH_NV                             0x0BA4
#define GL_TEXTURE_STACK_DEPTH                                        0x0BA5
#define GL_MODELVIEW_MATRIX                                           0x0BA6
#define GL_MODELVIEW0_MATRIX_EXT                                      0x0BA6
#define GL_PATH_MODELVIEW_MATRIX_NV                                   0x0BA6
#define GL_PROJECTION_MATRIX                                          0x0BA7
#define GL_PATH_PROJECTION_MATRIX_NV                                  0x0BA7
#define GL_TEXTURE_MATRIX                                             0x0BA8
#define GL_ATTRIB_STACK_DEPTH                                         0x0BB0
#define GL_CLIENT_ATTRIB_STACK_DEPTH                                  0x0BB1
#define GL_ALPHA_TEST                                                 0x0BC0
#define GL_ALPHA_TEST_FUNC                                            0x0BC1
#define GL_ALPHA_TEST_REF                                             0x0BC2
#define GL_DITHER                                                     0x0BD0
#define GL_BLEND_DST                                                  0x0BE0
#define GL_BLEND_SRC                                                  0x0BE1
#define GL_BLEND                                                      0x0BE2
#define GL_LOGIC_OP_MODE                                              0x0BF0
#define GL_INDEX_LOGIC_OP                                             0x0BF1
#define GL_LOGIC_OP                                                   0x0BF1
#define GL_COLOR_LOGIC_OP                                             0x0BF2
#define GL_AUX_BUFFERS                                                0x0C00
#define GL_DRAW_BUFFER                                                0x0C01
#define GL_READ_BUFFER                                                0x0C02
#define GL_SCISSOR_BOX                                                0x0C10
#define GL_SCISSOR_TEST                                               0x0C11
#define GL_INDEX_CLEAR_VALUE                                          0x0C20
#define GL_INDEX_WRITEMASK                                            0x0C21
#define GL_COLOR_CLEAR_VALUE                                          0x0C22
#define GL_COLOR_WRITEMASK                                            0x0C23
#define GL_INDEX_MODE                                                 0x0C30
#define GL_RGBA_MODE                                                  0x0C31
#define GL_DOUBLEBUFFER                                               0x0C32
#define GL_STEREO                                                     0x0C33
#define GL_RENDER_MODE                                                0x0C40
#define GL_PERSPECTIVE_CORRECTION_HINT                                0x0C50
#define GL_POINT_SMOOTH_HINT                                          0x0C51
#define GL_LINE_SMOOTH_HINT                                           0x0C52
#define GL_POLYGON_SMOOTH_HINT                                        0x0C53
#define GL_FOG_HINT                                                   0x0C54
#define GL_TEXTURE_GEN_S                                              0x0C60
#define GL_TEXTURE_GEN_T                                              0x0C61
#define GL_TEXTURE_GEN_R                                              0x0C62
#define GL_TEXTURE_GEN_Q                                              0x0C63
#define GL_PIXEL_MAP_I_TO_I                                           0x0C70
#define GL_PIXEL_MAP_S_TO_S                                           0x0C71
#define GL_PIXEL_MAP_I_TO_R                                           0x0C72
#define GL_PIXEL_MAP_I_TO_G                                           0x0C73
#define GL_PIXEL_MAP_I_TO_B                                           0x0C74
#define GL_PIXEL_MAP_I_TO_A                                           0x0C75
#define GL_PIXEL_MAP_R_TO_R                                           0x0C76
#define GL_PIXEL_MAP_G_TO_G                                           0x0C77
#define GL_PIXEL_MAP_B_TO_B                                           0x0C78
#define GL_PIXEL_MAP_A_TO_A                                           0x0C79
#define GL_PIXEL_MAP_I_TO_I_SIZE                                      0x0CB0
#define GL_PIXEL_MAP_S_TO_S_SIZE                                      0x0CB1
#define GL_PIXEL_MAP_I_TO_R_SIZE                                      0x0CB2
#define GL_PIXEL_MAP_I_TO_G_SIZE                                      0x0CB3
#define GL_PIXEL_MAP_I_TO_B_SIZE                                      0x0CB4
#define GL_PIXEL_MAP_I_TO_A_SIZE                                      0x0CB5
#define GL_PIXEL_MAP_R_TO_R_SIZE                                      0x0CB6
#define GL_PIXEL_MAP_G_TO_G_SIZE                                      0x0CB7
#define GL_PIXEL_MAP_B_TO_B_SIZE                                      0x0CB8
#define GL_PIXEL_MAP_A_TO_A_SIZE                                      0x0CB9
#define GL_UNPACK_SWAP_BYTES                                          0x0CF0
#define GL_UNPACK_LSB_FIRST                                           0x0CF1
#define GL_UNPACK_ROW_LENGTH                                          0x0CF2
#define GL_UNPACK_SKIP_ROWS                                           0x0CF3
#define GL_UNPACK_SKIP_PIXELS                                         0x0CF4
#define GL_UNPACK_ALIGNMENT                                           0x0CF5
#define GL_PACK_SWAP_BYTES                                            0x0D00
#define GL_PACK_LSB_FIRST                                             0x0D01
#define GL_PACK_ROW_LENGTH                                            0x0D02
#define GL_PACK_SKIP_ROWS                                             0x0D03
#define GL_PACK_SKIP_PIXELS                                           0x0D04
#define GL_PACK_ALIGNMENT                                             0x0D05
#define GL_MAP_COLOR                                                  0x0D10
#define GL_MAP_STENCIL                                                0x0D11
#define GL_INDEX_SHIFT                                                0x0D12
#define GL_INDEX_OFFSET                                               0x0D13
#define GL_RED_SCALE                                                  0x0D14
#define GL_RED_BIAS                                                   0x0D15
#define GL_ZOOM_X                                                     0x0D16
#define GL_ZOOM_Y                                                     0x0D17
#define GL_GREEN_SCALE                                                0x0D18
#define GL_GREEN_BIAS                                                 0x0D19
#define GL_BLUE_SCALE                                                 0x0D1A
#define GL_BLUE_BIAS                                                  0x0D1B
#define GL_ALPHA_SCALE                                                0x0D1C
#define GL_ALPHA_BIAS                                                 0x0D1D
#define GL_DEPTH_SCALE                                                0x0D1E
#define GL_DEPTH_BIAS                                                 0x0D1F
#define GL_MAX_EVAL_ORDER                                             0x0D30
#define GL_MAX_LIGHTS                                                 0x0D31
#define GL_MAX_CLIP_PLANES                                            0x0D32
#define GL_MAX_CLIP_DISTANCES                                         0x0D32
#define GL_MAX_TEXTURE_SIZE                                           0x0D33
#define GL_MAX_PIXEL_MAP_TABLE                                        0x0D34
#define GL_MAX_ATTRIB_STACK_DEPTH                                     0x0D35
#define GL_MAX_MODELVIEW_STACK_DEPTH                                  0x0D36
#define GL_PATH_MAX_MODELVIEW_STACK_DEPTH_NV                          0x0D36
#define GL_MAX_NAME_STACK_DEPTH                                       0x0D37
#define GL_MAX_PROJECTION_STACK_DEPTH                                 0x0D38
#define GL_PATH_MAX_PROJECTION_STACK_DEPTH_NV                         0x0D38
#define GL_MAX_TEXTURE_STACK_DEPTH                                    0x0D39
#define GL_MAX_VIEWPORT_DIMS                                          0x0D3A
#define GL_MAX_CLIENT_ATTRIB_STACK_DEPTH                              0x0D3B
#define GL_SUBPIXEL_BITS                                              0x0D50
#define GL_INDEX_BITS                                                 0x0D51
#define GL_RED_BITS                                                   0x0D52
#define GL_GREEN_BITS                                                 0x0D53
#define GL_BLUE_BITS                                                  0x0D54
#define GL_ALPHA_BITS                                                 0x0D55
#define GL_DEPTH_BITS                                                 0x0D56
#define GL_STENCIL_BITS                                               0x0D57
#define GL_ACCUM_RED_BITS                                             0x0D58
#define GL_ACCUM_GREEN_BITS                                           0x0D59
#define GL_ACCUM_BLUE_BITS                                            0x0D5A
#define GL_ACCUM_ALPHA_BITS                                           0x0D5B
#define GL_NAME_STACK_DEPTH                                           0x0D70
#define GL_AUTO_NORMAL                                                0x0D80
#define GL_MAP1_COLOR_4                                               0x0D90
#define GL_MAP1_INDEX                                                 0x0D91
#define GL_MAP1_NORMAL                                                0x0D92
#define GL_MAP1_TEXTURE_COORD_1                                       0x0D93
#define GL_MAP1_TEXTURE_COORD_2                                       0x0D94
#define GL_MAP1_TEXTURE_COORD_3                                       0x0D95
#define GL_MAP1_TEXTURE_COORD_4                                       0x0D96
#define GL_MAP1_VERTEX_3                                              0x0D97
#define GL_MAP1_VERTEX_4                                              0x0D98
#define GL_MAP2_COLOR_4                                               0x0DB0
#define GL_MAP2_INDEX                                                 0x0DB1
#define GL_MAP2_NORMAL                                                0x0DB2
#define GL_MAP2_TEXTURE_COORD_1                                       0x0DB3
#define GL_MAP2_TEXTURE_COORD_2                                       0x0DB4
#define GL_MAP2_TEXTURE_COORD_3                                       0x0DB5
#define GL_MAP2_TEXTURE_COORD_4                                       0x0DB6
#define GL_MAP2_VERTEX_3                                              0x0DB7
#define GL_MAP2_VERTEX_4                                              0x0DB8
#define GL_MAP1_GRID_DOMAIN                                           0x0DD0
#define GL_MAP1_GRID_SEGMENTS                                         0x0DD1
#define GL_MAP2_GRID_DOMAIN                                           0x0DD2
#define GL_MAP2_GRID_SEGMENTS                                         0x0DD3
#define GL_TEXTURE_1D                                                 0x0DE0
#define GL_TEXTURE_2D                                                 0x0DE1
#define GL_FEEDBACK_BUFFER_POINTER                                    0x0DF0
#define GL_FEEDBACK_BUFFER_SIZE                                       0x0DF1
#define GL_FEEDBACK_BUFFER_TYPE                                       0x0DF2
#define GL_SELECTION_BUFFER_POINTER                                   0x0DF3
#define GL_SELECTION_BUFFER_SIZE                                      0x0DF4
#define GL_TEXTURE_WIDTH                                              0x1000
#define GL_TEXTURE_HEIGHT                                             0x1001
#define GL_TEXTURE_INTERNAL_FORMAT                                    0x1003
#define GL_TEXTURE_COMPONENTS                                         0x1003
#define GL_TEXTURE_BORDER_COLOR                                       0x1004
#define GL_TEXTURE_BORDER                                             0x1005
#define GL_TEXTURE_TARGET                                             0x1006
#define GL_DONT_CARE                                                  0x1100
#define GL_FASTEST                                                    0x1101
#define GL_NICEST                                                     0x1102
#define GL_AMBIENT                                                    0x1200
#define GL_DIFFUSE                                                    0x1201
#define GL_SPECULAR                                                   0x1202
#define GL_POSITION                                                   0x1203
#define GL_SPOT_DIRECTION                                             0x1204
#define GL_SPOT_EXPONENT                                              0x1205
#define GL_SPOT_CUTOFF                                                0x1206
#define GL_CONSTANT_ATTENUATION                                       0x1207
#define GL_LINEAR_ATTENUATION                                         0x1208
#define GL_QUADRATIC_ATTENUATION                                      0x1209
#define GL_COMPILE                                                    0x1300
#define GL_COMPILE_AND_EXECUTE                                        0x1301
#define GL_BYTE                                                       0x1400
#define GL_UNSIGNED_BYTE                                              0x1401
#define GL_SHORT                                                      0x1402
#define GL_UNSIGNED_SHORT                                             0x1403
#define GL_INT                                                        0x1404
#define GL_UNSIGNED_INT                                               0x1405
#define GL_FLOAT                                                      0x1406
#define GL_2_BYTES                                                    0x1407
#define GL_2_BYTES_NV                                                 0x1407
#define GL_3_BYTES                                                    0x1408
#define GL_3_BYTES_NV                                                 0x1408
#define GL_4_BYTES                                                    0x1409
#define GL_4_BYTES_NV                                                 0x1409
#define GL_DOUBLE                                                     0x140A
#define GL_HALF_FLOAT                                                 0x140B
#define GL_HALF_FLOAT_ARB                                             0x140B
#define GL_HALF_FLOAT_NV                                              0x140B
#define GL_HALF_APPLE                                                 0x140B
#define GL_FIXED                                                      0x140C
#define GL_INT64_ARB                                                  0x140E
#define GL_INT64_NV                                                   0x140E
#define GL_UNSIGNED_INT64_ARB                                         0x140F
#define GL_UNSIGNED_INT64_NV                                          0x140F
#define GL_CLEAR                                                      0x1500
#define GL_AND                                                        0x1501
#define GL_AND_REVERSE                                                0x1502
#define GL_COPY                                                       0x1503
#define GL_AND_INVERTED                                               0x1504
#define GL_NOOP                                                       0x1505
#define GL_XOR                                                        0x1506
#define GL_XOR_NV                                                     0x1506
#define GL_OR                                                         0x1507
#define GL_NOR                                                        0x1508
#define GL_EQUIV                                                      0x1509
#define GL_INVERT                                                     0x150A
#define GL_OR_REVERSE                                                 0x150B
#define GL_COPY_INVERTED                                              0x150C
#define GL_OR_INVERTED                                                0x150D
#define GL_NAND                                                       0x150E
#define GL_SET                                                        0x150F
#define GL_EMISSION                                                   0x1600
#define GL_SHININESS                                                  0x1601
#define GL_AMBIENT_AND_DIFFUSE                                        0x1602
#define GL_COLOR_INDEXES                                              0x1603
#define GL_MODELVIEW                                                  0x1700
#define GL_MODELVIEW0_ARB                                             0x1700
#define GL_MODELVIEW0_EXT                                             0x1700
#define GL_PATH_MODELVIEW_NV                                          0x1700
#define GL_PROJECTION                                                 0x1701
#define GL_PATH_PROJECTION_NV                                         0x1701
#define GL_TEXTURE                                                    0x1702
#define GL_COLOR                                                      0x1800
#define GL_DEPTH                                                      0x1801
#define GL_STENCIL                                                    0x1802
#define GL_COLOR_INDEX                                                0x1900
#define GL_STENCIL_INDEX                                              0x1901
#define GL_DEPTH_COMPONENT                                            0x1902
#define GL_RED                                                        0x1903
#define GL_RED_NV                                                     0x1903
#define GL_GREEN                                                      0x1904
#define GL_GREEN_NV                                                   0x1904
#define GL_BLUE                                                       0x1905
#define GL_BLUE_NV                                                    0x1905
#define GL_ALPHA                                                      0x1906
#define GL_RGB                                                        0x1907
#define GL_RGBA                                                       0x1908
#define GL_LUMINANCE                                                  0x1909
#define GL_LUMINANCE_ALPHA                                            0x190A
#define GL_BITMAP                                                     0x1A00
#define GL_POINT                                                      0x1B00
#define GL_LINE                                                       0x1B01
#define GL_FILL                                                       0x1B02
#define GL_RENDER                                                     0x1C00
#define GL_FEEDBACK                                                   0x1C01
#define GL_SELECT                                                     0x1C02
#define GL_FLAT                                                       0x1D00
#define GL_SMOOTH                                                     0x1D01
#define GL_KEEP                                                       0x1E00
#define GL_REPLACE                                                    0x1E01
#define GL_INCR                                                       0x1E02
#define GL_DECR                                                       0x1E03
#define GL_VENDOR                                                     0x1F00
#define GL_RENDERER                                                   0x1F01
#define GL_VERSION                                                    0x1F02
#define GL_EXTENSIONS                                                 0x1F03
#define GL_S                                                          0x2000
#define GL_T                                                          0x2001
#define GL_R                                                          0x2002
#define GL_Q                                                          0x2003
#define GL_MODULATE                                                   0x2100
#define GL_DECAL                                                      0x2101
#define GL_TEXTURE_ENV_MODE                                           0x2200
#define GL_TEXTURE_ENV_COLOR                                          0x2201
#define GL_TEXTURE_ENV                                                0x2300
#define GL_EYE_LINEAR                                                 0x2400
#define GL_EYE_LINEAR_NV                                              0x2400
#define GL_OBJECT_LINEAR                                              0x2401
#define GL_OBJECT_LINEAR_NV                                           0x2401
#define GL_SPHERE_MAP                                                 0x2402
#define GL_TEXTURE_GEN_MODE                                           0x2500
#define GL_OBJECT_PLANE                                               0x2501
#define GL_EYE_PLANE                                                  0x2502
#define GL_NEAREST                                                    0x2600
#define GL_LINEAR                                                     0x2601
#define GL_NEAREST_MIPMAP_NEAREST                                     0x2700
#define GL_LINEAR_MIPMAP_NEAREST                                      0x2701
#define GL_NEAREST_MIPMAP_LINEAR                                      0x2702
#define GL_LINEAR_MIPMAP_LINEAR                                       0x2703
#define GL_TEXTURE_MAG_FILTER                                         0x2800
#define GL_TEXTURE_MIN_FILTER                                         0x2801
#define GL_TEXTURE_WRAP_S                                             0x2802
#define GL_TEXTURE_WRAP_T                                             0x2803
#define GL_CLAMP                                                      0x2900
#define GL_REPEAT                                                     0x2901
#define GL_POLYGON_OFFSET_UNITS                                       0x2A00
#define GL_POLYGON_OFFSET_POINT                                       0x2A01
#define GL_POLYGON_OFFSET_LINE                                        0x2A02
#define GL_R3_G3_B2                                                   0x2A10
#define GL_V2F                                                        0x2A20
#define GL_V3F                                                        0x2A21
#define GL_C4UB_V2F                                                   0x2A22
#define GL_C4UB_V3F                                                   0x2A23
#define GL_C3F_V3F                                                    0x2A24
#define GL_N3F_V3F                                                    0x2A25
#define GL_C4F_N3F_V3F                                                0x2A26
#define GL_T2F_V3F                                                    0x2A27
#define GL_T4F_V4F                                                    0x2A28
#define GL_T2F_C4UB_V3F                                               0x2A29
#define GL_T2F_C3F_V3F                                                0x2A2A
#define GL_T2F_N3F_V3F                                                0x2A2B
#define GL_T2F_C4F_N3F_V3F                                            0x2A2C
#define GL_T4F_C4F_N3F_V4F                                            0x2A2D
#define GL_CLIP_PLANE0                                                0x3000
#define GL_CLIP_DISTANCE0                                             0x3000
#define GL_CLIP_PLANE1                                                0x3001
#define GL_CLIP_DISTANCE1                                             0x3001
#define GL_CLIP_PLANE2                                                0x3002
#define GL_CLIP_DISTANCE2                                             0x3002
#define GL_CLIP_PLANE3                                                0x3003
#define GL_CLIP_DISTANCE3                                             0x3003
#define GL_CLIP_PLANE4                                                0x3004
#define GL_CLIP_DISTANCE4                                             0x3004
#define GL_CLIP_PLANE5                                                0x3005
#define GL_CLIP_DISTANCE5                                             0x3005
#define GL_CLIP_DISTANCE6                                             0x3006
#define GL_CLIP_DISTANCE7                                             0x3007
#define GL_LIGHT0                                                     0x4000
#define GL_LIGHT1                                                     0x4001
#define GL_LIGHT2                                                     0x4002
#define GL_LIGHT3                                                     0x4003
#define GL_LIGHT4                                                     0x4004
#define GL_LIGHT5                                                     0x4005
#define GL_LIGHT6                                                     0x4006
#define GL_LIGHT7                                                     0x4007
#define GL_ABGR_EXT                                                   0x8000
#define GL_CONSTANT_COLOR                                             0x8001
#define GL_CONSTANT_COLOR_EXT                                         0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR                                   0x8002
#define GL_ONE_MINUS_CONSTANT_COLOR_EXT                               0x8002
#define GL_CONSTANT_ALPHA                                             0x8003
#define GL_CONSTANT_ALPHA_EXT                                         0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA                                   0x8004
#define GL_ONE_MINUS_CONSTANT_ALPHA_EXT                               0x8004
#define GL_BLEND_COLOR                                                0x8005
#define GL_BLEND_COLOR_EXT                                            0x8005
#define GL_FUNC_ADD                                                   0x8006
#define GL_FUNC_ADD_EXT                                               0x8006
#define GL_MIN                                                        0x8007
#define GL_MIN_EXT                                                    0x8007
#define GL_MAX                                                        0x8008
#define GL_MAX_EXT                                                    0x8008
#define GL_BLEND_EQUATION                                             0x8009
#define GL_BLEND_EQUATION_EXT                                         0x8009
#define GL_BLEND_EQUATION_RGB                                         0x8009
#define GL_BLEND_EQUATION_RGB_EXT                                     0x8009
#define GL_FUNC_SUBTRACT                                              0x800A
#define GL_FUNC_SUBTRACT_EXT                                          0x800A
#define GL_FUNC_REVERSE_SUBTRACT                                      0x800B
#define GL_FUNC_REVERSE_SUBTRACT_EXT                                  0x800B
#define GL_CMYK_EXT                                                   0x800C
#define GL_CMYKA_EXT                                                  0x800D
#define GL_PACK_CMYK_HINT_EXT                                         0x800E
#define GL_UNPACK_CMYK_HINT_EXT                                       0x800F
#define GL_CONVOLUTION_1D                                             0x8010
#define GL_CONVOLUTION_1D_EXT                                         0x8010
#define GL_CONVOLUTION_2D                                             0x8011
#define GL_CONVOLUTION_2D_EXT                                         0x8011
#define GL_SEPARABLE_2D                                               0x8012
#define GL_SEPARABLE_2D_EXT                                           0x8012
#define GL_CONVOLUTION_BORDER_MODE                                    0x8013
#define GL_CONVOLUTION_BORDER_MODE_EXT                                0x8013
#define GL_CONVOLUTION_FILTER_SCALE                                   0x8014
#define GL_CONVOLUTION_FILTER_SCALE_EXT                               0x8014
#define GL_CONVOLUTION_FILTER_BIAS                                    0x8015
#define GL_CONVOLUTION_FILTER_BIAS_EXT                                0x8015
#define GL_REDUCE                                                     0x8016
#define GL_REDUCE_EXT                                                 0x8016
#define GL_CONVOLUTION_FORMAT                                         0x8017
#define GL_CONVOLUTION_FORMAT_EXT                                     0x8017
#define GL_CONVOLUTION_WIDTH                                          0x8018
#define GL_CONVOLUTION_WIDTH_EXT                                      0x8018
#define GL_CONVOLUTION_HEIGHT                                         0x8019
#define GL_CONVOLUTION_HEIGHT_EXT                                     0x8019
#define GL_MAX_CONVOLUTION_WIDTH                                      0x801A
#define GL_MAX_CONVOLUTION_WIDTH_EXT                                  0x801A
#define GL_MAX_CONVOLUTION_HEIGHT                                     0x801B
#define GL_MAX_CONVOLUTION_HEIGHT_EXT                                 0x801B
#define GL_POST_CONVOLUTION_RED_SCALE                                 0x801C
#define GL_POST_CONVOLUTION_RED_SCALE_EXT                             0x801C
#define GL_POST_CONVOLUTION_GREEN_SCALE                               0x801D
#define GL_POST_CONVOLUTION_GREEN_SCALE_EXT                           0x801D
#define GL_POST_CONVOLUTION_BLUE_SCALE                                0x801E
#define GL_POST_CONVOLUTION_BLUE_SCALE_EXT                            0x801E
#define GL_POST_CONVOLUTION_ALPHA_SCALE                               0x801F
#define GL_POST_CONVOLUTION_ALPHA_SCALE_EXT                           0x801F
#define GL_POST_CONVOLUTION_RED_BIAS                                  0x8020
#define GL_POST_CONVOLUTION_RED_BIAS_EXT                              0x8020
#define GL_POST_CONVOLUTION_GREEN_BIAS                                0x8021
#define GL_POST_CONVOLUTION_GREEN_BIAS_EXT                            0x8021
#define GL_POST_CONVOLUTION_BLUE_BIAS                                 0x8022
#define GL_POST_CONVOLUTION_BLUE_BIAS_EXT                             0x8022
#define GL_POST_CONVOLUTION_ALPHA_BIAS                                0x8023
#define GL_POST_CONVOLUTION_ALPHA_BIAS_EXT                            0x8023
#define GL_HISTOGRAM                                                  0x8024
#define GL_HISTOGRAM_EXT                                              0x8024
#define GL_PROXY_HISTOGRAM                                            0x8025
#define GL_PROXY_HISTOGRAM_EXT                                        0x8025
#define GL_HISTOGRAM_WIDTH                                            0x8026
#define GL_HISTOGRAM_WIDTH_EXT                                        0x8026
#define GL_HISTOGRAM_FORMAT                                           0x8027
#define GL_HISTOGRAM_FORMAT_EXT                                       0x8027
#define GL_HISTOGRAM_RED_SIZE                                         0x8028
#define GL_HISTOGRAM_RED_SIZE_EXT                                     0x8028
#define GL_HISTOGRAM_GREEN_SIZE                                       0x8029
#define GL_HISTOGRAM_GREEN_SIZE_EXT                                   0x8029
#define GL_HISTOGRAM_BLUE_SIZE                                        0x802A
#define GL_HISTOGRAM_BLUE_SIZE_EXT                                    0x802A
#define GL_HISTOGRAM_ALPHA_SIZE                                       0x802B
#define GL_HISTOGRAM_ALPHA_SIZE_EXT                                   0x802B
#define GL_HISTOGRAM_LUMINANCE_SIZE                                   0x802C
#define GL_HISTOGRAM_LUMINANCE_SIZE_EXT                               0x802C
#define GL_HISTOGRAM_SINK                                             0x802D
#define GL_HISTOGRAM_SINK_EXT                                         0x802D
#define GL_MINMAX                                                     0x802E
#define GL_MINMAX_EXT                                                 0x802E
#define GL_MINMAX_FORMAT                                              0x802F
#define GL_MINMAX_FORMAT_EXT                                          0x802F
#define GL_MINMAX_SINK                                                0x8030
#define GL_MINMAX_SINK_EXT                                            0x8030
#define GL_TABLE_TOO_LARGE_EXT                                        0x8031
#define GL_TABLE_TOO_LARGE                                            0x8031
#define GL_UNSIGNED_BYTE_3_3_2                                        0x8032
#define GL_UNSIGNED_BYTE_3_3_2_EXT                                    0x8032
#define GL_UNSIGNED_SHORT_4_4_4_4                                     0x8033
#define GL_UNSIGNED_SHORT_4_4_4_4_EXT                                 0x8033
#define GL_UNSIGNED_SHORT_5_5_5_1                                     0x8034
#define GL_UNSIGNED_SHORT_5_5_5_1_EXT                                 0x8034
#define GL_UNSIGNED_INT_8_8_8_8                                       0x8035
#define GL_UNSIGNED_INT_8_8_8_8_EXT                                   0x8035
#define GL_UNSIGNED_INT_10_10_10_2                                    0x8036
#define GL_UNSIGNED_INT_10_10_10_2_EXT                                0x8036
#define GL_POLYGON_OFFSET_EXT                                         0x8037
#define GL_POLYGON_OFFSET_FILL                                        0x8037
#define GL_POLYGON_OFFSET_FACTOR                                      0x8038
#define GL_POLYGON_OFFSET_FACTOR_EXT                                  0x8038
#define GL_POLYGON_OFFSET_BIAS_EXT                                    0x8039
#define GL_RESCALE_NORMAL                                             0x803A
#define GL_RESCALE_NORMAL_EXT                                         0x803A
#define GL_ALPHA4                                                     0x803B
#define GL_ALPHA4_EXT                                                 0x803B
#define GL_ALPHA8                                                     0x803C
#define GL_ALPHA8_EXT                                                 0x803C
#define GL_ALPHA12                                                    0x803D
#define GL_ALPHA12_EXT                                                0x803D
#define GL_ALPHA16                                                    0x803E
#define GL_ALPHA16_EXT                                                0x803E
#define GL_LUMINANCE4                                                 0x803F
#define GL_LUMINANCE4_EXT                                             0x803F
#define GL_LUMINANCE8                                                 0x8040
#define GL_LUMINANCE8_EXT                                             0x8040
#define GL_LUMINANCE12                                                0x8041
#define GL_LUMINANCE12_EXT                                            0x8041
#define GL_LUMINANCE16                                                0x8042
#define GL_LUMINANCE16_EXT                                            0x8042
#define GL_LUMINANCE4_ALPHA4                                          0x8043
#define GL_LUMINANCE4_ALPHA4_EXT                                      0x8043
#define GL_LUMINANCE6_ALPHA2                                          0x8044
#define GL_LUMINANCE6_ALPHA2_EXT                                      0x8044
#define GL_LUMINANCE8_ALPHA8                                          0x8045
#define GL_LUMINANCE8_ALPHA8_EXT                                      0x8045
#define GL_LUMINANCE12_ALPHA4                                         0x8046
#define GL_LUMINANCE12_ALPHA4_EXT                                     0x8046
#define GL_LUMINANCE12_ALPHA12                                        0x8047
#define GL_LUMINANCE12_ALPHA12_EXT                                    0x8047
#define GL_LUMINANCE16_ALPHA16                                        0x8048
#define GL_LUMINANCE16_ALPHA16_EXT                                    0x8048
#define GL_INTENSITY                                                  0x8049
#define GL_INTENSITY_EXT                                              0x8049
#define GL_INTENSITY4                                                 0x804A
#define GL_INTENSITY4_EXT                                             0x804A
#define GL_INTENSITY8                                                 0x804B
#define GL_INTENSITY8_EXT                                             0x804B
#define GL_INTENSITY12                                                0x804C
#define GL_INTENSITY12_EXT                                            0x804C
#define GL_INTENSITY16                                                0x804D
#define GL_INTENSITY16_EXT                                            0x804D
#define GL_RGB2_EXT                                                   0x804E
#define GL_RGB4                                                       0x804F
#define GL_RGB4_EXT                                                   0x804F
#define GL_RGB5                                                       0x8050
#define GL_RGB5_EXT                                                   0x8050
#define GL_RGB8                                                       0x8051
#define GL_RGB8_EXT                                                   0x8051
#define GL_RGB10                                                      0x8052
#define GL_RGB10_EXT                                                  0x8052
#define GL_RGB12                                                      0x8053
#define GL_RGB12_EXT                                                  0x8053
#define GL_RGB16                                                      0x8054
#define GL_RGB16_EXT                                                  0x8054
#define GL_RGBA2                                                      0x8055
#define GL_RGBA2_EXT                                                  0x8055
#define GL_RGBA4                                                      0x8056
#define GL_RGBA4_EXT                                                  0x8056
#define GL_RGB5_A1                                                    0x8057
#define GL_RGB5_A1_EXT                                                0x8057
#define GL_RGBA8                                                      0x8058
#define GL_RGBA8_EXT                                                  0x8058
#define GL_RGB10_A2                                                   0x8059
#define GL_RGB10_A2_EXT                                               0x8059
#define GL_RGBA12                                                     0x805A
#define GL_RGBA12_EXT                                                 0x805A
#define GL_RGBA16                                                     0x805B
#define GL_RGBA16_EXT                                                 0x805B
#define GL_TEXTURE_RED_SIZE                                           0x805C
#define GL_TEXTURE_RED_SIZE_EXT                                       0x805C
#define GL_TEXTURE_GREEN_SIZE                                         0x805D
#define GL_TEXTURE_GREEN_SIZE_EXT                                     0x805D
#define GL_TEXTURE_BLUE_SIZE                                          0x805E
#define GL_TEXTURE_BLUE_SIZE_EXT                                      0x805E
#define GL_TEXTURE_ALPHA_SIZE                                         0x805F
#define GL_TEXTURE_ALPHA_SIZE_EXT                                     0x805F
#define GL_TEXTURE_LUMINANCE_SIZE                                     0x8060
#define GL_TEXTURE_LUMINANCE_SIZE_EXT                                 0x8060
#define GL_TEXTURE_INTENSITY_SIZE                                     0x8061
#define GL_TEXTURE_INTENSITY_SIZE_EXT                                 0x8061
#define GL_REPLACE_EXT                                                0x8062
#define GL_PROXY_TEXTURE_1D                                           0x8063
#define GL_PROXY_TEXTURE_1D_EXT                                       0x8063
#define GL_PROXY_TEXTURE_2D                                           0x8064
#define GL_PROXY_TEXTURE_2D_EXT                                       0x8064
#define GL_TEXTURE_TOO_LARGE_EXT                                      0x8065
#define GL_TEXTURE_PRIORITY                                           0x8066
#define GL_TEXTURE_PRIORITY_EXT                                       0x8066
#define GL_TEXTURE_RESIDENT                                           0x8067
#define GL_TEXTURE_RESIDENT_EXT                                       0x8067
#define GL_TEXTURE_1D_BINDING_EXT                                     0x8068
#define GL_TEXTURE_BINDING_1D                                         0x8068
#define GL_TEXTURE_2D_BINDING_EXT                                     0x8069
#define GL_TEXTURE_BINDING_2D                                         0x8069
#define GL_TEXTURE_3D_BINDING_EXT                                     0x806A
#define GL_TEXTURE_BINDING_3D                                         0x806A
#define GL_PACK_SKIP_IMAGES                                           0x806B
#define GL_PACK_SKIP_IMAGES_EXT                                       0x806B
#define GL_PACK_IMAGE_HEIGHT                                          0x806C
#define GL_PACK_IMAGE_HEIGHT_EXT                                      0x806C
#define GL_UNPACK_SKIP_IMAGES                                         0x806D
#define GL_UNPACK_SKIP_IMAGES_EXT                                     0x806D
#define GL_UNPACK_IMAGE_HEIGHT                                        0x806E
#define GL_UNPACK_IMAGE_HEIGHT_EXT                                    0x806E
#define GL_TEXTURE_3D                                                 0x806F
#define GL_TEXTURE_3D_EXT                                             0x806F
#define GL_PROXY_TEXTURE_3D                                           0x8070
#define GL_PROXY_TEXTURE_3D_EXT                                       0x8070
#define GL_TEXTURE_DEPTH                                              0x8071
#define GL_TEXTURE_DEPTH_EXT                                          0x8071
#define GL_TEXTURE_WRAP_R                                             0x8072
#define GL_TEXTURE_WRAP_R_EXT                                         0x8072
#define GL_MAX_3D_TEXTURE_SIZE                                        0x8073
#define GL_MAX_3D_TEXTURE_SIZE_EXT                                    0x8073
#define GL_VERTEX_ARRAY                                               0x8074
#define GL_VERTEX_ARRAY_EXT                                           0x8074
#define GL_VERTEX_ARRAY_KHR                                           0x8074
#define GL_NORMAL_ARRAY                                               0x8075
#define GL_NORMAL_ARRAY_EXT                                           0x8075
#define GL_COLOR_ARRAY                                                0x8076
#define GL_COLOR_ARRAY_EXT                                            0x8076
#define GL_INDEX_ARRAY                                                0x8077
#define GL_INDEX_ARRAY_EXT                                            0x8077
#define GL_TEXTURE_COORD_ARRAY                                        0x8078
#define GL_TEXTURE_COORD_ARRAY_EXT                                    0x8078
#define GL_EDGE_FLAG_ARRAY                                            0x8079
#define GL_EDGE_FLAG_ARRAY_EXT                                        0x8079
#define GL_VERTEX_ARRAY_SIZE                                          0x807A
#define GL_VERTEX_ARRAY_SIZE_EXT                                      0x807A
#define GL_VERTEX_ARRAY_TYPE                                          0x807B
#define GL_VERTEX_ARRAY_TYPE_EXT                                      0x807B
#define GL_VERTEX_ARRAY_STRIDE                                        0x807C
#define GL_VERTEX_ARRAY_STRIDE_EXT                                    0x807C
#define GL_VERTEX_ARRAY_COUNT_EXT                                     0x807D
#define GL_NORMAL_ARRAY_TYPE                                          0x807E
#define GL_NORMAL_ARRAY_TYPE_EXT                                      0x807E
#define GL_NORMAL_ARRAY_STRIDE                                        0x807F
#define GL_NORMAL_ARRAY_STRIDE_EXT                                    0x807F
#define GL_NORMAL_ARRAY_COUNT_EXT                                     0x8080
#define GL_COLOR_ARRAY_SIZE                                           0x8081
#define GL_COLOR_ARRAY_SIZE_EXT                                       0x8081
#define GL_COLOR_ARRAY_TYPE                                           0x8082
#define GL_COLOR_ARRAY_TYPE_EXT                                       0x8082
#define GL_COLOR_ARRAY_STRIDE                                         0x8083
#define GL_COLOR_ARRAY_STRIDE_EXT                                     0x8083
#define GL_COLOR_ARRAY_COUNT_EXT                                      0x8084
#define GL_INDEX_ARRAY_TYPE                                           0x8085
#define GL_INDEX_ARRAY_TYPE_EXT                                       0x8085
#define GL_INDEX_ARRAY_STRIDE                                         0x8086
#define GL_INDEX_ARRAY_STRIDE_EXT                                     0x8086
#define GL_INDEX_ARRAY_COUNT_EXT                                      0x8087
#define GL_TEXTURE_COORD_ARRAY_SIZE                                   0x8088
#define GL_TEXTURE_COORD_ARRAY_SIZE_EXT                               0x8088
#define GL_TEXTURE_COORD_ARRAY_TYPE                                   0x8089
#define GL_TEXTURE_COORD_ARRAY_TYPE_EXT                               0x8089
#define GL_TEXTURE_COORD_ARRAY_STRIDE                                 0x808A
#define GL_TEXTURE_COORD_ARRAY_STRIDE_EXT                             0x808A
#define GL_TEXTURE_COORD_ARRAY_COUNT_EXT                              0x808B
#define GL_EDGE_FLAG_ARRAY_STRIDE                                     0x808C
#define GL_EDGE_FLAG_ARRAY_STRIDE_EXT                                 0x808C
#define GL_EDGE_FLAG_ARRAY_COUNT_EXT                                  0x808D
#define GL_VERTEX_ARRAY_POINTER                                       0x808E
#define GL_VERTEX_ARRAY_POINTER_EXT                                   0x808E
#define GL_NORMAL_ARRAY_POINTER                                       0x808F
#define GL_NORMAL_ARRAY_POINTER_EXT                                   0x808F
#define GL_COLOR_ARRAY_POINTER                                        0x8090
#define GL_COLOR_ARRAY_POINTER_EXT                                    0x8090
#define GL_INDEX_ARRAY_POINTER                                        0x8091
#define GL_INDEX_ARRAY_POINTER_EXT                                    0x8091
#define GL_TEXTURE_COORD_ARRAY_POINTER                                0x8092
#define GL_TEXTURE_COORD_ARRAY_POINTER_EXT                            0x8092
#define GL_EDGE_FLAG_ARRAY_POINTER                                    0x8093
#define GL_EDGE_FLAG_ARRAY_POINTER_EXT                                0x8093
#define GL_MULTISAMPLE                                                0x809D
#define GL_MULTISAMPLE_ARB                                            0x809D
#define GL_MULTISAMPLE_EXT                                            0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE                                   0x809E
#define GL_SAMPLE_ALPHA_TO_COVERAGE_ARB                               0x809E
#define GL_SAMPLE_ALPHA_TO_MASK_EXT                                   0x809E
#define GL_SAMPLE_ALPHA_TO_ONE                                        0x809F
#define GL_SAMPLE_ALPHA_TO_ONE_ARB                                    0x809F
#define GL_SAMPLE_ALPHA_TO_ONE_EXT                                    0x809F
#define GL_SAMPLE_COVERAGE                                            0x80A0
#define GL_SAMPLE_COVERAGE_ARB                                        0x80A0
#define GL_SAMPLE_MASK_EXT                                            0x80A0
#define GL_1PASS_EXT                                                  0x80A1
#define GL_2PASS_0_EXT                                                0x80A2
#define GL_2PASS_1_EXT                                                0x80A3
#define GL_4PASS_0_EXT                                                0x80A4
#define GL_4PASS_1_EXT                                                0x80A5
#define GL_4PASS_2_EXT                                                0x80A6
#define GL_4PASS_3_EXT                                                0x80A7
#define GL_SAMPLE_BUFFERS                                             0x80A8
#define GL_SAMPLE_BUFFERS_ARB                                         0x80A8
#define GL_SAMPLE_BUFFERS_EXT                                         0x80A8
#define GL_SAMPLES                                                    0x80A9
#define GL_SAMPLES_ARB                                                0x80A9
#define GL_SAMPLES_EXT                                                0x80A9
#define GL_SAMPLE_COVERAGE_VALUE                                      0x80AA
#define GL_SAMPLE_COVERAGE_VALUE_ARB                                  0x80AA
#define GL_SAMPLE_MASK_VALUE_EXT                                      0x80AA
#define GL_SAMPLE_COVERAGE_INVERT                                     0x80AB
#define GL_SAMPLE_COVERAGE_INVERT_ARB                                 0x80AB
#define GL_SAMPLE_MASK_INVERT_EXT                                     0x80AB
#define GL_SAMPLE_PATTERN_EXT                                         0x80AC
#define GL_COLOR_MATRIX                                               0x80B1
#define GL_COLOR_MATRIX_STACK_DEPTH                                   0x80B2
#define GL_MAX_COLOR_MATRIX_STACK_DEPTH                               0x80B3
#define GL_POST_COLOR_MATRIX_RED_SCALE                                0x80B4
#define GL_POST_COLOR_MATRIX_GREEN_SCALE                              0x80B5
#define GL_POST_COLOR_MATRIX_BLUE_SCALE                               0x80B6
#define GL_POST_COLOR_MATRIX_ALPHA_SCALE                              0x80B7
#define GL_POST_COLOR_MATRIX_RED_BIAS                                 0x80B8
#define GL_POST_COLOR_MATRIX_GREEN_BIAS                               0x80B9
#define GL_POST_COLOR_MATRIX_BLUE_BIAS                                0x80BA
#define GL_POST_COLOR_MATRIX_ALPHA_BIAS                               0x80BB
#define GL_TEXTURE_COMPARE_FAIL_VALUE_ARB                             0x80BF
#define GL_BLEND_DST_RGB                                              0x80C8
#define GL_BLEND_DST_RGB_EXT                                          0x80C8
#define GL_BLEND_SRC_RGB                                              0x80C9
#define GL_BLEND_SRC_RGB_EXT                                          0x80C9
#define GL_BLEND_DST_ALPHA                                            0x80CA
#define GL_BLEND_DST_ALPHA_EXT                                        0x80CA
#define GL_BLEND_SRC_ALPHA                                            0x80CB
#define GL_BLEND_SRC_ALPHA_EXT                                        0x80CB
#define GL_422_EXT                                                    0x80CC
#define GL_422_REV_EXT                                                0x80CD
#define GL_422_AVERAGE_EXT                                            0x80CE
#define GL_422_REV_AVERAGE_EXT                                        0x80CF
#define GL_COLOR_TABLE                                                0x80D0
#define GL_POST_CONVOLUTION_COLOR_TABLE                               0x80D1
#define GL_POST_COLOR_MATRIX_COLOR_TABLE                              0x80D2
#define GL_PROXY_COLOR_TABLE                                          0x80D3
#define GL_PROXY_POST_CONVOLUTION_COLOR_TABLE                         0x80D4
#define GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE                        0x80D5
#define GL_COLOR_TABLE_SCALE                                          0x80D6
#define GL_COLOR_TABLE_BIAS                                           0x80D7
#define GL_COLOR_TABLE_FORMAT                                         0x80D8
#define GL_COLOR_TABLE_WIDTH                                          0x80D9
#define GL_COLOR_TABLE_RED_SIZE                                       0x80DA
#define GL_COLOR_TABLE_GREEN_SIZE                                     0x80DB
#define GL_COLOR_TABLE_BLUE_SIZE                                      0x80DC
#define GL_COLOR_TABLE_ALPHA_SIZE                                     0x80DD
#define GL_COLOR_TABLE_LUMINANCE_SIZE                                 0x80DE
#define GL_COLOR_TABLE_INTENSITY_SIZE                                 0x80DF
#define GL_BGR                                                        0x80E0
#define GL_BGR_EXT                                                    0x80E0
#define GL_BGRA                                                       0x80E1
#define GL_BGRA_EXT                                                   0x80E1
#define GL_COLOR_INDEX1_EXT                                           0x80E2
#define GL_COLOR_INDEX2_EXT                                           0x80E3
#define GL_COLOR_INDEX4_EXT                                           0x80E4
#define GL_COLOR_INDEX8_EXT                                           0x80E5
#define GL_COLOR_INDEX12_EXT                                          0x80E6
#define GL_COLOR_INDEX16_EXT                                          0x80E7
#define GL_MAX_ELEMENTS_VERTICES                                      0x80E8
#define GL_MAX_ELEMENTS_VERTICES_EXT                                  0x80E8
#define GL_MAX_ELEMENTS_INDICES                                       0x80E9
#define GL_MAX_ELEMENTS_INDICES_EXT                                   0x80E9
#define GL_TEXTURE_INDEX_SIZE_EXT                                     0x80ED
#define GL_PARAMETER_BUFFER                                           0x80EE
#define GL_PARAMETER_BUFFER_ARB                                       0x80EE
#define GL_PARAMETER_BUFFER_BINDING                                   0x80EF
#define GL_PARAMETER_BUFFER_BINDING_ARB                               0x80EF
#define GL_CLIP_VOLUME_CLIPPING_HINT_EXT                              0x80F0
#define GL_POINT_SIZE_MIN                                             0x8126
#define GL_POINT_SIZE_MIN_ARB                                         0x8126
#define GL_POINT_SIZE_MIN_EXT                                         0x8126
#define GL_POINT_SIZE_MAX                                             0x8127
#define GL_POINT_SIZE_MAX_ARB                                         0x8127
#define GL_POINT_SIZE_MAX_EXT                                         0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE                                  0x8128
#define GL_POINT_FADE_THRESHOLD_SIZE_ARB                              0x8128
#define GL_POINT_FADE_THRESHOLD_SIZE_EXT                              0x8128
#define GL_DISTANCE_ATTENUATION_EXT                                   0x8129
#define GL_POINT_DISTANCE_ATTENUATION                                 0x8129
#define GL_POINT_DISTANCE_ATTENUATION_ARB                             0x8129
#define GL_CLAMP_TO_BORDER                                            0x812D
#define GL_CLAMP_TO_BORDER_ARB                                        0x812D
#define GL_CLAMP_TO_EDGE                                              0x812F
#define GL_TEXTURE_MIN_LOD                                            0x813A
#define GL_TEXTURE_MAX_LOD                                            0x813B
#define GL_TEXTURE_BASE_LEVEL                                         0x813C
#define GL_TEXTURE_MAX_LEVEL                                          0x813D
#define GL_CONSTANT_BORDER                                            0x8151
#define GL_REPLICATE_BORDER                                           0x8153
#define GL_CONVOLUTION_BORDER_COLOR                                   0x8154
#define GL_GENERATE_MIPMAP                                            0x8191
#define GL_GENERATE_MIPMAP_HINT                                       0x8192
#define GL_DEPTH_COMPONENT16                                          0x81A5
#define GL_DEPTH_COMPONENT16_ARB                                      0x81A5
#define GL_DEPTH_COMPONENT24                                          0x81A6
#define GL_DEPTH_COMPONENT24_ARB                                      0x81A6
#define GL_DEPTH_COMPONENT32                                          0x81A7
#define GL_DEPTH_COMPONENT32_ARB                                      0x81A7
#define GL_ARRAY_ELEMENT_LOCK_FIRST_EXT                               0x81A8
#define GL_ARRAY_ELEMENT_LOCK_COUNT_EXT                               0x81A9
#define GL_CULL_VERTEX_EXT                                            0x81AA
#define GL_CULL_VERTEX_EYE_POSITION_EXT                               0x81AB
#define GL_CULL_VERTEX_OBJECT_POSITION_EXT                            0x81AC
#define GL_IUI_V2F_EXT                                                0x81AD
#define GL_IUI_V3F_EXT                                                0x81AE
#define GL_IUI_N3F_V2F_EXT                                            0x81AF
#define GL_IUI_N3F_V3F_EXT                                            0x81B0
#define GL_T2F_IUI_V2F_EXT                                            0x81B1
#define GL_T2F_IUI_V3F_EXT                                            0x81B2
#define GL_T2F_IUI_N3F_V2F_EXT                                        0x81B3
#define GL_T2F_IUI_N3F_V3F_EXT                                        0x81B4
#define GL_INDEX_TEST_EXT                                             0x81B5
#define GL_INDEX_TEST_FUNC_EXT                                        0x81B6
#define GL_INDEX_TEST_REF_EXT                                         0x81B7
#define GL_INDEX_MATERIAL_EXT                                         0x81B8
#define GL_INDEX_MATERIAL_PARAMETER_EXT                               0x81B9
#define GL_INDEX_MATERIAL_FACE_EXT                                    0x81BA
#define GL_LIGHT_MODEL_COLOR_CONTROL                                  0x81F8
#define GL_LIGHT_MODEL_COLOR_CONTROL_EXT                              0x81F8
#define GL_SINGLE_COLOR                                               0x81F9
#define GL_SINGLE_COLOR_EXT                                           0x81F9
#define GL_SEPARATE_SPECULAR_COLOR                                    0x81FA
#define GL_SEPARATE_SPECULAR_COLOR_EXT                                0x81FA
#define GL_SHARED_TEXTURE_PALETTE_EXT                                 0x81FB
#define GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING                      0x8210
#define GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE                      0x8211
#define GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE                            0x8212
#define GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE                          0x8213
#define GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE                           0x8214
#define GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE                          0x8215
#define GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE                          0x8216
#define GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE                        0x8217
#define GL_FRAMEBUFFER_DEFAULT                                        0x8218
#define GL_FRAMEBUFFER_UNDEFINED                                      0x8219
#define GL_DEPTH_STENCIL_ATTACHMENT                                   0x821A
#define GL_MAJOR_VERSION                                              0x821B
#define GL_MINOR_VERSION                                              0x821C
#define GL_NUM_EXTENSIONS                                             0x821D
#define GL_CONTEXT_FLAGS                                              0x821E
#define GL_BUFFER_IMMUTABLE_STORAGE                                   0x821F
#define GL_BUFFER_STORAGE_FLAGS                                       0x8220
#define GL_PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED                    0x8221
#define GL_INDEX                                                      0x8222
#define GL_COMPRESSED_RED                                             0x8225
#define GL_COMPRESSED_RG                                              0x8226
#define GL_RG                                                         0x8227
#define GL_RG_INTEGER                                                 0x8228
#define GL_R8                                                         0x8229
#define GL_R16                                                        0x822A
#define GL_RG8                                                        0x822B
#define GL_RG16                                                       0x822C
#define GL_R16F                                                       0x822D
#define GL_R32F                                                       0x822E
#define GL_RG16F                                                      0x822F
#define GL_RG32F                                                      0x8230
#define GL_R8I                                                        0x8231
#define GL_R8UI                                                       0x8232
#define GL_R16I                                                       0x8233
#define GL_R16UI                                                      0x8234
#define GL_R32I                                                       0x8235
#define GL_R32UI                                                      0x8236
#define GL_RG8I                                                       0x8237
#define GL_RG8UI                                                      0x8238
#define GL_RG16I                                                      0x8239
#define GL_RG16UI                                                     0x823A
#define GL_RG32I                                                      0x823B
#define GL_RG32UI                                                     0x823C
#define GL_SYNC_CL_EVENT_ARB                                          0x8240
#define GL_SYNC_CL_EVENT_COMPLETE_ARB                                 0x8241
#define GL_DEBUG_OUTPUT_SYNCHRONOUS                                   0x8242
#define GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB                               0x8242
#define GL_DEBUG_OUTPUT_SYNCHRONOUS_KHR                               0x8242
#define GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH                           0x8243
#define GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH_ARB                       0x8243
#define GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH_KHR                       0x8243
#define GL_DEBUG_CALLBACK_FUNCTION                                    0x8244
#define GL_DEBUG_CALLBACK_FUNCTION_ARB                                0x8244
#define GL_DEBUG_CALLBACK_FUNCTION_KHR                                0x8244
#define GL_DEBUG_CALLBACK_USER_PARAM                                  0x8245
#define GL_DEBUG_CALLBACK_USER_PARAM_ARB                              0x8245
#define GL_DEBUG_CALLBACK_USER_PARAM_KHR                              0x8245
#define GL_DEBUG_SOURCE_API                                           0x8246
#define GL_DEBUG_SOURCE_API_ARB                                       0x8246
#define GL_DEBUG_SOURCE_API_KHR                                       0x8246
#define GL_DEBUG_SOURCE_WINDOW_SYSTEM                                 0x8247
#define GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB                             0x8247
#define GL_DEBUG_SOURCE_WINDOW_SYSTEM_KHR                             0x8247
#define GL_DEBUG_SOURCE_SHADER_COMPILER                               0x8248
#define GL_DEBUG_SOURCE_SHADER_COMPILER_ARB                           0x8248
#define GL_DEBUG_SOURCE_SHADER_COMPILER_KHR                           0x8248
#define GL_DEBUG_SOURCE_THIRD_PARTY                                   0x8249
#define GL_DEBUG_SOURCE_THIRD_PARTY_ARB                               0x8249
#define GL_DEBUG_SOURCE_THIRD_PARTY_KHR                               0x8249
#define GL_DEBUG_SOURCE_APPLICATION                                   0x824A
#define GL_DEBUG_SOURCE_APPLICATION_ARB                               0x824A
#define GL_DEBUG_SOURCE_APPLICATION_KHR                               0x824A
#define GL_DEBUG_SOURCE_OTHER                                         0x824B
#define GL_DEBUG_SOURCE_OTHER_ARB                                     0x824B
#define GL_DEBUG_SOURCE_OTHER_KHR                                     0x824B
#define GL_DEBUG_TYPE_ERROR                                           0x824C
#define GL_DEBUG_TYPE_ERROR_ARB                                       0x824C
#define GL_DEBUG_TYPE_ERROR_KHR                                       0x824C
#define GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR                             0x824D
#define GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB                         0x824D
#define GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_KHR                         0x824D
#define GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR                              0x824E
#define GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB                          0x824E
#define GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_KHR                          0x824E
#define GL_DEBUG_TYPE_PORTABILITY                                     0x824F
#define GL_DEBUG_TYPE_PORTABILITY_ARB                                 0x824F
#define GL_DEBUG_TYPE_PORTABILITY_KHR                                 0x824F
#define GL_DEBUG_TYPE_PERFORMANCE                                     0x8250
#define GL_DEBUG_TYPE_PERFORMANCE_ARB                                 0x8250
#define GL_DEBUG_TYPE_PERFORMANCE_KHR                                 0x8250
#define GL_DEBUG_TYPE_OTHER                                           0x8251
#define GL_DEBUG_TYPE_OTHER_ARB                                       0x8251
#define GL_DEBUG_TYPE_OTHER_KHR                                       0x8251
#define GL_LOSE_CONTEXT_ON_RESET                                      0x8252
#define GL_LOSE_CONTEXT_ON_RESET_ARB                                  0x8252
#define GL_LOSE_CONTEXT_ON_RESET_KHR                                  0x8252
#define GL_GUILTY_CONTEXT_RESET                                       0x8253
#define GL_GUILTY_CONTEXT_RESET_ARB                                   0x8253
#define GL_GUILTY_CONTEXT_RESET_KHR                                   0x8253
#define GL_INNOCENT_CONTEXT_RESET                                     0x8254
#define GL_INNOCENT_CONTEXT_RESET_ARB                                 0x8254
#define GL_INNOCENT_CONTEXT_RESET_KHR                                 0x8254
#define GL_UNKNOWN_CONTEXT_RESET                                      0x8255
#define GL_UNKNOWN_CONTEXT_RESET_ARB                                  0x8255
#define GL_UNKNOWN_CONTEXT_RESET_KHR                                  0x8255
#define GL_RESET_NOTIFICATION_STRATEGY                                0x8256
#define GL_RESET_NOTIFICATION_STRATEGY_ARB                            0x8256
#define GL_RESET_NOTIFICATION_STRATEGY_KHR                            0x8256
#define GL_PROGRAM_BINARY_RETRIEVABLE_HINT                            0x8257
#define GL_PROGRAM_SEPARABLE                                          0x8258
#define GL_PROGRAM_SEPARABLE_EXT                                      0x8258
#define GL_ACTIVE_PROGRAM                                             0x8259
#define GL_ACTIVE_PROGRAM_EXT                                         0x8259
#define GL_PROGRAM_PIPELINE_BINDING                                   0x825A
#define GL_PROGRAM_PIPELINE_BINDING_EXT                               0x825A
#define GL_MAX_VIEWPORTS                                              0x825B
#define GL_VIEWPORT_SUBPIXEL_BITS                                     0x825C
#define GL_VIEWPORT_BOUNDS_RANGE                                      0x825D
#define GL_LAYER_PROVOKING_VERTEX                                     0x825E
#define GL_VIEWPORT_INDEX_PROVOKING_VERTEX                            0x825F
#define GL_UNDEFINED_VERTEX                                           0x8260
#define GL_NO_RESET_NOTIFICATION                                      0x8261
#define GL_NO_RESET_NOTIFICATION_ARB                                  0x8261
#define GL_NO_RESET_NOTIFICATION_KHR                                  0x8261
#define GL_MAX_COMPUTE_SHARED_MEMORY_SIZE                             0x8262
#define GL_MAX_COMPUTE_UNIFORM_COMPONENTS                             0x8263
#define GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS                         0x8264
#define GL_MAX_COMPUTE_ATOMIC_COUNTERS                                0x8265
#define GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS                    0x8266
#define GL_COMPUTE_WORK_GROUP_SIZE                                    0x8267
#define GL_DEBUG_TYPE_MARKER                                          0x8268
#define GL_DEBUG_TYPE_MARKER_KHR                                      0x8268
#define GL_DEBUG_TYPE_PUSH_GROUP                                      0x8269
#define GL_DEBUG_TYPE_PUSH_GROUP_KHR                                  0x8269
#define GL_DEBUG_TYPE_POP_GROUP                                       0x826A
#define GL_DEBUG_TYPE_POP_GROUP_KHR                                   0x826A
#define GL_DEBUG_SEVERITY_NOTIFICATION                                0x826B
#define GL_DEBUG_SEVERITY_NOTIFICATION_KHR                            0x826B
#define GL_MAX_DEBUG_GROUP_STACK_DEPTH                                0x826C
#define GL_MAX_DEBUG_GROUP_STACK_DEPTH_KHR                            0x826C
#define GL_DEBUG_GROUP_STACK_DEPTH                                    0x826D
#define GL_DEBUG_GROUP_STACK_DEPTH_KHR                                0x826D
#define GL_MAX_UNIFORM_LOCATIONS                                      0x826E
#define GL_INTERNALFORMAT_SUPPORTED                                   0x826F
#define GL_INTERNALFORMAT_PREFERRED                                   0x8270
#define GL_INTERNALFORMAT_RED_SIZE                                    0x8271
#define GL_INTERNALFORMAT_GREEN_SIZE                                  0x8272
#define GL_INTERNALFORMAT_BLUE_SIZE                                   0x8273
#define GL_INTERNALFORMAT_ALPHA_SIZE                                  0x8274
#define GL_INTERNALFORMAT_DEPTH_SIZE                                  0x8275
#define GL_INTERNALFORMAT_STENCIL_SIZE                                0x8276
#define GL_INTERNALFORMAT_SHARED_SIZE                                 0x8277
#define GL_INTERNALFORMAT_RED_TYPE                                    0x8278
#define GL_INTERNALFORMAT_GREEN_TYPE                                  0x8279
#define GL_INTERNALFORMAT_BLUE_TYPE                                   0x827A
#define GL_INTERNALFORMAT_ALPHA_TYPE                                  0x827B
#define GL_INTERNALFORMAT_DEPTH_TYPE                                  0x827C
#define GL_INTERNALFORMAT_STENCIL_TYPE                                0x827D
#define GL_MAX_WIDTH                                                  0x827E
#define GL_MAX_HEIGHT                                                 0x827F
#define GL_MAX_DEPTH                                                  0x8280
#define GL_MAX_LAYERS                                                 0x8281
#define GL_MAX_COMBINED_DIMENSIONS                                    0x8282
#define GL_COLOR_COMPONENTS                                           0x8283
#define GL_DEPTH_COMPONENTS                                           0x8284
#define GL_STENCIL_COMPONENTS                                         0x8285
#define GL_COLOR_RENDERABLE                                           0x8286
#define GL_DEPTH_RENDERABLE                                           0x8287
#define GL_STENCIL_RENDERABLE                                         0x8288
#define GL_FRAMEBUFFER_RENDERABLE                                     0x8289
#define GL_FRAMEBUFFER_RENDERABLE_LAYERED                             0x828A
#define GL_FRAMEBUFFER_BLEND                                          0x828B
#define GL_READ_PIXELS                                                0x828C
#define GL_READ_PIXELS_FORMAT                                         0x828D
#define GL_READ_PIXELS_TYPE                                           0x828E
#define GL_TEXTURE_IMAGE_FORMAT                                       0x828F
#define GL_TEXTURE_IMAGE_TYPE                                         0x8290
#define GL_GET_TEXTURE_IMAGE_FORMAT                                   0x8291
#define GL_GET_TEXTURE_IMAGE_TYPE                                     0x8292
#define GL_MIPMAP                                                     0x8293
#define GL_MANUAL_GENERATE_MIPMAP                                     0x8294
#define GL_AUTO_GENERATE_MIPMAP                                       0x8295
#define GL_COLOR_ENCODING                                             0x8296
#define GL_SRGB_READ                                                  0x8297
#define GL_SRGB_WRITE                                                 0x8298
#define GL_SRGB_DECODE_ARB                                            0x8299
#define GL_FILTER                                                     0x829A
#define GL_VERTEX_TEXTURE                                             0x829B
#define GL_TESS_CONTROL_TEXTURE                                       0x829C
#define GL_TESS_EVALUATION_TEXTURE                                    0x829D
#define GL_GEOMETRY_TEXTURE                                           0x829E
#define GL_FRAGMENT_TEXTURE                                           0x829F
#define GL_COMPUTE_TEXTURE                                            0x82A0
#define GL_TEXTURE_SHADOW                                             0x82A1
#define GL_TEXTURE_GATHER                                             0x82A2
#define GL_TEXTURE_GATHER_SHADOW                                      0x82A3
#define GL_SHADER_IMAGE_LOAD                                          0x82A4
#define GL_SHADER_IMAGE_STORE                                         0x82A5
#define GL_SHADER_IMAGE_ATOMIC                                        0x82A6
#define GL_IMAGE_TEXEL_SIZE                                           0x82A7
#define GL_IMAGE_COMPATIBILITY_CLASS                                  0x82A8
#define GL_IMAGE_PIXEL_FORMAT                                         0x82A9
#define GL_IMAGE_PIXEL_TYPE                                           0x82AA
#define GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST                        0x82AC
#define GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST                      0x82AD
#define GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE                       0x82AE
#define GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE                     0x82AF
#define GL_TEXTURE_COMPRESSED_BLOCK_WIDTH                             0x82B1
#define GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT                            0x82B2
#define GL_TEXTURE_COMPRESSED_BLOCK_SIZE                              0x82B3
#define GL_CLEAR_BUFFER                                               0x82B4
#define GL_TEXTURE_VIEW                                               0x82B5
#define GL_VIEW_COMPATIBILITY_CLASS                                   0x82B6
#define GL_FULL_SUPPORT                                               0x82B7
#define GL_CAVEAT_SUPPORT                                             0x82B8
#define GL_IMAGE_CLASS_4_X_32                                         0x82B9
#define GL_IMAGE_CLASS_2_X_32                                         0x82BA
#define GL_IMAGE_CLASS_1_X_32                                         0x82BB
#define GL_IMAGE_CLASS_4_X_16                                         0x82BC
#define GL_IMAGE_CLASS_2_X_16                                         0x82BD
#define GL_IMAGE_CLASS_1_X_16                                         0x82BE
#define GL_IMAGE_CLASS_4_X_8                                          0x82BF
#define GL_IMAGE_CLASS_2_X_8                                          0x82C0
#define GL_IMAGE_CLASS_1_X_8                                          0x82C1
#define GL_IMAGE_CLASS_11_11_10                                       0x82C2
#define GL_IMAGE_CLASS_10_10_10_2                                     0x82C3
#define GL_VIEW_CLASS_128_BITS                                        0x82C4
#define GL_VIEW_CLASS_96_BITS                                         0x82C5
#define GL_VIEW_CLASS_64_BITS                                         0x82C6
#define GL_VIEW_CLASS_48_BITS                                         0x82C7
#define GL_VIEW_CLASS_32_BITS                                         0x82C8
#define GL_VIEW_CLASS_24_BITS                                         0x82C9
#define GL_VIEW_CLASS_16_BITS                                         0x82CA
#define GL_VIEW_CLASS_8_BITS                                          0x82CB
#define GL_VIEW_CLASS_S3TC_DXT1_RGB                                   0x82CC
#define GL_VIEW_CLASS_S3TC_DXT1_RGBA                                  0x82CD
#define GL_VIEW_CLASS_S3TC_DXT3_RGBA                                  0x82CE
#define GL_VIEW_CLASS_S3TC_DXT5_RGBA                                  0x82CF
#define GL_VIEW_CLASS_RGTC1_RED                                       0x82D0
#define GL_VIEW_CLASS_RGTC2_RG                                        0x82D1
#define GL_VIEW_CLASS_BPTC_UNORM                                      0x82D2
#define GL_VIEW_CLASS_BPTC_FLOAT                                      0x82D3
#define GL_VERTEX_ATTRIB_BINDING                                      0x82D4
#define GL_VERTEX_ATTRIB_RELATIVE_OFFSET                              0x82D5
#define GL_VERTEX_BINDING_DIVISOR                                     0x82D6
#define GL_VERTEX_BINDING_OFFSET                                      0x82D7
#define GL_VERTEX_BINDING_STRIDE                                      0x82D8
#define GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET                          0x82D9
#define GL_MAX_VERTEX_ATTRIB_BINDINGS                                 0x82DA
#define GL_TEXTURE_VIEW_MIN_LEVEL                                     0x82DB
#define GL_TEXTURE_VIEW_NUM_LEVELS                                    0x82DC
#define GL_TEXTURE_VIEW_MIN_LAYER                                     0x82DD
#define GL_TEXTURE_VIEW_NUM_LAYERS                                    0x82DE
#define GL_TEXTURE_IMMUTABLE_LEVELS                                   0x82DF
#define GL_BUFFER                                                     0x82E0
#define GL_BUFFER_KHR                                                 0x82E0
#define GL_SHADER                                                     0x82E1
#define GL_SHADER_KHR                                                 0x82E1
#define GL_PROGRAM                                                    0x82E2
#define GL_PROGRAM_KHR                                                0x82E2
#define GL_QUERY                                                      0x82E3
#define GL_QUERY_KHR                                                  0x82E3
#define GL_PROGRAM_PIPELINE                                           0x82E4
#define GL_PROGRAM_PIPELINE_KHR                                       0x82E4
#define GL_MAX_VERTEX_ATTRIB_STRIDE                                   0x82E5
#define GL_SAMPLER                                                    0x82E6
#define GL_SAMPLER_KHR                                                0x82E6
#define GL_DISPLAY_LIST                                               0x82E7
#define GL_MAX_LABEL_LENGTH                                           0x82E8
#define GL_MAX_LABEL_LENGTH_KHR                                       0x82E8
#define GL_NUM_SHADING_LANGUAGE_VERSIONS                              0x82E9
#define GL_QUERY_TARGET                                               0x82EA
#define GL_TRANSFORM_FEEDBACK_OVERFLOW                                0x82EC
#define GL_TRANSFORM_FEEDBACK_OVERFLOW_ARB                            0x82EC
#define GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW                         0x82ED
#define GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW_ARB                     0x82ED
#define GL_VERTICES_SUBMITTED                                         0x82EE
#define GL_VERTICES_SUBMITTED_ARB                                     0x82EE
#define GL_PRIMITIVES_SUBMITTED                                       0x82EF
#define GL_PRIMITIVES_SUBMITTED_ARB                                   0x82EF
#define GL_VERTEX_SHADER_INVOCATIONS                                  0x82F0
#define GL_VERTEX_SHADER_INVOCATIONS_ARB                              0x82F0
#define GL_TESS_CONTROL_SHADER_PATCHES                                0x82F1
#define GL_TESS_CONTROL_SHADER_PATCHES_ARB                            0x82F1
#define GL_TESS_EVALUATION_SHADER_INVOCATIONS                         0x82F2
#define GL_TESS_EVALUATION_SHADER_INVOCATIONS_ARB                     0x82F2
#define GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED                         0x82F3
#define GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED_ARB                     0x82F3
#define GL_FRAGMENT_SHADER_INVOCATIONS                                0x82F4
#define GL_FRAGMENT_SHADER_INVOCATIONS_ARB                            0x82F4
#define GL_COMPUTE_SHADER_INVOCATIONS                                 0x82F5
#define GL_COMPUTE_SHADER_INVOCATIONS_ARB                             0x82F5
#define GL_CLIPPING_INPUT_PRIMITIVES                                  0x82F6
#define GL_CLIPPING_INPUT_PRIMITIVES_ARB                              0x82F6
#define GL_CLIPPING_OUTPUT_PRIMITIVES                                 0x82F7
#define GL_CLIPPING_OUTPUT_PRIMITIVES_ARB                             0x82F7
#define GL_SPARSE_BUFFER_PAGE_SIZE_ARB                                0x82F8
#define GL_MAX_CULL_DISTANCES                                         0x82F9
#define GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES                       0x82FA
#define GL_CONTEXT_RELEASE_BEHAVIOR                                   0x82FB
#define GL_CONTEXT_RELEASE_BEHAVIOR_KHR                               0x82FB
#define GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH                             0x82FC
#define GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_KHR                         0x82FC
#define GL_PIXEL_TRANSFORM_2D_EXT                                     0x8330
#define GL_PIXEL_MAG_FILTER_EXT                                       0x8331
#define GL_PIXEL_MIN_FILTER_EXT                                       0x8332
#define GL_PIXEL_CUBIC_WEIGHT_EXT                                     0x8333
#define GL_CUBIC_EXT                                                  0x8334
#define GL_AVERAGE_EXT                                                0x8335
#define GL_PIXEL_TRANSFORM_2D_STACK_DEPTH_EXT                         0x8336
#define GL_MAX_PIXEL_TRANSFORM_2D_STACK_DEPTH_EXT                     0x8337
#define GL_PIXEL_TRANSFORM_2D_MATRIX_EXT                              0x8338
#define GL_FRAGMENT_MATERIAL_EXT                                      0x8349
#define GL_FRAGMENT_NORMAL_EXT                                        0x834A
#define GL_FRAGMENT_COLOR_EXT                                         0x834C
#define GL_ATTENUATION_EXT                                            0x834D
#define GL_SHADOW_ATTENUATION_EXT                                     0x834E
#define GL_TEXTURE_APPLICATION_MODE_EXT                               0x834F
#define GL_TEXTURE_LIGHT_EXT                                          0x8350
#define GL_TEXTURE_MATERIAL_FACE_EXT                                  0x8351
#define GL_TEXTURE_MATERIAL_PARAMETER_EXT                             0x8352
#define GL_UNSIGNED_BYTE_2_3_3_REV                                    0x8362
#define GL_UNSIGNED_SHORT_5_6_5                                       0x8363
#define GL_UNSIGNED_SHORT_5_6_5_REV                                   0x8364
#define GL_UNSIGNED_SHORT_4_4_4_4_REV                                 0x8365
#define GL_UNSIGNED_SHORT_1_5_5_5_REV                                 0x8366
#define GL_UNSIGNED_INT_8_8_8_8_REV                                   0x8367
#define GL_UNSIGNED_INT_2_10_10_10_REV                                0x8368
#define GL_MIRRORED_REPEAT                                            0x8370
#define GL_MIRRORED_REPEAT_ARB                                        0x8370
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT                               0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT                              0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT                              0x83F2
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT                              0x83F3
#define GL_PARALLEL_ARRAYS_INTEL                                      0x83F4
#define GL_VERTEX_ARRAY_PARALLEL_POINTERS_INTEL                       0x83F5
#define GL_NORMAL_ARRAY_PARALLEL_POINTERS_INTEL                       0x83F6
#define GL_COLOR_ARRAY_PARALLEL_POINTERS_INTEL                        0x83F7
#define GL_TEXTURE_COORD_ARRAY_PARALLEL_POINTERS_INTEL                0x83F8
#define GL_PERFQUERY_DONOT_FLUSH_INTEL                                0x83F9
#define GL_PERFQUERY_FLUSH_INTEL                                      0x83FA
#define GL_PERFQUERY_WAIT_INTEL                                       0x83FB
#define GL_BLACKHOLE_RENDER_INTEL                                     0x83FC
#define GL_CONSERVATIVE_RASTERIZATION_INTEL                           0x83FE
#define GL_TEXTURE_MEMORY_LAYOUT_INTEL                                0x83FF
#define GL_TANGENT_ARRAY_EXT                                          0x8439
#define GL_BINORMAL_ARRAY_EXT                                         0x843A
#define GL_CURRENT_TANGENT_EXT                                        0x843B
#define GL_CURRENT_BINORMAL_EXT                                       0x843C
#define GL_TANGENT_ARRAY_TYPE_EXT                                     0x843E
#define GL_TANGENT_ARRAY_STRIDE_EXT                                   0x843F
#define GL_BINORMAL_ARRAY_TYPE_EXT                                    0x8440
#define GL_BINORMAL_ARRAY_STRIDE_EXT                                  0x8441
#define GL_TANGENT_ARRAY_POINTER_EXT                                  0x8442
#define GL_BINORMAL_ARRAY_POINTER_EXT                                 0x8443
#define GL_MAP1_TANGENT_EXT                                           0x8444
#define GL_MAP2_TANGENT_EXT                                           0x8445
#define GL_MAP1_BINORMAL_EXT                                          0x8446
#define GL_MAP2_BINORMAL_EXT                                          0x8447
#define GL_FOG_COORDINATE_SOURCE                                      0x8450
#define GL_FOG_COORDINATE_SOURCE_EXT                                  0x8450
#define GL_FOG_COORD_SRC                                              0x8450
#define GL_FOG_COORDINATE                                             0x8451
#define GL_FOG_COORD                                                  0x8451
#define GL_FOG_COORDINATE_EXT                                         0x8451
#define GL_FRAGMENT_DEPTH                                             0x8452
#define GL_FRAGMENT_DEPTH_EXT                                         0x8452
#define GL_CURRENT_FOG_COORDINATE                                     0x8453
#define GL_CURRENT_FOG_COORD                                          0x8453
#define GL_CURRENT_FOG_COORDINATE_EXT                                 0x8453
#define GL_FOG_COORDINATE_ARRAY_TYPE                                  0x8454
#define GL_FOG_COORDINATE_ARRAY_TYPE_EXT                              0x8454
#define GL_FOG_COORD_ARRAY_TYPE                                       0x8454
#define GL_FOG_COORDINATE_ARRAY_STRIDE                                0x8455
#define GL_FOG_COORDINATE_ARRAY_STRIDE_EXT                            0x8455
#define GL_FOG_COORD_ARRAY_STRIDE                                     0x8455
#define GL_FOG_COORDINATE_ARRAY_POINTER                               0x8456
#define GL_FOG_COORDINATE_ARRAY_POINTER_EXT                           0x8456
#define GL_FOG_COORD_ARRAY_POINTER                                    0x8456
#define GL_FOG_COORDINATE_ARRAY                                       0x8457
#define GL_FOG_COORDINATE_ARRAY_EXT                                   0x8457
#define GL_FOG_COORD_ARRAY                                            0x8457
#define GL_COLOR_SUM                                                  0x8458
#define GL_COLOR_SUM_ARB                                              0x8458
#define GL_COLOR_SUM_EXT                                              0x8458
#define GL_CURRENT_SECONDARY_COLOR                                    0x8459
#define GL_CURRENT_SECONDARY_COLOR_EXT                                0x8459
#define GL_SECONDARY_COLOR_ARRAY_SIZE                                 0x845A
#define GL_SECONDARY_COLOR_ARRAY_SIZE_EXT                             0x845A
#define GL_SECONDARY_COLOR_ARRAY_TYPE                                 0x845B
#define GL_SECONDARY_COLOR_ARRAY_TYPE_EXT                             0x845B
#define GL_SECONDARY_COLOR_ARRAY_STRIDE                               0x845C
#define GL_SECONDARY_COLOR_ARRAY_STRIDE_EXT                           0x845C
#define GL_SECONDARY_COLOR_ARRAY_POINTER                              0x845D
#define GL_SECONDARY_COLOR_ARRAY_POINTER_EXT                          0x845D
#define GL_SECONDARY_COLOR_ARRAY                                      0x845E
#define GL_SECONDARY_COLOR_ARRAY_EXT                                  0x845E
#define GL_CURRENT_RASTER_SECONDARY_COLOR                             0x845F
#define GL_ALIASED_POINT_SIZE_RANGE                                   0x846D
#define GL_ALIASED_LINE_WIDTH_RANGE                                   0x846E
#define GL_TEXTURE0                                                   0x84C0
#define GL_TEXTURE0_ARB                                               0x84C0
#define GL_TEXTURE1                                                   0x84C1
#define GL_TEXTURE1_ARB                                               0x84C1
#define GL_TEXTURE2                                                   0x84C2
#define GL_TEXTURE2_ARB                                               0x84C2
#define GL_TEXTURE3                                                   0x84C3
#define GL_TEXTURE3_ARB                                               0x84C3
#define GL_TEXTURE4                                                   0x84C4
#define GL_TEXTURE4_ARB                                               0x84C4
#define GL_TEXTURE5                                                   0x84C5
#define GL_TEXTURE5_ARB                                               0x84C5
#define GL_TEXTURE6                                                   0x84C6
#define GL_TEXTURE6_ARB                                               0x84C6
#define GL_TEXTURE7                                                   0x84C7
#define GL_TEXTURE7_ARB                                               0x84C7
#define GL_TEXTURE8                                                   0x84C8
#define GL_TEXTURE8_ARB                                               0x84C8
#define GL_TEXTURE9                                                   0x84C9
#define GL_TEXTURE9_ARB                                               0x84C9
#define GL_TEXTURE10                                                  0x84CA
#define GL_TEXTURE10_ARB                                              0x84CA
#define GL_TEXTURE11                                                  0x84CB
#define GL_TEXTURE11_ARB                                              0x84CB
#define GL_TEXTURE12                                                  0x84CC
#define GL_TEXTURE12_ARB                                              0x84CC
#define GL_TEXTURE13                                                  0x84CD
#define GL_TEXTURE13_ARB                                              0x84CD
#define GL_TEXTURE14                                                  0x84CE
#define GL_TEXTURE14_ARB                                              0x84CE
#define GL_TEXTURE15                                                  0x84CF
#define GL_TEXTURE15_ARB                                              0x84CF
#define GL_TEXTURE16                                                  0x84D0
#define GL_TEXTURE16_ARB                                              0x84D0
#define GL_TEXTURE17                                                  0x84D1
#define GL_TEXTURE17_ARB                                              0x84D1
#define GL_TEXTURE18                                                  0x84D2
#define GL_TEXTURE18_ARB                                              0x84D2
#define GL_TEXTURE19                                                  0x84D3
#define GL_TEXTURE19_ARB                                              0x84D3
#define GL_TEXTURE20                                                  0x84D4
#define GL_TEXTURE20_ARB                                              0x84D4
#define GL_TEXTURE21                                                  0x84D5
#define GL_TEXTURE21_ARB                                              0x84D5
#define GL_TEXTURE22                                                  0x84D6
#define GL_TEXTURE22_ARB                                              0x84D6
#define GL_TEXTURE23                                                  0x84D7
#define GL_TEXTURE23_ARB                                              0x84D7
#define GL_TEXTURE24                                                  0x84D8
#define GL_TEXTURE24_ARB                                              0x84D8
#define GL_TEXTURE25                                                  0x84D9
#define GL_TEXTURE25_ARB                                              0x84D9
#define GL_TEXTURE26                                                  0x84DA
#define GL_TEXTURE26_ARB                                              0x84DA
#define GL_TEXTURE27                                                  0x84DB
#define GL_TEXTURE27_ARB                                              0x84DB
#define GL_TEXTURE28                                                  0x84DC
#define GL_TEXTURE28_ARB                                              0x84DC
#define GL_TEXTURE29                                                  0x84DD
#define GL_TEXTURE29_ARB                                              0x84DD
#define GL_TEXTURE30                                                  0x84DE
#define GL_TEXTURE30_ARB                                              0x84DE
#define GL_TEXTURE31                                                  0x84DF
#define GL_TEXTURE31_ARB                                              0x84DF
#define GL_ACTIVE_TEXTURE                                             0x84E0
#define GL_ACTIVE_TEXTURE_ARB                                         0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE                                      0x84E1
#define GL_CLIENT_ACTIVE_TEXTURE_ARB                                  0x84E1
#define GL_MAX_TEXTURE_UNITS                                          0x84E2
#define GL_MAX_TEXTURE_UNITS_ARB                                      0x84E2
#define GL_TRANSPOSE_MODELVIEW_MATRIX                                 0x84E3
#define GL_TRANSPOSE_MODELVIEW_MATRIX_ARB                             0x84E3
#define GL_PATH_TRANSPOSE_MODELVIEW_MATRIX_NV                         0x84E3
#define GL_TRANSPOSE_PROJECTION_MATRIX                                0x84E4
#define GL_TRANSPOSE_PROJECTION_MATRIX_ARB                            0x84E4
#define GL_PATH_TRANSPOSE_PROJECTION_MATRIX_NV                        0x84E4
#define GL_TRANSPOSE_TEXTURE_MATRIX                                   0x84E5
#define GL_TRANSPOSE_TEXTURE_MATRIX_ARB                               0x84E5
#define GL_TRANSPOSE_COLOR_MATRIX                                     0x84E6
#define GL_TRANSPOSE_COLOR_MATRIX_ARB                                 0x84E6
#define GL_SUBTRACT                                                   0x84E7
#define GL_SUBTRACT_ARB                                               0x84E7
#define GL_MAX_RENDERBUFFER_SIZE                                      0x84E8
#define GL_MAX_RENDERBUFFER_SIZE_EXT                                  0x84E8
#define GL_COMPRESSED_ALPHA                                           0x84E9
#define GL_COMPRESSED_ALPHA_ARB                                       0x84E9
#define GL_COMPRESSED_LUMINANCE                                       0x84EA
#define GL_COMPRESSED_LUMINANCE_ARB                                   0x84EA
#define GL_COMPRESSED_LUMINANCE_ALPHA                                 0x84EB
#define GL_COMPRESSED_LUMINANCE_ALPHA_ARB                             0x84EB
#define GL_COMPRESSED_INTENSITY                                       0x84EC
#define GL_COMPRESSED_INTENSITY_ARB                                   0x84EC
#define GL_COMPRESSED_RGB                                             0x84ED
#define GL_COMPRESSED_RGB_ARB                                         0x84ED
#define GL_COMPRESSED_RGBA                                            0x84EE
#define GL_COMPRESSED_RGBA_ARB                                        0x84EE
#define GL_TEXTURE_COMPRESSION_HINT                                   0x84EF
#define GL_TEXTURE_COMPRESSION_HINT_ARB                               0x84EF
#define GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER            0x84F0
#define GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_EVALUATION_SHADER         0x84F1
#define GL_ALL_COMPLETED_NV                                           0x84F2
#define GL_FENCE_STATUS_NV                                            0x84F3
#define GL_FENCE_CONDITION_NV                                         0x84F4
#define GL_TEXTURE_RECTANGLE                                          0x84F5
#define GL_TEXTURE_RECTANGLE_ARB                                      0x84F5
#define GL_TEXTURE_RECTANGLE_NV                                       0x84F5
#define GL_TEXTURE_BINDING_RECTANGLE                                  0x84F6
#define GL_TEXTURE_BINDING_RECTANGLE_ARB                              0x84F6
#define GL_TEXTURE_BINDING_RECTANGLE_NV                               0x84F6
#define GL_PROXY_TEXTURE_RECTANGLE                                    0x84F7
#define GL_PROXY_TEXTURE_RECTANGLE_ARB                                0x84F7
#define GL_PROXY_TEXTURE_RECTANGLE_NV                                 0x84F7
#define GL_MAX_RECTANGLE_TEXTURE_SIZE                                 0x84F8
#define GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB                             0x84F8
#define GL_MAX_RECTANGLE_TEXTURE_SIZE_NV                              0x84F8
#define GL_DEPTH_STENCIL                                              0x84F9
#define GL_DEPTH_STENCIL_EXT                                          0x84F9
#define GL_DEPTH_STENCIL_NV                                           0x84F9
#define GL_UNSIGNED_INT_24_8                                          0x84FA
#define GL_UNSIGNED_INT_24_8_EXT                                      0x84FA
#define GL_UNSIGNED_INT_24_8_NV                                       0x84FA
#define GL_MAX_TEXTURE_LOD_BIAS                                       0x84FD
#define GL_MAX_TEXTURE_LOD_BIAS_EXT                                   0x84FD
#define GL_TEXTURE_MAX_ANISOTROPY                                     0x84FE
#define GL_TEXTURE_MAX_ANISOTROPY_EXT                                 0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY                                 0x84FF
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT                             0x84FF
#define GL_TEXTURE_FILTER_CONTROL                                     0x8500
#define GL_TEXTURE_FILTER_CONTROL_EXT                                 0x8500
#define GL_TEXTURE_LOD_BIAS                                           0x8501
#define GL_TEXTURE_LOD_BIAS_EXT                                       0x8501
#define GL_MODELVIEW1_STACK_DEPTH_EXT                                 0x8502
#define GL_COMBINE4_NV                                                0x8503
#define GL_MAX_SHININESS_NV                                           0x8504
#define GL_MAX_SPOT_EXPONENT_NV                                       0x8505
#define GL_MODELVIEW1_MATRIX_EXT                                      0x8506
#define GL_INCR_WRAP                                                  0x8507
#define GL_INCR_WRAP_EXT                                              0x8507
#define GL_DECR_WRAP                                                  0x8508
#define GL_DECR_WRAP_EXT                                              0x8508
#define GL_VERTEX_WEIGHTING_EXT                                       0x8509
#define GL_MODELVIEW1_ARB                                             0x850A
#define GL_MODELVIEW1_EXT                                             0x850A
#define GL_CURRENT_VERTEX_WEIGHT_EXT                                  0x850B
#define GL_VERTEX_WEIGHT_ARRAY_EXT                                    0x850C
#define GL_VERTEX_WEIGHT_ARRAY_SIZE_EXT                               0x850D
#define GL_VERTEX_WEIGHT_ARRAY_TYPE_EXT                               0x850E
#define GL_VERTEX_WEIGHT_ARRAY_STRIDE_EXT                             0x850F
#define GL_VERTEX_WEIGHT_ARRAY_POINTER_EXT                            0x8510
#define GL_NORMAL_MAP                                                 0x8511
#define GL_NORMAL_MAP_ARB                                             0x8511
#define GL_NORMAL_MAP_EXT                                             0x8511
#define GL_NORMAL_MAP_NV                                              0x8511
#define GL_REFLECTION_MAP                                             0x8512
#define GL_REFLECTION_MAP_ARB                                         0x8512
#define GL_REFLECTION_MAP_EXT                                         0x8512
#define GL_REFLECTION_MAP_NV                                          0x8512
#define GL_TEXTURE_CUBE_MAP                                           0x8513
#define GL_TEXTURE_CUBE_MAP_ARB                                       0x8513
#define GL_TEXTURE_CUBE_MAP_EXT                                       0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP                                   0x8514
#define GL_TEXTURE_BINDING_CUBE_MAP_ARB                               0x8514
#define GL_TEXTURE_BINDING_CUBE_MAP_EXT                               0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X                                0x8515
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB                            0x8515
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_EXT                            0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X                                0x8516
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB                            0x8516
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_EXT                            0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y                                0x8517
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB                            0x8517
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_EXT                            0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y                                0x8518
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB                            0x8518
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT                            0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z                                0x8519
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB                            0x8519
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_EXT                            0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z                                0x851A
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB                            0x851A
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT                            0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP                                     0x851B
#define GL_PROXY_TEXTURE_CUBE_MAP_ARB                                 0x851B
#define GL_PROXY_TEXTURE_CUBE_MAP_EXT                                 0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE                                  0x851C
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE_ARB                              0x851C
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE_EXT                              0x851C
#define GL_VERTEX_ARRAY_RANGE_APPLE                                   0x851D
#define GL_VERTEX_ARRAY_RANGE_NV                                      0x851D
#define GL_VERTEX_ARRAY_RANGE_LENGTH_APPLE                            0x851E
#define GL_VERTEX_ARRAY_RANGE_LENGTH_NV                               0x851E
#define GL_VERTEX_ARRAY_RANGE_VALID_NV                                0x851F
#define GL_VERTEX_ARRAY_STORAGE_HINT_APPLE                            0x851F
#define GL_MAX_VERTEX_ARRAY_RANGE_ELEMENT_NV                          0x8520
#define GL_VERTEX_ARRAY_RANGE_POINTER_APPLE                           0x8521
#define GL_VERTEX_ARRAY_RANGE_POINTER_NV                              0x8521
#define GL_REGISTER_COMBINERS_NV                                      0x8522
#define GL_VARIABLE_A_NV                                              0x8523
#define GL_VARIABLE_B_NV                                              0x8524
#define GL_VARIABLE_C_NV                                              0x8525
#define GL_VARIABLE_D_NV                                              0x8526
#define GL_VARIABLE_E_NV                                              0x8527
#define GL_VARIABLE_F_NV                                              0x8528
#define GL_VARIABLE_G_NV                                              0x8529
#define GL_CONSTANT_COLOR0_NV                                         0x852A
#define GL_CONSTANT_COLOR1_NV                                         0x852B
#define GL_PRIMARY_COLOR_NV                                           0x852C
#define GL_SECONDARY_COLOR_NV                                         0x852D
#define GL_SPARE0_NV                                                  0x852E
#define GL_SPARE1_NV                                                  0x852F
#define GL_DISCARD_NV                                                 0x8530
#define GL_E_TIMES_F_NV                                               0x8531
#define GL_SPARE0_PLUS_SECONDARY_COLOR_NV                             0x8532
#define GL_VERTEX_ARRAY_RANGE_WITHOUT_FLUSH_NV                        0x8533
#define GL_MULTISAMPLE_FILTER_HINT_NV                                 0x8534
#define GL_PER_STAGE_CONSTANTS_NV                                     0x8535
#define GL_UNSIGNED_IDENTITY_NV                                       0x8536
#define GL_UNSIGNED_INVERT_NV                                         0x8537
#define GL_EXPAND_NORMAL_NV                                           0x8538
#define GL_EXPAND_NEGATE_NV                                           0x8539
#define GL_HALF_BIAS_NORMAL_NV                                        0x853A
#define GL_HALF_BIAS_NEGATE_NV                                        0x853B
#define GL_SIGNED_IDENTITY_NV                                         0x853C
#define GL_SIGNED_NEGATE_NV                                           0x853D
#define GL_SCALE_BY_TWO_NV                                            0x853E
#define GL_SCALE_BY_FOUR_NV                                           0x853F
#define GL_SCALE_BY_ONE_HALF_NV                                       0x8540
#define GL_BIAS_BY_NEGATIVE_ONE_HALF_NV                               0x8541
#define GL_COMBINER_INPUT_NV                                          0x8542
#define GL_COMBINER_MAPPING_NV                                        0x8543
#define GL_COMBINER_COMPONENT_USAGE_NV                                0x8544
#define GL_COMBINER_AB_DOT_PRODUCT_NV                                 0x8545
#define GL_COMBINER_CD_DOT_PRODUCT_NV                                 0x8546
#define GL_COMBINER_MUX_SUM_NV                                        0x8547
#define GL_COMBINER_SCALE_NV                                          0x8548
#define GL_COMBINER_BIAS_NV                                           0x8549
#define GL_COMBINER_AB_OUTPUT_NV                                      0x854A
#define GL_COMBINER_CD_OUTPUT_NV                                      0x854B
#define GL_COMBINER_SUM_OUTPUT_NV                                     0x854C
#define GL_MAX_GENERAL_COMBINERS_NV                                   0x854D
#define GL_NUM_GENERAL_COMBINERS_NV                                   0x854E
#define GL_COLOR_SUM_CLAMP_NV                                         0x854F
#define GL_COMBINER0_NV                                               0x8550
#define GL_COMBINER1_NV                                               0x8551
#define GL_COMBINER2_NV                                               0x8552
#define GL_COMBINER3_NV                                               0x8553
#define GL_COMBINER4_NV                                               0x8554
#define GL_COMBINER5_NV                                               0x8555
#define GL_COMBINER6_NV                                               0x8556
#define GL_COMBINER7_NV                                               0x8557
#define GL_PRIMITIVE_RESTART_NV                                       0x8558
#define GL_PRIMITIVE_RESTART_INDEX_NV                                 0x8559
#define GL_FOG_DISTANCE_MODE_NV                                       0x855A
#define GL_EYE_RADIAL_NV                                              0x855B
#define GL_EYE_PLANE_ABSOLUTE_NV                                      0x855C
#define GL_EMBOSS_LIGHT_NV                                            0x855D
#define GL_EMBOSS_CONSTANT_NV                                         0x855E
#define GL_EMBOSS_MAP_NV                                              0x855F
#define GL_COMBINE                                                    0x8570
#define GL_COMBINE_ARB                                                0x8570
#define GL_COMBINE_EXT                                                0x8570
#define GL_COMBINE_RGB                                                0x8571
#define GL_COMBINE_RGB_ARB                                            0x8571
#define GL_COMBINE_RGB_EXT                                            0x8571
#define GL_COMBINE_ALPHA                                              0x8572
#define GL_COMBINE_ALPHA_ARB                                          0x8572
#define GL_COMBINE_ALPHA_EXT                                          0x8572
#define GL_RGB_SCALE                                                  0x8573
#define GL_RGB_SCALE_ARB                                              0x8573
#define GL_RGB_SCALE_EXT                                              0x8573
#define GL_ADD_SIGNED                                                 0x8574
#define GL_ADD_SIGNED_ARB                                             0x8574
#define GL_ADD_SIGNED_EXT                                             0x8574
#define GL_INTERPOLATE                                                0x8575
#define GL_INTERPOLATE_ARB                                            0x8575
#define GL_INTERPOLATE_EXT                                            0x8575
#define GL_CONSTANT                                                   0x8576
#define GL_CONSTANT_ARB                                               0x8576
#define GL_CONSTANT_EXT                                               0x8576
#define GL_CONSTANT_NV                                                0x8576
#define GL_PRIMARY_COLOR                                              0x8577
#define GL_PRIMARY_COLOR_ARB                                          0x8577
#define GL_PRIMARY_COLOR_EXT                                          0x8577
#define GL_PREVIOUS                                                   0x8578
#define GL_PREVIOUS_ARB                                               0x8578
#define GL_PREVIOUS_EXT                                               0x8578
#define GL_SOURCE0_RGB                                                0x8580
#define GL_SOURCE0_RGB_ARB                                            0x8580
#define GL_SOURCE0_RGB_EXT                                            0x8580
#define GL_SRC0_RGB                                                   0x8580
#define GL_SOURCE1_RGB                                                0x8581
#define GL_SOURCE1_RGB_ARB                                            0x8581
#define GL_SOURCE1_RGB_EXT                                            0x8581
#define GL_SRC1_RGB                                                   0x8581
#define GL_SOURCE2_RGB                                                0x8582
#define GL_SOURCE2_RGB_ARB                                            0x8582
#define GL_SOURCE2_RGB_EXT                                            0x8582
#define GL_SRC2_RGB                                                   0x8582
#define GL_SOURCE3_RGB_NV                                             0x8583
#define GL_SOURCE0_ALPHA                                              0x8588
#define GL_SOURCE0_ALPHA_ARB                                          0x8588
#define GL_SOURCE0_ALPHA_EXT                                          0x8588
#define GL_SRC0_ALPHA                                                 0x8588
#define GL_SOURCE1_ALPHA                                              0x8589
#define GL_SOURCE1_ALPHA_ARB                                          0x8589
#define GL_SOURCE1_ALPHA_EXT                                          0x8589
#define GL_SRC1_ALPHA                                                 0x8589
#define GL_SOURCE2_ALPHA                                              0x858A
#define GL_SOURCE2_ALPHA_ARB                                          0x858A
#define GL_SOURCE2_ALPHA_EXT                                          0x858A
#define GL_SRC2_ALPHA                                                 0x858A
#define GL_SOURCE3_ALPHA_NV                                           0x858B
#define GL_OPERAND0_RGB                                               0x8590
#define GL_OPERAND0_RGB_ARB                                           0x8590
#define GL_OPERAND0_RGB_EXT                                           0x8590
#define GL_OPERAND1_RGB                                               0x8591
#define GL_OPERAND1_RGB_ARB                                           0x8591
#define GL_OPERAND1_RGB_EXT                                           0x8591
#define GL_OPERAND2_RGB                                               0x8592
#define GL_OPERAND2_RGB_ARB                                           0x8592
#define GL_OPERAND2_RGB_EXT                                           0x8592
#define GL_OPERAND3_RGB_NV                                            0x8593
#define GL_OPERAND0_ALPHA                                             0x8598
#define GL_OPERAND0_ALPHA_ARB                                         0x8598
#define GL_OPERAND0_ALPHA_EXT                                         0x8598
#define GL_OPERAND1_ALPHA                                             0x8599
#define GL_OPERAND1_ALPHA_ARB                                         0x8599
#define GL_OPERAND1_ALPHA_EXT                                         0x8599
#define GL_OPERAND2_ALPHA                                             0x859A
#define GL_OPERAND2_ALPHA_ARB                                         0x859A
#define GL_OPERAND2_ALPHA_EXT                                         0x859A
#define GL_OPERAND3_ALPHA_NV                                          0x859B
#define GL_PERTURB_EXT                                                0x85AE
#define GL_TEXTURE_NORMAL_EXT                                         0x85AF
#define GL_LIGHT_MODEL_SPECULAR_VECTOR_APPLE                          0x85B0
#define GL_TRANSFORM_HINT_APPLE                                       0x85B1
#define GL_UNPACK_CLIENT_STORAGE_APPLE                                0x85B2
#define GL_BUFFER_OBJECT_APPLE                                        0x85B3
#define GL_STORAGE_CLIENT_APPLE                                       0x85B4
#define GL_VERTEX_ARRAY_BINDING                                       0x85B5
#define GL_VERTEX_ARRAY_BINDING_APPLE                                 0x85B5
#define GL_TEXTURE_RANGE_LENGTH_APPLE                                 0x85B7
#define GL_TEXTURE_RANGE_POINTER_APPLE                                0x85B8
#define GL_YCBCR_422_APPLE                                            0x85B9
#define GL_UNSIGNED_SHORT_8_8_APPLE                                   0x85BA
#define GL_UNSIGNED_SHORT_8_8_REV_APPLE                               0x85BB
#define GL_TEXTURE_STORAGE_HINT_APPLE                                 0x85BC
#define GL_STORAGE_PRIVATE_APPLE                                      0x85BD
#define GL_STORAGE_CACHED_APPLE                                       0x85BE
#define GL_STORAGE_SHARED_APPLE                                       0x85BF
#define GL_VERTEX_PROGRAM_ARB                                         0x8620
#define GL_VERTEX_PROGRAM_NV                                          0x8620
#define GL_VERTEX_STATE_PROGRAM_NV                                    0x8621
#define GL_VERTEX_ATTRIB_ARRAY_ENABLED                                0x8622
#define GL_VERTEX_ATTRIB_ARRAY_ENABLED_ARB                            0x8622
#define GL_ATTRIB_ARRAY_SIZE_NV                                       0x8623
#define GL_VERTEX_ATTRIB_ARRAY_SIZE                                   0x8623
#define GL_VERTEX_ATTRIB_ARRAY_SIZE_ARB                               0x8623
#define GL_ATTRIB_ARRAY_STRIDE_NV                                     0x8624
#define GL_VERTEX_ATTRIB_ARRAY_STRIDE                                 0x8624
#define GL_VERTEX_ATTRIB_ARRAY_STRIDE_ARB                             0x8624
#define GL_ATTRIB_ARRAY_TYPE_NV                                       0x8625
#define GL_VERTEX_ATTRIB_ARRAY_TYPE                                   0x8625
#define GL_VERTEX_ATTRIB_ARRAY_TYPE_ARB                               0x8625
#define GL_CURRENT_ATTRIB_NV                                          0x8626
#define GL_CURRENT_VERTEX_ATTRIB                                      0x8626
#define GL_CURRENT_VERTEX_ATTRIB_ARB                                  0x8626
#define GL_PROGRAM_LENGTH_ARB                                         0x8627
#define GL_PROGRAM_LENGTH_NV                                          0x8627
#define GL_PROGRAM_STRING_ARB                                         0x8628
#define GL_PROGRAM_STRING_NV                                          0x8628
#define GL_MODELVIEW_PROJECTION_NV                                    0x8629
#define GL_IDENTITY_NV                                                0x862A
#define GL_INVERSE_NV                                                 0x862B
#define GL_TRANSPOSE_NV                                               0x862C
#define GL_INVERSE_TRANSPOSE_NV                                       0x862D
#define GL_MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB                         0x862E
#define GL_MAX_TRACK_MATRIX_STACK_DEPTH_NV                            0x862E
#define GL_MAX_PROGRAM_MATRICES_ARB                                   0x862F
#define GL_MAX_TRACK_MATRICES_NV                                      0x862F
#define GL_MATRIX0_NV                                                 0x8630
#define GL_MATRIX1_NV                                                 0x8631
#define GL_MATRIX2_NV                                                 0x8632
#define GL_MATRIX3_NV                                                 0x8633
#define GL_MATRIX4_NV                                                 0x8634
#define GL_MATRIX5_NV                                                 0x8635
#define GL_MATRIX6_NV                                                 0x8636
#define GL_MATRIX7_NV                                                 0x8637
#define GL_CURRENT_MATRIX_STACK_DEPTH_ARB                             0x8640
#define GL_CURRENT_MATRIX_STACK_DEPTH_NV                              0x8640
#define GL_CURRENT_MATRIX_ARB                                         0x8641
#define GL_CURRENT_MATRIX_NV                                          0x8641
#define GL_VERTEX_PROGRAM_POINT_SIZE                                  0x8642
#define GL_VERTEX_PROGRAM_POINT_SIZE_ARB                              0x8642
#define GL_VERTEX_PROGRAM_POINT_SIZE_NV                               0x8642
#define GL_PROGRAM_POINT_SIZE                                         0x8642
#define GL_PROGRAM_POINT_SIZE_ARB                                     0x8642
#define GL_PROGRAM_POINT_SIZE_EXT                                     0x8642
#define GL_VERTEX_PROGRAM_TWO_SIDE                                    0x8643
#define GL_VERTEX_PROGRAM_TWO_SIDE_ARB                                0x8643
#define GL_VERTEX_PROGRAM_TWO_SIDE_NV                                 0x8643
#define GL_PROGRAM_PARAMETER_NV                                       0x8644
#define GL_ATTRIB_ARRAY_POINTER_NV                                    0x8645
#define GL_VERTEX_ATTRIB_ARRAY_POINTER                                0x8645
#define GL_VERTEX_ATTRIB_ARRAY_POINTER_ARB                            0x8645
#define GL_PROGRAM_TARGET_NV                                          0x8646
#define GL_PROGRAM_RESIDENT_NV                                        0x8647
#define GL_TRACK_MATRIX_NV                                            0x8648
#define GL_TRACK_MATRIX_TRANSFORM_NV                                  0x8649
#define GL_VERTEX_PROGRAM_BINDING_NV                                  0x864A
#define GL_PROGRAM_ERROR_POSITION_ARB                                 0x864B
#define GL_PROGRAM_ERROR_POSITION_NV                                  0x864B
#define GL_OFFSET_TEXTURE_RECTANGLE_NV                                0x864C
#define GL_OFFSET_TEXTURE_RECTANGLE_SCALE_NV                          0x864D
#define GL_DOT_PRODUCT_TEXTURE_RECTANGLE_NV                           0x864E
#define GL_DEPTH_CLAMP                                                0x864F
#define GL_DEPTH_CLAMP_NV                                             0x864F
#define GL_VERTEX_ATTRIB_ARRAY0_NV                                    0x8650
#define GL_VERTEX_ATTRIB_ARRAY1_NV                                    0x8651
#define GL_VERTEX_ATTRIB_ARRAY2_NV                                    0x8652
#define GL_VERTEX_ATTRIB_ARRAY3_NV                                    0x8653
#define GL_VERTEX_ATTRIB_ARRAY4_NV                                    0x8654
#define GL_VERTEX_ATTRIB_ARRAY5_NV                                    0x8655
#define GL_VERTEX_ATTRIB_ARRAY6_NV                                    0x8656
#define GL_VERTEX_ATTRIB_ARRAY7_NV                                    0x8657
#define GL_VERTEX_ATTRIB_ARRAY8_NV                                    0x8658
#define GL_VERTEX_ATTRIB_ARRAY9_NV                                    0x8659
#define GL_VERTEX_ATTRIB_ARRAY10_NV                                   0x865A
#define GL_VERTEX_ATTRIB_ARRAY11_NV                                   0x865B
#define GL_VERTEX_ATTRIB_ARRAY12_NV                                   0x865C
#define GL_VERTEX_ATTRIB_ARRAY13_NV                                   0x865D
#define GL_VERTEX_ATTRIB_ARRAY14_NV                                   0x865E
#define GL_VERTEX_ATTRIB_ARRAY15_NV                                   0x865F
#define GL_MAP1_VERTEX_ATTRIB0_4_NV                                   0x8660
#define GL_MAP1_VERTEX_ATTRIB1_4_NV                                   0x8661
#define GL_MAP1_VERTEX_ATTRIB2_4_NV                                   0x8662
#define GL_MAP1_VERTEX_ATTRIB3_4_NV                                   0x8663
#define GL_MAP1_VERTEX_ATTRIB4_4_NV                                   0x8664
#define GL_MAP1_VERTEX_ATTRIB5_4_NV                                   0x8665
#define GL_MAP1_VERTEX_ATTRIB6_4_NV                                   0x8666
#define GL_MAP1_VERTEX_ATTRIB7_4_NV                                   0x8667
#define GL_MAP1_VERTEX_ATTRIB8_4_NV                                   0x8668
#define GL_MAP1_VERTEX_ATTRIB9_4_NV                                   0x8669
#define GL_MAP1_VERTEX_ATTRIB10_4_NV                                  0x866A
#define GL_MAP1_VERTEX_ATTRIB11_4_NV                                  0x866B
#define GL_MAP1_VERTEX_ATTRIB12_4_NV                                  0x866C
#define GL_MAP1_VERTEX_ATTRIB13_4_NV                                  0x866D
#define GL_MAP1_VERTEX_ATTRIB14_4_NV                                  0x866E
#define GL_MAP1_VERTEX_ATTRIB15_4_NV                                  0x866F
#define GL_MAP2_VERTEX_ATTRIB0_4_NV                                   0x8670
#define GL_MAP2_VERTEX_ATTRIB1_4_NV                                   0x8671
#define GL_MAP2_VERTEX_ATTRIB2_4_NV                                   0x8672
#define GL_MAP2_VERTEX_ATTRIB3_4_NV                                   0x8673
#define GL_MAP2_VERTEX_ATTRIB4_4_NV                                   0x8674
#define GL_MAP2_VERTEX_ATTRIB5_4_NV                                   0x8675
#define GL_MAP2_VERTEX_ATTRIB6_4_NV                                   0x8676
#define GL_MAP2_VERTEX_ATTRIB7_4_NV                                   0x8677
#define GL_PROGRAM_BINDING_ARB                                        0x8677
#define GL_MAP2_VERTEX_ATTRIB8_4_NV                                   0x8678
#define GL_MAP2_VERTEX_ATTRIB9_4_NV                                   0x8679
#define GL_MAP2_VERTEX_ATTRIB10_4_NV                                  0x867A
#define GL_MAP2_VERTEX_ATTRIB11_4_NV                                  0x867B
#define GL_MAP2_VERTEX_ATTRIB12_4_NV                                  0x867C
#define GL_MAP2_VERTEX_ATTRIB13_4_NV                                  0x867D
#define GL_MAP2_VERTEX_ATTRIB14_4_NV                                  0x867E
#define GL_MAP2_VERTEX_ATTRIB15_4_NV                                  0x867F
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE                              0x86A0
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE_ARB                          0x86A0
#define GL_TEXTURE_COMPRESSED                                         0x86A1
#define GL_TEXTURE_COMPRESSED_ARB                                     0x86A1
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS                             0x86A2
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS_ARB                         0x86A2
#define GL_COMPRESSED_TEXTURE_FORMATS                                 0x86A3
#define GL_COMPRESSED_TEXTURE_FORMATS_ARB                             0x86A3
#define GL_MAX_VERTEX_UNITS_ARB                                       0x86A4
#define GL_ACTIVE_VERTEX_UNITS_ARB                                    0x86A5
#define GL_WEIGHT_SUM_UNITY_ARB                                       0x86A6
#define GL_VERTEX_BLEND_ARB                                           0x86A7
#define GL_CURRENT_WEIGHT_ARB                                         0x86A8
#define GL_WEIGHT_ARRAY_TYPE_ARB                                      0x86A9
#define GL_WEIGHT_ARRAY_STRIDE_ARB                                    0x86AA
#define GL_WEIGHT_ARRAY_SIZE_ARB                                      0x86AB
#define GL_WEIGHT_ARRAY_POINTER_ARB                                   0x86AC
#define GL_WEIGHT_ARRAY_ARB                                           0x86AD
#define GL_DOT3_RGB                                                   0x86AE
#define GL_DOT3_RGB_ARB                                               0x86AE
#define GL_DOT3_RGBA                                                  0x86AF
#define GL_DOT3_RGBA_ARB                                              0x86AF
#define GL_EVAL_2D_NV                                                 0x86C0
#define GL_EVAL_TRIANGULAR_2D_NV                                      0x86C1
#define GL_MAP_TESSELLATION_NV                                        0x86C2
#define GL_MAP_ATTRIB_U_ORDER_NV                                      0x86C3
#define GL_MAP_ATTRIB_V_ORDER_NV                                      0x86C4
#define GL_EVAL_FRACTIONAL_TESSELLATION_NV                            0x86C5
#define GL_EVAL_VERTEX_ATTRIB0_NV                                     0x86C6
#define GL_EVAL_VERTEX_ATTRIB1_NV                                     0x86C7
#define GL_EVAL_VERTEX_ATTRIB2_NV                                     0x86C8
#define GL_EVAL_VERTEX_ATTRIB3_NV                                     0x86C9
#define GL_EVAL_VERTEX_ATTRIB4_NV                                     0x86CA
#define GL_EVAL_VERTEX_ATTRIB5_NV                                     0x86CB
#define GL_EVAL_VERTEX_ATTRIB6_NV                                     0x86CC
#define GL_EVAL_VERTEX_ATTRIB7_NV                                     0x86CD
#define GL_EVAL_VERTEX_ATTRIB8_NV                                     0x86CE
#define GL_EVAL_VERTEX_ATTRIB9_NV                                     0x86CF
#define GL_EVAL_VERTEX_ATTRIB10_NV                                    0x86D0
#define GL_EVAL_VERTEX_ATTRIB11_NV                                    0x86D1
#define GL_EVAL_VERTEX_ATTRIB12_NV                                    0x86D2
#define GL_EVAL_VERTEX_ATTRIB13_NV                                    0x86D3
#define GL_EVAL_VERTEX_ATTRIB14_NV                                    0x86D4
#define GL_EVAL_VERTEX_ATTRIB15_NV                                    0x86D5
#define GL_MAX_MAP_TESSELLATION_NV                                    0x86D6
#define GL_MAX_RATIONAL_EVAL_ORDER_NV                                 0x86D7
#define GL_MAX_PROGRAM_PATCH_ATTRIBS_NV                               0x86D8
#define GL_RGBA_UNSIGNED_DOT_PRODUCT_MAPPING_NV                       0x86D9
#define GL_UNSIGNED_INT_S8_S8_8_8_NV                                  0x86DA
#define GL_UNSIGNED_INT_8_8_S8_S8_REV_NV                              0x86DB
#define GL_DSDT_MAG_INTENSITY_NV                                      0x86DC
#define GL_SHADER_CONSISTENT_NV                                       0x86DD
#define GL_TEXTURE_SHADER_NV                                          0x86DE
#define GL_SHADER_OPERATION_NV                                        0x86DF
#define GL_CULL_MODES_NV                                              0x86E0
#define GL_OFFSET_TEXTURE_MATRIX_NV                                   0x86E1
#define GL_OFFSET_TEXTURE_2D_MATRIX_NV                                0x86E1
#define GL_OFFSET_TEXTURE_SCALE_NV                                    0x86E2
#define GL_OFFSET_TEXTURE_2D_SCALE_NV                                 0x86E2
#define GL_OFFSET_TEXTURE_BIAS_NV                                     0x86E3
#define GL_OFFSET_TEXTURE_2D_BIAS_NV                                  0x86E3
#define GL_PREVIOUS_TEXTURE_INPUT_NV                                  0x86E4
#define GL_CONST_EYE_NV                                               0x86E5
#define GL_PASS_THROUGH_NV                                            0x86E6
#define GL_CULL_FRAGMENT_NV                                           0x86E7
#define GL_OFFSET_TEXTURE_2D_NV                                       0x86E8
#define GL_DEPENDENT_AR_TEXTURE_2D_NV                                 0x86E9
#define GL_DEPENDENT_GB_TEXTURE_2D_NV                                 0x86EA
#define GL_SURFACE_STATE_NV                                           0x86EB
#define GL_DOT_PRODUCT_NV                                             0x86EC
#define GL_DOT_PRODUCT_DEPTH_REPLACE_NV                               0x86ED
#define GL_DOT_PRODUCT_TEXTURE_2D_NV                                  0x86EE
#define GL_DOT_PRODUCT_TEXTURE_3D_NV                                  0x86EF
#define GL_DOT_PRODUCT_TEXTURE_CUBE_MAP_NV                            0x86F0
#define GL_DOT_PRODUCT_DIFFUSE_CUBE_MAP_NV                            0x86F1
#define GL_DOT_PRODUCT_REFLECT_CUBE_MAP_NV                            0x86F2
#define GL_DOT_PRODUCT_CONST_EYE_REFLECT_CUBE_MAP_NV                  0x86F3
#define GL_HILO_NV                                                    0x86F4
#define GL_DSDT_NV                                                    0x86F5
#define GL_DSDT_MAG_NV                                                0x86F6
#define GL_DSDT_MAG_VIB_NV                                            0x86F7
#define GL_HILO16_NV                                                  0x86F8
#define GL_SIGNED_HILO_NV                                             0x86F9
#define GL_SIGNED_HILO16_NV                                           0x86FA
#define GL_SIGNED_RGBA_NV                                             0x86FB
#define GL_SIGNED_RGBA8_NV                                            0x86FC
#define GL_SURFACE_REGISTERED_NV                                      0x86FD
#define GL_SIGNED_RGB_NV                                              0x86FE
#define GL_SIGNED_RGB8_NV                                             0x86FF
#define GL_SURFACE_MAPPED_NV                                          0x8700
#define GL_SIGNED_LUMINANCE_NV                                        0x8701
#define GL_SIGNED_LUMINANCE8_NV                                       0x8702
#define GL_SIGNED_LUMINANCE_ALPHA_NV                                  0x8703
#define GL_SIGNED_LUMINANCE8_ALPHA8_NV                                0x8704
#define GL_SIGNED_ALPHA_NV                                            0x8705
#define GL_SIGNED_ALPHA8_NV                                           0x8706
#define GL_SIGNED_INTENSITY_NV                                        0x8707
#define GL_SIGNED_INTENSITY8_NV                                       0x8708
#define GL_DSDT8_NV                                                   0x8709
#define GL_DSDT8_MAG8_NV                                              0x870A
#define GL_DSDT8_MAG8_INTENSITY8_NV                                   0x870B
#define GL_SIGNED_RGB_UNSIGNED_ALPHA_NV                               0x870C
#define GL_SIGNED_RGB8_UNSIGNED_ALPHA8_NV                             0x870D
#define GL_HI_SCALE_NV                                                0x870E
#define GL_LO_SCALE_NV                                                0x870F
#define GL_DS_SCALE_NV                                                0x8710
#define GL_DT_SCALE_NV                                                0x8711
#define GL_MAGNITUDE_SCALE_NV                                         0x8712
#define GL_VIBRANCE_SCALE_NV                                          0x8713
#define GL_HI_BIAS_NV                                                 0x8714
#define GL_LO_BIAS_NV                                                 0x8715
#define GL_DS_BIAS_NV                                                 0x8716
#define GL_DT_BIAS_NV                                                 0x8717
#define GL_MAGNITUDE_BIAS_NV                                          0x8718
#define GL_VIBRANCE_BIAS_NV                                           0x8719
#define GL_TEXTURE_BORDER_VALUES_NV                                   0x871A
#define GL_TEXTURE_HI_SIZE_NV                                         0x871B
#define GL_TEXTURE_LO_SIZE_NV                                         0x871C
#define GL_TEXTURE_DS_SIZE_NV                                         0x871D
#define GL_TEXTURE_DT_SIZE_NV                                         0x871E
#define GL_TEXTURE_MAG_SIZE_NV                                        0x871F
#define GL_MODELVIEW2_ARB                                             0x8722
#define GL_MODELVIEW3_ARB                                             0x8723
#define GL_MODELVIEW4_ARB                                             0x8724
#define GL_MODELVIEW5_ARB                                             0x8725
#define GL_MODELVIEW6_ARB                                             0x8726
#define GL_MODELVIEW7_ARB                                             0x8727
#define GL_MODELVIEW8_ARB                                             0x8728
#define GL_MODELVIEW9_ARB                                             0x8729
#define GL_MODELVIEW10_ARB                                            0x872A
#define GL_MODELVIEW11_ARB                                            0x872B
#define GL_MODELVIEW12_ARB                                            0x872C
#define GL_MODELVIEW13_ARB                                            0x872D
#define GL_MODELVIEW14_ARB                                            0x872E
#define GL_MODELVIEW15_ARB                                            0x872F
#define GL_MODELVIEW16_ARB                                            0x8730
#define GL_MODELVIEW17_ARB                                            0x8731
#define GL_MODELVIEW18_ARB                                            0x8732
#define GL_MODELVIEW19_ARB                                            0x8733
#define GL_MODELVIEW20_ARB                                            0x8734
#define GL_MODELVIEW21_ARB                                            0x8735
#define GL_MODELVIEW22_ARB                                            0x8736
#define GL_MODELVIEW23_ARB                                            0x8737
#define GL_MODELVIEW24_ARB                                            0x8738
#define GL_MODELVIEW25_ARB                                            0x8739
#define GL_MODELVIEW26_ARB                                            0x873A
#define GL_MODELVIEW27_ARB                                            0x873B
#define GL_MODELVIEW28_ARB                                            0x873C
#define GL_MODELVIEW29_ARB                                            0x873D
#define GL_MODELVIEW30_ARB                                            0x873E
#define GL_MODELVIEW31_ARB                                            0x873F
#define GL_DOT3_RGB_EXT                                               0x8740
#define GL_DOT3_RGBA_EXT                                              0x8741
#define GL_PROGRAM_BINARY_LENGTH                                      0x8741
#define GL_MIRROR_CLAMP_EXT                                           0x8742
#define GL_MIRROR_CLAMP_TO_EDGE                                       0x8743
#define GL_MIRROR_CLAMP_TO_EDGE_EXT                                   0x8743
#define GL_SET_AMD                                                    0x874A
#define GL_REPLACE_VALUE_AMD                                          0x874B
#define GL_STENCIL_OP_VALUE_AMD                                       0x874C
#define GL_STENCIL_BACK_OP_VALUE_AMD                                  0x874D
#define GL_VERTEX_ATTRIB_ARRAY_LONG                                   0x874E
#define GL_OCCLUSION_QUERY_EVENT_MASK_AMD                             0x874F
#define GL_BUFFER_SIZE                                                0x8764
#define GL_BUFFER_SIZE_ARB                                            0x8764
#define GL_BUFFER_USAGE                                               0x8765
#define GL_BUFFER_USAGE_ARB                                           0x8765
#define GL_VERTEX_SHADER_EXT                                          0x8780
#define GL_VERTEX_SHADER_BINDING_EXT                                  0x8781
#define GL_OP_INDEX_EXT                                               0x8782
#define GL_OP_NEGATE_EXT                                              0x8783
#define GL_OP_DOT3_EXT                                                0x8784
#define GL_OP_DOT4_EXT                                                0x8785
#define GL_OP_MUL_EXT                                                 0x8786
#define GL_OP_ADD_EXT                                                 0x8787
#define GL_OP_MADD_EXT                                                0x8788
#define GL_OP_FRAC_EXT                                                0x8789
#define GL_OP_MAX_EXT                                                 0x878A
#define GL_OP_MIN_EXT                                                 0x878B
#define GL_OP_SET_GE_EXT                                              0x878C
#define GL_OP_SET_LT_EXT                                              0x878D
#define GL_OP_CLAMP_EXT                                               0x878E
#define GL_OP_FLOOR_EXT                                               0x878F
#define GL_OP_ROUND_EXT                                               0x8790
#define GL_OP_EXP_BASE_2_EXT                                          0x8791
#define GL_OP_LOG_BASE_2_EXT                                          0x8792
#define GL_OP_POWER_EXT                                               0x8793
#define GL_OP_RECIP_EXT                                               0x8794
#define GL_OP_RECIP_SQRT_EXT                                          0x8795
#define GL_OP_SUB_EXT                                                 0x8796
#define GL_OP_CROSS_PRODUCT_EXT                                       0x8797
#define GL_OP_MULTIPLY_MATRIX_EXT                                     0x8798
#define GL_OP_MOV_EXT                                                 0x8799
#define GL_OUTPUT_VERTEX_EXT                                          0x879A
#define GL_OUTPUT_COLOR0_EXT                                          0x879B
#define GL_OUTPUT_COLOR1_EXT                                          0x879C
#define GL_OUTPUT_TEXTURE_COORD0_EXT                                  0x879D
#define GL_OUTPUT_TEXTURE_COORD1_EXT                                  0x879E
#define GL_OUTPUT_TEXTURE_COORD2_EXT                                  0x879F
#define GL_OUTPUT_TEXTURE_COORD3_EXT                                  0x87A0
#define GL_OUTPUT_TEXTURE_COORD4_EXT                                  0x87A1
#define GL_OUTPUT_TEXTURE_COORD5_EXT                                  0x87A2
#define GL_OUTPUT_TEXTURE_COORD6_EXT                                  0x87A3
#define GL_OUTPUT_TEXTURE_COORD7_EXT                                  0x87A4
#define GL_OUTPUT_TEXTURE_COORD8_EXT                                  0x87A5
#define GL_OUTPUT_TEXTURE_COORD9_EXT                                  0x87A6
#define GL_OUTPUT_TEXTURE_COORD10_EXT                                 0x87A7
#define GL_OUTPUT_TEXTURE_COORD11_EXT                                 0x87A8
#define GL_OUTPUT_TEXTURE_COORD12_EXT                                 0x87A9
#define GL_OUTPUT_TEXTURE_COORD13_EXT                                 0x87AA
#define GL_OUTPUT_TEXTURE_COORD14_EXT                                 0x87AB
#define GL_OUTPUT_TEXTURE_COORD15_EXT                                 0x87AC
#define GL_OUTPUT_TEXTURE_COORD16_EXT                                 0x87AD
#define GL_OUTPUT_TEXTURE_COORD17_EXT                                 0x87AE
#define GL_OUTPUT_TEXTURE_COORD18_EXT                                 0x87AF
#define GL_OUTPUT_TEXTURE_COORD19_EXT                                 0x87B0
#define GL_OUTPUT_TEXTURE_COORD20_EXT                                 0x87B1
#define GL_OUTPUT_TEXTURE_COORD21_EXT                                 0x87B2
#define GL_OUTPUT_TEXTURE_COORD22_EXT                                 0x87B3
#define GL_OUTPUT_TEXTURE_COORD23_EXT                                 0x87B4
#define GL_OUTPUT_TEXTURE_COORD24_EXT                                 0x87B5
#define GL_OUTPUT_TEXTURE_COORD25_EXT                                 0x87B6
#define GL_OUTPUT_TEXTURE_COORD26_EXT                                 0x87B7
#define GL_OUTPUT_TEXTURE_COORD27_EXT                                 0x87B8
#define GL_OUTPUT_TEXTURE_COORD28_EXT                                 0x87B9
#define GL_OUTPUT_TEXTURE_COORD29_EXT                                 0x87BA
#define GL_OUTPUT_TEXTURE_COORD30_EXT                                 0x87BB
#define GL_OUTPUT_TEXTURE_COORD31_EXT                                 0x87BC
#define GL_OUTPUT_FOG_EXT                                             0x87BD
#define GL_SCALAR_EXT                                                 0x87BE
#define GL_VECTOR_EXT                                                 0x87BF
#define GL_MATRIX_EXT                                                 0x87C0
#define GL_VARIANT_EXT                                                0x87C1
#define GL_INVARIANT_EXT                                              0x87C2
#define GL_LOCAL_CONSTANT_EXT                                         0x87C3
#define GL_LOCAL_EXT                                                  0x87C4
#define GL_MAX_VERTEX_SHADER_INSTRUCTIONS_EXT                         0x87C5
#define GL_MAX_VERTEX_SHADER_VARIANTS_EXT                             0x87C6
#define GL_MAX_VERTEX_SHADER_INVARIANTS_EXT                           0x87C7
#define GL_MAX_VERTEX_SHADER_LOCAL_CONSTANTS_EXT                      0x87C8
#define GL_MAX_VERTEX_SHADER_LOCALS_EXT                               0x87C9
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_INSTRUCTIONS_EXT               0x87CA
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_VARIANTS_EXT                   0x87CB
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_LOCAL_CONSTANTS_EXT            0x87CC
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_INVARIANTS_EXT                 0x87CD
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_LOCALS_EXT                     0x87CE
#define GL_VERTEX_SHADER_INSTRUCTIONS_EXT                             0x87CF
#define GL_VERTEX_SHADER_VARIANTS_EXT                                 0x87D0
#define GL_VERTEX_SHADER_INVARIANTS_EXT                               0x87D1
#define GL_VERTEX_SHADER_LOCAL_CONSTANTS_EXT                          0x87D2
#define GL_VERTEX_SHADER_LOCALS_EXT                                   0x87D3
#define GL_VERTEX_SHADER_OPTIMIZED_EXT                                0x87D4
#define GL_X_EXT                                                      0x87D5
#define GL_Y_EXT                                                      0x87D6
#define GL_Z_EXT                                                      0x87D7
#define GL_W_EXT                                                      0x87D8
#define GL_NEGATIVE_X_EXT                                             0x87D9
#define GL_NEGATIVE_Y_EXT                                             0x87DA
#define GL_NEGATIVE_Z_EXT                                             0x87DB
#define GL_NEGATIVE_W_EXT                                             0x87DC
#define GL_ZERO_EXT                                                   0x87DD
#define GL_ONE_EXT                                                    0x87DE
#define GL_NEGATIVE_ONE_EXT                                           0x87DF
#define GL_NORMALIZED_RANGE_EXT                                       0x87E0
#define GL_FULL_RANGE_EXT                                             0x87E1
#define GL_CURRENT_VERTEX_EXT                                         0x87E2
#define GL_MVP_MATRIX_EXT                                             0x87E3
#define GL_VARIANT_VALUE_EXT                                          0x87E4
#define GL_VARIANT_DATATYPE_EXT                                       0x87E5
#define GL_VARIANT_ARRAY_STRIDE_EXT                                   0x87E6
#define GL_VARIANT_ARRAY_TYPE_EXT                                     0x87E7
#define GL_VARIANT_ARRAY_EXT                                          0x87E8
#define GL_VARIANT_ARRAY_POINTER_EXT                                  0x87E9
#define GL_INVARIANT_VALUE_EXT                                        0x87EA
#define GL_INVARIANT_DATATYPE_EXT                                     0x87EB
#define GL_LOCAL_CONSTANT_VALUE_EXT                                   0x87EC
#define GL_LOCAL_CONSTANT_DATATYPE_EXT                                0x87ED
#define GL_NUM_PROGRAM_BINARY_FORMATS                                 0x87FE
#define GL_PROGRAM_BINARY_FORMATS                                     0x87FF
#define GL_STENCIL_BACK_FUNC                                          0x8800
#define GL_STENCIL_BACK_FAIL                                          0x8801
#define GL_STENCIL_BACK_PASS_DEPTH_FAIL                               0x8802
#define GL_STENCIL_BACK_PASS_DEPTH_PASS                               0x8803
#define GL_FRAGMENT_PROGRAM_ARB                                       0x8804
#define GL_PROGRAM_ALU_INSTRUCTIONS_ARB                               0x8805
#define GL_PROGRAM_TEX_INSTRUCTIONS_ARB                               0x8806
#define GL_PROGRAM_TEX_INDIRECTIONS_ARB                               0x8807
#define GL_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB                        0x8808
#define GL_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB                        0x8809
#define GL_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB                        0x880A
#define GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB                           0x880B
#define GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB                           0x880C
#define GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB                           0x880D
#define GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB                    0x880E
#define GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB                    0x880F
#define GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB                    0x8810
#define GL_RGBA32F                                                    0x8814
#define GL_RGBA32F_ARB                                                0x8814
#define GL_RGBA_FLOAT32_APPLE                                         0x8814
#define GL_RGB32F                                                     0x8815
#define GL_RGB32F_ARB                                                 0x8815
#define GL_RGB_FLOAT32_APPLE                                          0x8815
#define GL_ALPHA32F_ARB                                               0x8816
#define GL_ALPHA_FLOAT32_APPLE                                        0x8816
#define GL_INTENSITY32F_ARB                                           0x8817
#define GL_INTENSITY_FLOAT32_APPLE                                    0x8817
#define GL_LUMINANCE32F_ARB                                           0x8818
#define GL_LUMINANCE_FLOAT32_APPLE                                    0x8818
#define GL_LUMINANCE_ALPHA32F_ARB                                     0x8819
#define GL_LUMINANCE_ALPHA_FLOAT32_APPLE                              0x8819
#define GL_RGBA16F                                                    0x881A
#define GL_RGBA16F_ARB                                                0x881A
#define GL_RGBA_FLOAT16_APPLE                                         0x881A
#define GL_RGB16F                                                     0x881B
#define GL_RGB16F_ARB                                                 0x881B
#define GL_RGB_FLOAT16_APPLE                                          0x881B
#define GL_ALPHA16F_ARB                                               0x881C
#define GL_ALPHA_FLOAT16_APPLE                                        0x881C
#define GL_INTENSITY16F_ARB                                           0x881D
#define GL_INTENSITY_FLOAT16_APPLE                                    0x881D
#define GL_LUMINANCE16F_ARB                                           0x881E
#define GL_LUMINANCE_FLOAT16_APPLE                                    0x881E
#define GL_LUMINANCE_ALPHA16F_ARB                                     0x881F
#define GL_LUMINANCE_ALPHA_FLOAT16_APPLE                              0x881F
#define GL_RGBA_FLOAT_MODE_ARB                                        0x8820
#define GL_MAX_DRAW_BUFFERS                                           0x8824
#define GL_MAX_DRAW_BUFFERS_ARB                                       0x8824
#define GL_DRAW_BUFFER0                                               0x8825
#define GL_DRAW_BUFFER0_ARB                                           0x8825
#define GL_DRAW_BUFFER1                                               0x8826
#define GL_DRAW_BUFFER1_ARB                                           0x8826
#define GL_DRAW_BUFFER2                                               0x8827
#define GL_DRAW_BUFFER2_ARB                                           0x8827
#define GL_DRAW_BUFFER3                                               0x8828
#define GL_DRAW_BUFFER3_ARB                                           0x8828
#define GL_DRAW_BUFFER4                                               0x8829
#define GL_DRAW_BUFFER4_ARB                                           0x8829
#define GL_DRAW_BUFFER5                                               0x882A
#define GL_DRAW_BUFFER5_ARB                                           0x882A
#define GL_DRAW_BUFFER6                                               0x882B
#define GL_DRAW_BUFFER6_ARB                                           0x882B
#define GL_DRAW_BUFFER7                                               0x882C
#define GL_DRAW_BUFFER7_ARB                                           0x882C
#define GL_DRAW_BUFFER8                                               0x882D
#define GL_DRAW_BUFFER8_ARB                                           0x882D
#define GL_DRAW_BUFFER9                                               0x882E
#define GL_DRAW_BUFFER9_ARB                                           0x882E
#define GL_DRAW_BUFFER10                                              0x882F
#define GL_DRAW_BUFFER10_ARB                                          0x882F
#define GL_DRAW_BUFFER11                                              0x8830
#define GL_DRAW_BUFFER11_ARB                                          0x8830
#define GL_DRAW_BUFFER12                                              0x8831
#define GL_DRAW_BUFFER12_ARB                                          0x8831
#define GL_DRAW_BUFFER13                                              0x8832
#define GL_DRAW_BUFFER13_ARB                                          0x8832
#define GL_DRAW_BUFFER14                                              0x8833
#define GL_DRAW_BUFFER14_ARB                                          0x8833
#define GL_DRAW_BUFFER15                                              0x8834
#define GL_DRAW_BUFFER15_ARB                                          0x8834
#define GL_BLEND_EQUATION_ALPHA                                       0x883D
#define GL_BLEND_EQUATION_ALPHA_EXT                                   0x883D
#define GL_SUBSAMPLE_DISTANCE_AMD                                     0x883F
#define GL_MATRIX_PALETTE_ARB                                         0x8840
#define GL_MAX_MATRIX_PALETTE_STACK_DEPTH_ARB                         0x8841
#define GL_MAX_PALETTE_MATRICES_ARB                                   0x8842
#define GL_CURRENT_PALETTE_MATRIX_ARB                                 0x8843
#define GL_MATRIX_INDEX_ARRAY_ARB                                     0x8844
#define GL_CURRENT_MATRIX_INDEX_ARB                                   0x8845
#define GL_MATRIX_INDEX_ARRAY_SIZE_ARB                                0x8846
#define GL_MATRIX_INDEX_ARRAY_TYPE_ARB                                0x8847
#define GL_MATRIX_INDEX_ARRAY_STRIDE_ARB                              0x8848
#define GL_MATRIX_INDEX_ARRAY_POINTER_ARB                             0x8849
#define GL_TEXTURE_DEPTH_SIZE                                         0x884A
#define GL_TEXTURE_DEPTH_SIZE_ARB                                     0x884A
#define GL_DEPTH_TEXTURE_MODE                                         0x884B
#define GL_DEPTH_TEXTURE_MODE_ARB                                     0x884B
#define GL_TEXTURE_COMPARE_MODE                                       0x884C
#define GL_TEXTURE_COMPARE_MODE_ARB                                   0x884C
#define GL_TEXTURE_COMPARE_FUNC                                       0x884D
#define GL_TEXTURE_COMPARE_FUNC_ARB                                   0x884D
#define GL_COMPARE_R_TO_TEXTURE                                       0x884E
#define GL_COMPARE_R_TO_TEXTURE_ARB                                   0x884E
#define GL_COMPARE_REF_DEPTH_TO_TEXTURE_EXT                           0x884E
#define GL_COMPARE_REF_TO_TEXTURE                                     0x884E
#define GL_TEXTURE_CUBE_MAP_SEAMLESS                                  0x884F
#define GL_OFFSET_PROJECTIVE_TEXTURE_2D_NV                            0x8850
#define GL_OFFSET_PROJECTIVE_TEXTURE_2D_SCALE_NV                      0x8851
#define GL_OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_NV                     0x8852
#define GL_OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_SCALE_NV               0x8853
#define GL_OFFSET_HILO_TEXTURE_2D_NV                                  0x8854
#define GL_OFFSET_HILO_TEXTURE_RECTANGLE_NV                           0x8855
#define GL_OFFSET_HILO_PROJECTIVE_TEXTURE_2D_NV                       0x8856
#define GL_OFFSET_HILO_PROJECTIVE_TEXTURE_RECTANGLE_NV                0x8857
#define GL_DEPENDENT_HILO_TEXTURE_2D_NV                               0x8858
#define GL_DEPENDENT_RGB_TEXTURE_3D_NV                                0x8859
#define GL_DEPENDENT_RGB_TEXTURE_CUBE_MAP_NV                          0x885A
#define GL_DOT_PRODUCT_PASS_THROUGH_NV                                0x885B
#define GL_DOT_PRODUCT_TEXTURE_1D_NV                                  0x885C
#define GL_DOT_PRODUCT_AFFINE_DEPTH_REPLACE_NV                        0x885D
#define GL_HILO8_NV                                                   0x885E
#define GL_SIGNED_HILO8_NV                                            0x885F
#define GL_FORCE_BLUE_TO_ONE_NV                                       0x8860
#define GL_POINT_SPRITE                                               0x8861
#define GL_POINT_SPRITE_ARB                                           0x8861
#define GL_POINT_SPRITE_NV                                            0x8861
#define GL_COORD_REPLACE                                              0x8862
#define GL_COORD_REPLACE_ARB                                          0x8862
#define GL_COORD_REPLACE_NV                                           0x8862
#define GL_POINT_SPRITE_R_MODE_NV                                     0x8863
#define GL_PIXEL_COUNTER_BITS_NV                                      0x8864
#define GL_QUERY_COUNTER_BITS                                         0x8864
#define GL_QUERY_COUNTER_BITS_ARB                                     0x8864
#define GL_CURRENT_OCCLUSION_QUERY_ID_NV                              0x8865
#define GL_CURRENT_QUERY                                              0x8865
#define GL_CURRENT_QUERY_ARB                                          0x8865
#define GL_PIXEL_COUNT_NV                                             0x8866
#define GL_QUERY_RESULT                                               0x8866
#define GL_QUERY_RESULT_ARB                                           0x8866
#define GL_PIXEL_COUNT_AVAILABLE_NV                                   0x8867
#define GL_QUERY_RESULT_AVAILABLE                                     0x8867
#define GL_QUERY_RESULT_AVAILABLE_ARB                                 0x8867
#define GL_MAX_FRAGMENT_PROGRAM_LOCAL_PARAMETERS_NV                   0x8868
#define GL_MAX_VERTEX_ATTRIBS                                         0x8869
#define GL_MAX_VERTEX_ATTRIBS_ARB                                     0x8869
#define GL_VERTEX_ATTRIB_ARRAY_NORMALIZED                             0x886A
#define GL_VERTEX_ATTRIB_ARRAY_NORMALIZED_ARB                         0x886A
#define GL_MAX_TESS_CONTROL_INPUT_COMPONENTS                          0x886C
#define GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS                       0x886D
#define GL_DEPTH_STENCIL_TO_RGBA_NV                                   0x886E
#define GL_DEPTH_STENCIL_TO_BGRA_NV                                   0x886F
#define GL_FRAGMENT_PROGRAM_NV                                        0x8870
#define GL_MAX_TEXTURE_COORDS                                         0x8871
#define GL_MAX_TEXTURE_COORDS_ARB                                     0x8871
#define GL_MAX_TEXTURE_COORDS_NV                                      0x8871
#define GL_MAX_TEXTURE_IMAGE_UNITS                                    0x8872
#define GL_MAX_TEXTURE_IMAGE_UNITS_ARB                                0x8872
#define GL_MAX_TEXTURE_IMAGE_UNITS_NV                                 0x8872
#define GL_FRAGMENT_PROGRAM_BINDING_NV                                0x8873
#define GL_PROGRAM_ERROR_STRING_ARB                                   0x8874
#define GL_PROGRAM_ERROR_STRING_NV                                    0x8874
#define GL_PROGRAM_FORMAT_ASCII_ARB                                   0x8875
#define GL_PROGRAM_FORMAT_ARB                                         0x8876
#define GL_WRITE_PIXEL_DATA_RANGE_NV                                  0x8878
#define GL_READ_PIXEL_DATA_RANGE_NV                                   0x8879
#define GL_WRITE_PIXEL_DATA_RANGE_LENGTH_NV                           0x887A
#define GL_READ_PIXEL_DATA_RANGE_LENGTH_NV                            0x887B
#define GL_WRITE_PIXEL_DATA_RANGE_POINTER_NV                          0x887C
#define GL_READ_PIXEL_DATA_RANGE_POINTER_NV                           0x887D
#define GL_GEOMETRY_SHADER_INVOCATIONS                                0x887F
#define GL_FLOAT_R_NV                                                 0x8880
#define GL_FLOAT_RG_NV                                                0x8881
#define GL_FLOAT_RGB_NV                                               0x8882
#define GL_FLOAT_RGBA_NV                                              0x8883
#define GL_FLOAT_R16_NV                                               0x8884
#define GL_FLOAT_R32_NV                                               0x8885
#define GL_FLOAT_RG16_NV                                              0x8886
#define GL_FLOAT_RG32_NV                                              0x8887
#define GL_FLOAT_RGB16_NV                                             0x8888
#define GL_FLOAT_RGB32_NV                                             0x8889
#define GL_FLOAT_RGBA16_NV                                            0x888A
#define GL_FLOAT_RGBA32_NV                                            0x888B
#define GL_TEXTURE_FLOAT_COMPONENTS_NV                                0x888C
#define GL_FLOAT_CLEAR_COLOR_VALUE_NV                                 0x888D
#define GL_FLOAT_RGBA_MODE_NV                                         0x888E
#define GL_TEXTURE_UNSIGNED_REMAP_MODE_NV                             0x888F
#define GL_DEPTH_BOUNDS_TEST_EXT                                      0x8890
#define GL_DEPTH_BOUNDS_EXT                                           0x8891
#define GL_ARRAY_BUFFER                                               0x8892
#define GL_ARRAY_BUFFER_ARB                                           0x8892
#define GL_ELEMENT_ARRAY_BUFFER                                       0x8893
#define GL_ELEMENT_ARRAY_BUFFER_ARB                                   0x8893
#define GL_ARRAY_BUFFER_BINDING                                       0x8894
#define GL_ARRAY_BUFFER_BINDING_ARB                                   0x8894
#define GL_ELEMENT_ARRAY_BUFFER_BINDING                               0x8895
#define GL_ELEMENT_ARRAY_BUFFER_BINDING_ARB                           0x8895
#define GL_VERTEX_ARRAY_BUFFER_BINDING                                0x8896
#define GL_VERTEX_ARRAY_BUFFER_BINDING_ARB                            0x8896
#define GL_NORMAL_ARRAY_BUFFER_BINDING                                0x8897
#define GL_NORMAL_ARRAY_BUFFER_BINDING_ARB                            0x8897
#define GL_COLOR_ARRAY_BUFFER_BINDING                                 0x8898
#define GL_COLOR_ARRAY_BUFFER_BINDING_ARB                             0x8898
#define GL_INDEX_ARRAY_BUFFER_BINDING                                 0x8899
#define GL_INDEX_ARRAY_BUFFER_BINDING_ARB                             0x8899
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING                         0x889A
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING_ARB                     0x889A
#define GL_EDGE_FLAG_ARRAY_BUFFER_BINDING                             0x889B
#define GL_EDGE_FLAG_ARRAY_BUFFER_BINDING_ARB                         0x889B
#define GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING                       0x889C
#define GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING_ARB                   0x889C
#define GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING_ARB                    0x889D
#define GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING                        0x889D
#define GL_FOG_COORD_ARRAY_BUFFER_BINDING                             0x889D
#define GL_WEIGHT_ARRAY_BUFFER_BINDING                                0x889E
#define GL_WEIGHT_ARRAY_BUFFER_BINDING_ARB                            0x889E
#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING                         0x889F
#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING_ARB                     0x889F
#define GL_PROGRAM_INSTRUCTIONS_ARB                                   0x88A0
#define GL_MAX_PROGRAM_INSTRUCTIONS_ARB                               0x88A1
#define GL_PROGRAM_NATIVE_INSTRUCTIONS_ARB                            0x88A2
#define GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB                        0x88A3
#define GL_PROGRAM_TEMPORARIES_ARB                                    0x88A4
#define GL_MAX_PROGRAM_TEMPORARIES_ARB                                0x88A5
#define GL_PROGRAM_NATIVE_TEMPORARIES_ARB                             0x88A6
#define GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB                         0x88A7
#define GL_PROGRAM_PARAMETERS_ARB                                     0x88A8
#define GL_MAX_PROGRAM_PARAMETERS_ARB                                 0x88A9
#define GL_PROGRAM_NATIVE_PARAMETERS_ARB                              0x88AA
#define GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB                          0x88AB
#define GL_PROGRAM_ATTRIBS_ARB                                        0x88AC
#define GL_MAX_PROGRAM_ATTRIBS_ARB                                    0x88AD
#define GL_PROGRAM_NATIVE_ATTRIBS_ARB                                 0x88AE
#define GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB                             0x88AF
#define GL_PROGRAM_ADDRESS_REGISTERS_ARB                              0x88B0
#define GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB                          0x88B1
#define GL_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB                       0x88B2
#define GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB                   0x88B3
#define GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB                           0x88B4
#define GL_MAX_PROGRAM_ENV_PARAMETERS_ARB                             0x88B5
#define GL_PROGRAM_UNDER_NATIVE_LIMITS_ARB                            0x88B6
#define GL_TRANSPOSE_CURRENT_MATRIX_ARB                               0x88B7
#define GL_READ_ONLY                                                  0x88B8
#define GL_READ_ONLY_ARB                                              0x88B8
#define GL_WRITE_ONLY                                                 0x88B9
#define GL_WRITE_ONLY_ARB                                             0x88B9
#define GL_READ_WRITE                                                 0x88BA
#define GL_READ_WRITE_ARB                                             0x88BA
#define GL_BUFFER_ACCESS                                              0x88BB
#define GL_BUFFER_ACCESS_ARB                                          0x88BB
#define GL_BUFFER_MAPPED                                              0x88BC
#define GL_BUFFER_MAPPED_ARB                                          0x88BC
#define GL_BUFFER_MAP_POINTER                                         0x88BD
#define GL_BUFFER_MAP_POINTER_ARB                                     0x88BD
#define GL_WRITE_DISCARD_NV                                           0x88BE
#define GL_TIME_ELAPSED                                               0x88BF
#define GL_TIME_ELAPSED_EXT                                           0x88BF
#define GL_MATRIX0_ARB                                                0x88C0
#define GL_MATRIX1_ARB                                                0x88C1
#define GL_MATRIX2_ARB                                                0x88C2
#define GL_MATRIX3_ARB                                                0x88C3
#define GL_MATRIX4_ARB                                                0x88C4
#define GL_MATRIX5_ARB                                                0x88C5
#define GL_MATRIX6_ARB                                                0x88C6
#define GL_MATRIX7_ARB                                                0x88C7
#define GL_MATRIX8_ARB                                                0x88C8
#define GL_MATRIX9_ARB                                                0x88C9
#define GL_MATRIX10_ARB                                               0x88CA
#define GL_MATRIX11_ARB                                               0x88CB
#define GL_MATRIX12_ARB                                               0x88CC
#define GL_MATRIX13_ARB                                               0x88CD
#define GL_MATRIX14_ARB                                               0x88CE
#define GL_MATRIX15_ARB                                               0x88CF
#define GL_MATRIX16_ARB                                               0x88D0
#define GL_MATRIX17_ARB                                               0x88D1
#define GL_MATRIX18_ARB                                               0x88D2
#define GL_MATRIX19_ARB                                               0x88D3
#define GL_MATRIX20_ARB                                               0x88D4
#define GL_MATRIX21_ARB                                               0x88D5
#define GL_MATRIX22_ARB                                               0x88D6
#define GL_MATRIX23_ARB                                               0x88D7
#define GL_MATRIX24_ARB                                               0x88D8
#define GL_MATRIX25_ARB                                               0x88D9
#define GL_MATRIX26_ARB                                               0x88DA
#define GL_MATRIX27_ARB                                               0x88DB
#define GL_MATRIX28_ARB                                               0x88DC
#define GL_MATRIX29_ARB                                               0x88DD
#define GL_MATRIX30_ARB                                               0x88DE
#define GL_MATRIX31_ARB                                               0x88DF
#define GL_STREAM_DRAW                                                0x88E0
#define GL_STREAM_DRAW_ARB                                            0x88E0
#define GL_STREAM_READ                                                0x88E1
#define GL_STREAM_READ_ARB                                            0x88E1
#define GL_STREAM_COPY                                                0x88E2
#define GL_STREAM_COPY_ARB                                            0x88E2
#define GL_STATIC_DRAW                                                0x88E4
#define GL_STATIC_DRAW_ARB                                            0x88E4
#define GL_STATIC_READ                                                0x88E5
#define GL_STATIC_READ_ARB                                            0x88E5
#define GL_STATIC_COPY                                                0x88E6
#define GL_STATIC_COPY_ARB                                            0x88E6
#define GL_DYNAMIC_DRAW                                               0x88E8
#define GL_DYNAMIC_DRAW_ARB                                           0x88E8
#define GL_DYNAMIC_READ                                               0x88E9
#define GL_DYNAMIC_READ_ARB                                           0x88E9
#define GL_DYNAMIC_COPY                                               0x88EA
#define GL_DYNAMIC_COPY_ARB                                           0x88EA
#define GL_PIXEL_PACK_BUFFER                                          0x88EB
#define GL_PIXEL_PACK_BUFFER_ARB                                      0x88EB
#define GL_PIXEL_PACK_BUFFER_EXT                                      0x88EB
#define GL_PIXEL_UNPACK_BUFFER                                        0x88EC
#define GL_PIXEL_UNPACK_BUFFER_ARB                                    0x88EC
#define GL_PIXEL_UNPACK_BUFFER_EXT                                    0x88EC
#define GL_PIXEL_PACK_BUFFER_BINDING                                  0x88ED
#define GL_PIXEL_PACK_BUFFER_BINDING_ARB                              0x88ED
#define GL_PIXEL_PACK_BUFFER_BINDING_EXT                              0x88ED
#define GL_PIXEL_UNPACK_BUFFER_BINDING                                0x88EF
#define GL_PIXEL_UNPACK_BUFFER_BINDING_ARB                            0x88EF
#define GL_PIXEL_UNPACK_BUFFER_BINDING_EXT                            0x88EF
#define GL_DEPTH24_STENCIL8                                           0x88F0
#define GL_DEPTH24_STENCIL8_EXT                                       0x88F0
#define GL_TEXTURE_STENCIL_SIZE                                       0x88F1
#define GL_TEXTURE_STENCIL_SIZE_EXT                                   0x88F1
#define GL_STENCIL_TAG_BITS_EXT                                       0x88F2
#define GL_STENCIL_CLEAR_TAG_VALUE_EXT                                0x88F3
#define GL_MAX_PROGRAM_EXEC_INSTRUCTIONS_NV                           0x88F4
#define GL_MAX_PROGRAM_CALL_DEPTH_NV                                  0x88F5
#define GL_MAX_PROGRAM_IF_DEPTH_NV                                    0x88F6
#define GL_MAX_PROGRAM_LOOP_DEPTH_NV                                  0x88F7
#define GL_MAX_PROGRAM_LOOP_COUNT_NV                                  0x88F8
#define GL_SRC1_COLOR                                                 0x88F9
#define GL_ONE_MINUS_SRC1_COLOR                                       0x88FA
#define GL_ONE_MINUS_SRC1_ALPHA                                       0x88FB
#define GL_MAX_DUAL_SOURCE_DRAW_BUFFERS                               0x88FC
#define GL_VERTEX_ATTRIB_ARRAY_INTEGER                                0x88FD
#define GL_VERTEX_ATTRIB_ARRAY_INTEGER_EXT                            0x88FD
#define GL_VERTEX_ATTRIB_ARRAY_INTEGER_NV                             0x88FD
#define GL_VERTEX_ATTRIB_ARRAY_DIVISOR                                0x88FE
#define GL_VERTEX_ATTRIB_ARRAY_DIVISOR_ARB                            0x88FE
#define GL_MAX_ARRAY_TEXTURE_LAYERS                                   0x88FF
#define GL_MAX_ARRAY_TEXTURE_LAYERS_EXT                               0x88FF
#define GL_MIN_PROGRAM_TEXEL_OFFSET                                   0x8904
#define GL_MIN_PROGRAM_TEXEL_OFFSET_EXT                               0x8904
#define GL_MIN_PROGRAM_TEXEL_OFFSET_NV                                0x8904
#define GL_MAX_PROGRAM_TEXEL_OFFSET                                   0x8905
#define GL_MAX_PROGRAM_TEXEL_OFFSET_EXT                               0x8905
#define GL_MAX_PROGRAM_TEXEL_OFFSET_NV                                0x8905
#define GL_PROGRAM_ATTRIB_COMPONENTS_NV                               0x8906
#define GL_PROGRAM_RESULT_COMPONENTS_NV                               0x8907
#define GL_MAX_PROGRAM_ATTRIB_COMPONENTS_NV                           0x8908
#define GL_MAX_PROGRAM_RESULT_COMPONENTS_NV                           0x8909
#define GL_STENCIL_TEST_TWO_SIDE_EXT                                  0x8910
#define GL_ACTIVE_STENCIL_FACE_EXT                                    0x8911
#define GL_MIRROR_CLAMP_TO_BORDER_EXT                                 0x8912
#define GL_SAMPLES_PASSED                                             0x8914
#define GL_SAMPLES_PASSED_ARB                                         0x8914
#define GL_GEOMETRY_VERTICES_OUT                                      0x8916
#define GL_GEOMETRY_INPUT_TYPE                                        0x8917
#define GL_GEOMETRY_OUTPUT_TYPE                                       0x8918
#define GL_SAMPLER_BINDING                                            0x8919
#define GL_CLAMP_VERTEX_COLOR                                         0x891A
#define GL_CLAMP_VERTEX_COLOR_ARB                                     0x891A
#define GL_CLAMP_FRAGMENT_COLOR                                       0x891B
#define GL_CLAMP_FRAGMENT_COLOR_ARB                                   0x891B
#define GL_CLAMP_READ_COLOR                                           0x891C
#define GL_CLAMP_READ_COLOR_ARB                                       0x891C
#define GL_FIXED_ONLY                                                 0x891D
#define GL_FIXED_ONLY_ARB                                             0x891D
#define GL_TESS_CONTROL_PROGRAM_NV                                    0x891E
#define GL_TESS_EVALUATION_PROGRAM_NV                                 0x891F
#define GL_VERTEX_ATTRIB_MAP1_APPLE                                   0x8A00
#define GL_VERTEX_ATTRIB_MAP2_APPLE                                   0x8A01
#define GL_VERTEX_ATTRIB_MAP1_SIZE_APPLE                              0x8A02
#define GL_VERTEX_ATTRIB_MAP1_COEFF_APPLE                             0x8A03
#define GL_VERTEX_ATTRIB_MAP1_ORDER_APPLE                             0x8A04
#define GL_VERTEX_ATTRIB_MAP1_DOMAIN_APPLE                            0x8A05
#define GL_VERTEX_ATTRIB_MAP2_SIZE_APPLE                              0x8A06
#define GL_VERTEX_ATTRIB_MAP2_COEFF_APPLE                             0x8A07
#define GL_VERTEX_ATTRIB_MAP2_ORDER_APPLE                             0x8A08
#define GL_VERTEX_ATTRIB_MAP2_DOMAIN_APPLE                            0x8A09
#define GL_DRAW_PIXELS_APPLE                                          0x8A0A
#define GL_FENCE_APPLE                                                0x8A0B
#define GL_ELEMENT_ARRAY_APPLE                                        0x8A0C
#define GL_ELEMENT_ARRAY_TYPE_APPLE                                   0x8A0D
#define GL_ELEMENT_ARRAY_POINTER_APPLE                                0x8A0E
#define GL_COLOR_FLOAT_APPLE                                          0x8A0F
#define GL_UNIFORM_BUFFER                                             0x8A11
#define GL_BUFFER_SERIALIZED_MODIFY_APPLE                             0x8A12
#define GL_BUFFER_FLUSHING_UNMAP_APPLE                                0x8A13
#define GL_AUX_DEPTH_STENCIL_APPLE                                    0x8A14
#define GL_PACK_ROW_BYTES_APPLE                                       0x8A15
#define GL_UNPACK_ROW_BYTES_APPLE                                     0x8A16
#define GL_RELEASED_APPLE                                             0x8A19
#define GL_VOLATILE_APPLE                                             0x8A1A
#define GL_RETAINED_APPLE                                             0x8A1B
#define GL_UNDEFINED_APPLE                                            0x8A1C
#define GL_PURGEABLE_APPLE                                            0x8A1D
#define GL_RGB_422_APPLE                                              0x8A1F
#define GL_UNIFORM_BUFFER_BINDING                                     0x8A28
#define GL_UNIFORM_BUFFER_START                                       0x8A29
#define GL_UNIFORM_BUFFER_SIZE                                        0x8A2A
#define GL_MAX_VERTEX_UNIFORM_BLOCKS                                  0x8A2B
#define GL_MAX_GEOMETRY_UNIFORM_BLOCKS                                0x8A2C
#define GL_MAX_FRAGMENT_UNIFORM_BLOCKS                                0x8A2D
#define GL_MAX_COMBINED_UNIFORM_BLOCKS                                0x8A2E
#define GL_MAX_UNIFORM_BUFFER_BINDINGS                                0x8A2F
#define GL_MAX_UNIFORM_BLOCK_SIZE                                     0x8A30
#define GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS                     0x8A31
#define GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS                   0x8A32
#define GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS                   0x8A33
#define GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT                            0x8A34
#define GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH                       0x8A35
#define GL_ACTIVE_UNIFORM_BLOCKS                                      0x8A36
#define GL_UNIFORM_TYPE                                               0x8A37
#define GL_UNIFORM_SIZE                                               0x8A38
#define GL_UNIFORM_NAME_LENGTH                                        0x8A39
#define GL_UNIFORM_BLOCK_INDEX                                        0x8A3A
#define GL_UNIFORM_OFFSET                                             0x8A3B
#define GL_UNIFORM_ARRAY_STRIDE                                       0x8A3C
#define GL_UNIFORM_MATRIX_STRIDE                                      0x8A3D
#define GL_UNIFORM_IS_ROW_MAJOR                                       0x8A3E
#define GL_UNIFORM_BLOCK_BINDING                                      0x8A3F
#define GL_UNIFORM_BLOCK_DATA_SIZE                                    0x8A40
#define GL_UNIFORM_BLOCK_NAME_LENGTH                                  0x8A41
#define GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS                              0x8A42
#define GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES                       0x8A43
#define GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER                  0x8A44
#define GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER                0x8A45
#define GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER                0x8A46
#define GL_TEXTURE_SRGB_DECODE_EXT                                    0x8A48
#define GL_DECODE_EXT                                                 0x8A49
#define GL_SKIP_DECODE_EXT                                            0x8A4A
#define GL_PROGRAM_PIPELINE_OBJECT_EXT                                0x8A4F
#define GL_RGB_RAW_422_APPLE                                          0x8A51
#define GL_FRAGMENT_SHADER_DISCARDS_SAMPLES_EXT                       0x8A52
#define GL_FRAGMENT_SHADER                                            0x8B30
#define GL_FRAGMENT_SHADER_ARB                                        0x8B30
#define GL_VERTEX_SHADER                                              0x8B31
#define GL_VERTEX_SHADER_ARB                                          0x8B31
#define GL_PROGRAM_OBJECT_ARB                                         0x8B40
#define GL_PROGRAM_OBJECT_EXT                                         0x8B40
#define GL_SHADER_OBJECT_ARB                                          0x8B48
#define GL_SHADER_OBJECT_EXT                                          0x8B48
#define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS                            0x8B49
#define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB                        0x8B49
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS                              0x8B4A
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS_ARB                          0x8B4A
#define GL_MAX_VARYING_FLOATS                                         0x8B4B
#define GL_MAX_VARYING_COMPONENTS                                     0x8B4B
#define GL_MAX_VARYING_COMPONENTS_EXT                                 0x8B4B
#define GL_MAX_VARYING_FLOATS_ARB                                     0x8B4B
#define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS                             0x8B4C
#define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB                         0x8B4C
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS                           0x8B4D
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB                       0x8B4D
#define GL_OBJECT_TYPE_ARB                                            0x8B4E
#define GL_SHADER_TYPE                                                0x8B4F
#define GL_OBJECT_SUBTYPE_ARB                                         0x8B4F
#define GL_FLOAT_VEC2                                                 0x8B50
#define GL_FLOAT_VEC2_ARB                                             0x8B50
#define GL_FLOAT_VEC3                                                 0x8B51
#define GL_FLOAT_VEC3_ARB                                             0x8B51
#define GL_FLOAT_VEC4                                                 0x8B52
#define GL_FLOAT_VEC4_ARB                                             0x8B52
#define GL_INT_VEC2                                                   0x8B53
#define GL_INT_VEC2_ARB                                               0x8B53
#define GL_INT_VEC3                                                   0x8B54
#define GL_INT_VEC3_ARB                                               0x8B54
#define GL_INT_VEC4                                                   0x8B55
#define GL_INT_VEC4_ARB                                               0x8B55
#define GL_BOOL                                                       0x8B56
#define GL_BOOL_ARB                                                   0x8B56
#define GL_BOOL_VEC2                                                  0x8B57
#define GL_BOOL_VEC2_ARB                                              0x8B57
#define GL_BOOL_VEC3                                                  0x8B58
#define GL_BOOL_VEC3_ARB                                              0x8B58
#define GL_BOOL_VEC4                                                  0x8B59
#define GL_BOOL_VEC4_ARB                                              0x8B59
#define GL_FLOAT_MAT2                                                 0x8B5A
#define GL_FLOAT_MAT2_ARB                                             0x8B5A
#define GL_FLOAT_MAT3                                                 0x8B5B
#define GL_FLOAT_MAT3_ARB                                             0x8B5B
#define GL_FLOAT_MAT4                                                 0x8B5C
#define GL_FLOAT_MAT4_ARB                                             0x8B5C
#define GL_SAMPLER_1D                                                 0x8B5D
#define GL_SAMPLER_1D_ARB                                             0x8B5D
#define GL_SAMPLER_2D                                                 0x8B5E
#define GL_SAMPLER_2D_ARB                                             0x8B5E
#define GL_SAMPLER_3D                                                 0x8B5F
#define GL_SAMPLER_3D_ARB                                             0x8B5F
#define GL_SAMPLER_CUBE                                               0x8B60
#define GL_SAMPLER_CUBE_ARB                                           0x8B60
#define GL_SAMPLER_1D_SHADOW                                          0x8B61
#define GL_SAMPLER_1D_SHADOW_ARB                                      0x8B61
#define GL_SAMPLER_2D_SHADOW                                          0x8B62
#define GL_SAMPLER_2D_SHADOW_ARB                                      0x8B62
#define GL_SAMPLER_2D_RECT                                            0x8B63
#define GL_SAMPLER_2D_RECT_ARB                                        0x8B63
#define GL_SAMPLER_2D_RECT_SHADOW                                     0x8B64
#define GL_SAMPLER_2D_RECT_SHADOW_ARB                                 0x8B64
#define GL_FLOAT_MAT2x3                                               0x8B65
#define GL_FLOAT_MAT2x4                                               0x8B66
#define GL_FLOAT_MAT3x2                                               0x8B67
#define GL_FLOAT_MAT3x4                                               0x8B68
#define GL_FLOAT_MAT4x2                                               0x8B69
#define GL_FLOAT_MAT4x3                                               0x8B6A
#define GL_DELETE_STATUS                                              0x8B80
#define GL_OBJECT_DELETE_STATUS_ARB                                   0x8B80
#define GL_COMPILE_STATUS                                             0x8B81
#define GL_OBJECT_COMPILE_STATUS_ARB                                  0x8B81
#define GL_LINK_STATUS                                                0x8B82
#define GL_OBJECT_LINK_STATUS_ARB                                     0x8B82
#define GL_VALIDATE_STATUS                                            0x8B83
#define GL_OBJECT_VALIDATE_STATUS_ARB                                 0x8B83
#define GL_INFO_LOG_LENGTH                                            0x8B84
#define GL_OBJECT_INFO_LOG_LENGTH_ARB                                 0x8B84
#define GL_ATTACHED_SHADERS                                           0x8B85
#define GL_OBJECT_ATTACHED_OBJECTS_ARB                                0x8B85
#define GL_ACTIVE_UNIFORMS                                            0x8B86
#define GL_OBJECT_ACTIVE_UNIFORMS_ARB                                 0x8B86
#define GL_ACTIVE_UNIFORM_MAX_LENGTH                                  0x8B87
#define GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB                       0x8B87
#define GL_SHADER_SOURCE_LENGTH                                       0x8B88
#define GL_OBJECT_SHADER_SOURCE_LENGTH_ARB                            0x8B88
#define GL_ACTIVE_ATTRIBUTES                                          0x8B89
#define GL_OBJECT_ACTIVE_ATTRIBUTES_ARB                               0x8B89
#define GL_ACTIVE_ATTRIBUTE_MAX_LENGTH                                0x8B8A
#define GL_OBJECT_ACTIVE_ATTRIBUTE_MAX_LENGTH_ARB                     0x8B8A
#define GL_FRAGMENT_SHADER_DERIVATIVE_HINT                            0x8B8B
#define GL_FRAGMENT_SHADER_DERIVATIVE_HINT_ARB                        0x8B8B
#define GL_SHADING_LANGUAGE_VERSION                                   0x8B8C
#define GL_SHADING_LANGUAGE_VERSION_ARB                               0x8B8C
#define GL_CURRENT_PROGRAM                                            0x8B8D
#define GL_IMPLEMENTATION_COLOR_READ_TYPE                             0x8B9A
#define GL_IMPLEMENTATION_COLOR_READ_FORMAT                           0x8B9B
#define GL_COUNTER_TYPE_AMD                                           0x8BC0
#define GL_COUNTER_RANGE_AMD                                          0x8BC1
#define GL_UNSIGNED_INT64_AMD                                         0x8BC2
#define GL_PERCENTAGE_AMD                                             0x8BC3
#define GL_PERFMON_RESULT_AVAILABLE_AMD                               0x8BC4
#define GL_PERFMON_RESULT_SIZE_AMD                                    0x8BC5
#define GL_PERFMON_RESULT_AMD                                         0x8BC6
#define GL_TEXTURE_RED_TYPE                                           0x8C10
#define GL_TEXTURE_RED_TYPE_ARB                                       0x8C10
#define GL_TEXTURE_GREEN_TYPE                                         0x8C11
#define GL_TEXTURE_GREEN_TYPE_ARB                                     0x8C11
#define GL_TEXTURE_BLUE_TYPE                                          0x8C12
#define GL_TEXTURE_BLUE_TYPE_ARB                                      0x8C12
#define GL_TEXTURE_ALPHA_TYPE                                         0x8C13
#define GL_TEXTURE_ALPHA_TYPE_ARB                                     0x8C13
#define GL_TEXTURE_LUMINANCE_TYPE                                     0x8C14
#define GL_TEXTURE_LUMINANCE_TYPE_ARB                                 0x8C14
#define GL_TEXTURE_INTENSITY_TYPE                                     0x8C15
#define GL_TEXTURE_INTENSITY_TYPE_ARB                                 0x8C15
#define GL_TEXTURE_DEPTH_TYPE                                         0x8C16
#define GL_TEXTURE_DEPTH_TYPE_ARB                                     0x8C16
#define GL_UNSIGNED_NORMALIZED                                        0x8C17
#define GL_UNSIGNED_NORMALIZED_ARB                                    0x8C17
#define GL_TEXTURE_1D_ARRAY                                           0x8C18
#define GL_TEXTURE_1D_ARRAY_EXT                                       0x8C18
#define GL_PROXY_TEXTURE_1D_ARRAY                                     0x8C19
#define GL_PROXY_TEXTURE_1D_ARRAY_EXT                                 0x8C19
#define GL_TEXTURE_2D_ARRAY                                           0x8C1A
#define GL_TEXTURE_2D_ARRAY_EXT                                       0x8C1A
#define GL_PROXY_TEXTURE_2D_ARRAY                                     0x8C1B
#define GL_PROXY_TEXTURE_2D_ARRAY_EXT                                 0x8C1B
#define GL_TEXTURE_BINDING_1D_ARRAY                                   0x8C1C
#define GL_TEXTURE_BINDING_1D_ARRAY_EXT                               0x8C1C
#define GL_TEXTURE_BINDING_2D_ARRAY                                   0x8C1D
#define GL_TEXTURE_BINDING_2D_ARRAY_EXT                               0x8C1D
#define GL_GEOMETRY_PROGRAM_NV                                        0x8C26
#define GL_MAX_PROGRAM_OUTPUT_VERTICES_NV                             0x8C27
#define GL_MAX_PROGRAM_TOTAL_OUTPUT_COMPONENTS_NV                     0x8C28
#define GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS                           0x8C29
#define GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS_ARB                       0x8C29
#define GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS_EXT                       0x8C29
#define GL_TEXTURE_BUFFER                                             0x8C2A
#define GL_TEXTURE_BUFFER_ARB                                         0x8C2A
#define GL_TEXTURE_BUFFER_EXT                                         0x8C2A
#define GL_TEXTURE_BUFFER_BINDING                                     0x8C2A
#define GL_MAX_TEXTURE_BUFFER_SIZE                                    0x8C2B
#define GL_MAX_TEXTURE_BUFFER_SIZE_ARB                                0x8C2B
#define GL_MAX_TEXTURE_BUFFER_SIZE_EXT                                0x8C2B
#define GL_TEXTURE_BINDING_BUFFER                                     0x8C2C
#define GL_TEXTURE_BINDING_BUFFER_ARB                                 0x8C2C
#define GL_TEXTURE_BINDING_BUFFER_EXT                                 0x8C2C
#define GL_TEXTURE_BUFFER_DATA_STORE_BINDING                          0x8C2D
#define GL_TEXTURE_BUFFER_DATA_STORE_BINDING_ARB                      0x8C2D
#define GL_TEXTURE_BUFFER_DATA_STORE_BINDING_EXT                      0x8C2D
#define GL_TEXTURE_BUFFER_FORMAT_ARB                                  0x8C2E
#define GL_TEXTURE_BUFFER_FORMAT_EXT                                  0x8C2E
#define GL_ANY_SAMPLES_PASSED                                         0x8C2F
#define GL_SAMPLE_SHADING                                             0x8C36
#define GL_SAMPLE_SHADING_ARB                                         0x8C36
#define GL_MIN_SAMPLE_SHADING_VALUE                                   0x8C37
#define GL_MIN_SAMPLE_SHADING_VALUE_ARB                               0x8C37
#define GL_R11F_G11F_B10F                                             0x8C3A
#define GL_R11F_G11F_B10F_EXT                                         0x8C3A
#define GL_UNSIGNED_INT_10F_11F_11F_REV                               0x8C3B
#define GL_UNSIGNED_INT_10F_11F_11F_REV_EXT                           0x8C3B
#define GL_RGBA_SIGNED_COMPONENTS_EXT                                 0x8C3C
#define GL_RGB9_E5                                                    0x8C3D
#define GL_RGB9_E5_EXT                                                0x8C3D
#define GL_UNSIGNED_INT_5_9_9_9_REV                                   0x8C3E
#define GL_UNSIGNED_INT_5_9_9_9_REV_EXT                               0x8C3E
#define GL_TEXTURE_SHARED_SIZE                                        0x8C3F
#define GL_TEXTURE_SHARED_SIZE_EXT                                    0x8C3F
#define GL_SRGB                                                       0x8C40
#define GL_SRGB_EXT                                                   0x8C40
#define GL_SRGB8                                                      0x8C41
#define GL_SRGB8_EXT                                                  0x8C41
#define GL_SRGB_ALPHA                                                 0x8C42
#define GL_SRGB_ALPHA_EXT                                             0x8C42
#define GL_SRGB8_ALPHA8                                               0x8C43
#define GL_SRGB8_ALPHA8_EXT                                           0x8C43
#define GL_SLUMINANCE_ALPHA                                           0x8C44
#define GL_SLUMINANCE_ALPHA_EXT                                       0x8C44
#define GL_SLUMINANCE8_ALPHA8                                         0x8C45
#define GL_SLUMINANCE8_ALPHA8_EXT                                     0x8C45
#define GL_SLUMINANCE                                                 0x8C46
#define GL_SLUMINANCE_EXT                                             0x8C46
#define GL_SLUMINANCE8                                                0x8C47
#define GL_SLUMINANCE8_EXT                                            0x8C47
#define GL_COMPRESSED_SRGB                                            0x8C48
#define GL_COMPRESSED_SRGB_EXT                                        0x8C48
#define GL_COMPRESSED_SRGB_ALPHA                                      0x8C49
#define GL_COMPRESSED_SRGB_ALPHA_EXT                                  0x8C49
#define GL_COMPRESSED_SLUMINANCE                                      0x8C4A
#define GL_COMPRESSED_SLUMINANCE_EXT                                  0x8C4A
#define GL_COMPRESSED_SLUMINANCE_ALPHA                                0x8C4B
#define GL_COMPRESSED_SLUMINANCE_ALPHA_EXT                            0x8C4B
#define GL_COMPRESSED_SRGB_S3TC_DXT1_EXT                              0x8C4C
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT                        0x8C4D
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT                        0x8C4E
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT                        0x8C4F
#define GL_COMPRESSED_LUMINANCE_LATC1_EXT                             0x8C70
#define GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT                      0x8C71
#define GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT                       0x8C72
#define GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT                0x8C73
#define GL_TESS_CONTROL_PROGRAM_PARAMETER_BUFFER_NV                   0x8C74
#define GL_TESS_EVALUATION_PROGRAM_PARAMETER_BUFFER_NV                0x8C75
#define GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH                      0x8C76
#define GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH_EXT                  0x8C76
#define GL_BACK_PRIMARY_COLOR_NV                                      0x8C77
#define GL_BACK_SECONDARY_COLOR_NV                                    0x8C78
#define GL_TEXTURE_COORD_NV                                           0x8C79
#define GL_CLIP_DISTANCE_NV                                           0x8C7A
#define GL_VERTEX_ID_NV                                               0x8C7B
#define GL_PRIMITIVE_ID_NV                                            0x8C7C
#define GL_GENERIC_ATTRIB_NV                                          0x8C7D
#define GL_TRANSFORM_FEEDBACK_ATTRIBS_NV                              0x8C7E
#define GL_TRANSFORM_FEEDBACK_BUFFER_MODE                             0x8C7F
#define GL_TRANSFORM_FEEDBACK_BUFFER_MODE_EXT                         0x8C7F
#define GL_TRANSFORM_FEEDBACK_BUFFER_MODE_NV                          0x8C7F
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS                 0x8C80
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS_EXT             0x8C80
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS_NV              0x8C80
#define GL_ACTIVE_VARYINGS_NV                                         0x8C81
#define GL_ACTIVE_VARYING_MAX_LENGTH_NV                               0x8C82
#define GL_TRANSFORM_FEEDBACK_VARYINGS                                0x8C83
#define GL_TRANSFORM_FEEDBACK_VARYINGS_EXT                            0x8C83
#define GL_TRANSFORM_FEEDBACK_VARYINGS_NV                             0x8C83
#define GL_TRANSFORM_FEEDBACK_BUFFER_START                            0x8C84
#define GL_TRANSFORM_FEEDBACK_BUFFER_START_EXT                        0x8C84
#define GL_TRANSFORM_FEEDBACK_BUFFER_START_NV                         0x8C84
#define GL_TRANSFORM_FEEDBACK_BUFFER_SIZE                             0x8C85
#define GL_TRANSFORM_FEEDBACK_BUFFER_SIZE_EXT                         0x8C85
#define GL_TRANSFORM_FEEDBACK_BUFFER_SIZE_NV                          0x8C85
#define GL_TRANSFORM_FEEDBACK_RECORD_NV                               0x8C86
#define GL_PRIMITIVES_GENERATED                                       0x8C87
#define GL_PRIMITIVES_GENERATED_EXT                                   0x8C87
#define GL_PRIMITIVES_GENERATED_NV                                    0x8C87
#define GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN                      0x8C88
#define GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_EXT                  0x8C88
#define GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV                   0x8C88
#define GL_RASTERIZER_DISCARD                                         0x8C89
#define GL_RASTERIZER_DISCARD_EXT                                     0x8C89
#define GL_RASTERIZER_DISCARD_NV                                      0x8C89
#define GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS              0x8C8A
#define GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS_EXT          0x8C8A
#define GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS_NV           0x8C8A
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS                    0x8C8B
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS_EXT                0x8C8B
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS_NV                 0x8C8B
#define GL_INTERLEAVED_ATTRIBS                                        0x8C8C
#define GL_INTERLEAVED_ATTRIBS_EXT                                    0x8C8C
#define GL_INTERLEAVED_ATTRIBS_NV                                     0x8C8C
#define GL_SEPARATE_ATTRIBS                                           0x8C8D
#define GL_SEPARATE_ATTRIBS_EXT                                       0x8C8D
#define GL_SEPARATE_ATTRIBS_NV                                        0x8C8D
#define GL_TRANSFORM_FEEDBACK_BUFFER                                  0x8C8E
#define GL_TRANSFORM_FEEDBACK_BUFFER_EXT                              0x8C8E
#define GL_TRANSFORM_FEEDBACK_BUFFER_NV                               0x8C8E
#define GL_TRANSFORM_FEEDBACK_BUFFER_BINDING                          0x8C8F
#define GL_TRANSFORM_FEEDBACK_BUFFER_BINDING_EXT                      0x8C8F
#define GL_TRANSFORM_FEEDBACK_BUFFER_BINDING_NV                       0x8C8F
#define GL_POINT_SPRITE_COORD_ORIGIN                                  0x8CA0
#define GL_LOWER_LEFT                                                 0x8CA1
#define GL_UPPER_LEFT                                                 0x8CA2
#define GL_STENCIL_BACK_REF                                           0x8CA3
#define GL_STENCIL_BACK_VALUE_MASK                                    0x8CA4
#define GL_STENCIL_BACK_WRITEMASK                                     0x8CA5
#define GL_DRAW_FRAMEBUFFER_BINDING                                   0x8CA6
#define GL_DRAW_FRAMEBUFFER_BINDING_EXT                               0x8CA6
#define GL_FRAMEBUFFER_BINDING                                        0x8CA6
#define GL_FRAMEBUFFER_BINDING_EXT                                    0x8CA6
#define GL_RENDERBUFFER_BINDING                                       0x8CA7
#define GL_RENDERBUFFER_BINDING_EXT                                   0x8CA7
#define GL_READ_FRAMEBUFFER                                           0x8CA8
#define GL_READ_FRAMEBUFFER_EXT                                       0x8CA8
#define GL_DRAW_FRAMEBUFFER                                           0x8CA9
#define GL_DRAW_FRAMEBUFFER_EXT                                       0x8CA9
#define GL_READ_FRAMEBUFFER_BINDING                                   0x8CAA
#define GL_READ_FRAMEBUFFER_BINDING_EXT                               0x8CAA
#define GL_RENDERBUFFER_COVERAGE_SAMPLES_NV                           0x8CAB
#define GL_RENDERBUFFER_SAMPLES                                       0x8CAB
#define GL_RENDERBUFFER_SAMPLES_EXT                                   0x8CAB
#define GL_DEPTH_COMPONENT32F                                         0x8CAC
#define GL_DEPTH32F_STENCIL8                                          0x8CAD
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE                         0x8CD0
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT                     0x8CD0
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME                         0x8CD1
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT                     0x8CD1
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL                       0x8CD2
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL_EXT                   0x8CD2
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE               0x8CD3
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE_EXT           0x8CD3
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_EXT              0x8CD4
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER                       0x8CD4
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER_EXT                   0x8CD4
#define GL_FRAMEBUFFER_COMPLETE                                       0x8CD5
#define GL_FRAMEBUFFER_COMPLETE_EXT                                   0x8CD5
#define GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT                          0x8CD6
#define GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT                      0x8CD6
#define GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT                  0x8CD7
#define GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT              0x8CD7
#define GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT                      0x8CD9
#define GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT                         0x8CDA
#define GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER                         0x8CDB
#define GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT                     0x8CDB
#define GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER                         0x8CDC
#define GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT                     0x8CDC
#define GL_FRAMEBUFFER_UNSUPPORTED                                    0x8CDD
#define GL_FRAMEBUFFER_UNSUPPORTED_EXT                                0x8CDD
#define GL_MAX_COLOR_ATTACHMENTS                                      0x8CDF
#define GL_MAX_COLOR_ATTACHMENTS_EXT                                  0x8CDF
#define GL_COLOR_ATTACHMENT0                                          0x8CE0
#define GL_COLOR_ATTACHMENT0_EXT                                      0x8CE0
#define GL_COLOR_ATTACHMENT1                                          0x8CE1
#define GL_COLOR_ATTACHMENT1_EXT                                      0x8CE1
#define GL_COLOR_ATTACHMENT2                                          0x8CE2
#define GL_COLOR_ATTACHMENT2_EXT                                      0x8CE2
#define GL_COLOR_ATTACHMENT3                                          0x8CE3
#define GL_COLOR_ATTACHMENT3_EXT                                      0x8CE3
#define GL_COLOR_ATTACHMENT4                                          0x8CE4
#define GL_COLOR_ATTACHMENT4_EXT                                      0x8CE4
#define GL_COLOR_ATTACHMENT5                                          0x8CE5
#define GL_COLOR_ATTACHMENT5_EXT                                      0x8CE5
#define GL_COLOR_ATTACHMENT6                                          0x8CE6
#define GL_COLOR_ATTACHMENT6_EXT                                      0x8CE6
#define GL_COLOR_ATTACHMENT7                                          0x8CE7
#define GL_COLOR_ATTACHMENT7_EXT                                      0x8CE7
#define GL_COLOR_ATTACHMENT8                                          0x8CE8
#define GL_COLOR_ATTACHMENT8_EXT                                      0x8CE8
#define GL_COLOR_ATTACHMENT9                                          0x8CE9
#define GL_COLOR_ATTACHMENT9_EXT                                      0x8CE9
#define GL_COLOR_ATTACHMENT10                                         0x8CEA
#define GL_COLOR_ATTACHMENT10_EXT                                     0x8CEA
#define GL_COLOR_ATTACHMENT11                                         0x8CEB
#define GL_COLOR_ATTACHMENT11_EXT                                     0x8CEB
#define GL_COLOR_ATTACHMENT12                                         0x8CEC
#define GL_COLOR_ATTACHMENT12_EXT                                     0x8CEC
#define GL_COLOR_ATTACHMENT13                                         0x8CED
#define GL_COLOR_ATTACHMENT13_EXT                                     0x8CED
#define GL_COLOR_ATTACHMENT14                                         0x8CEE
#define GL_COLOR_ATTACHMENT14_EXT                                     0x8CEE
#define GL_COLOR_ATTACHMENT15                                         0x8CEF
#define GL_COLOR_ATTACHMENT15_EXT                                     0x8CEF
#define GL_COLOR_ATTACHMENT16                                         0x8CF0
#define GL_COLOR_ATTACHMENT17                                         0x8CF1
#define GL_COLOR_ATTACHMENT18                                         0x8CF2
#define GL_COLOR_ATTACHMENT19                                         0x8CF3
#define GL_COLOR_ATTACHMENT20                                         0x8CF4
#define GL_COLOR_ATTACHMENT21                                         0x8CF5
#define GL_COLOR_ATTACHMENT22                                         0x8CF6
#define GL_COLOR_ATTACHMENT23                                         0x8CF7
#define GL_COLOR_ATTACHMENT24                                         0x8CF8
#define GL_COLOR_ATTACHMENT25                                         0x8CF9
#define GL_COLOR_ATTACHMENT26                                         0x8CFA
#define GL_COLOR_ATTACHMENT27                                         0x8CFB
#define GL_COLOR_ATTACHMENT28                                         0x8CFC
#define GL_COLOR_ATTACHMENT29                                         0x8CFD
#define GL_COLOR_ATTACHMENT30                                         0x8CFE
#define GL_COLOR_ATTACHMENT31                                         0x8CFF
#define GL_DEPTH_ATTACHMENT                                           0x8D00
#define GL_DEPTH_ATTACHMENT_EXT                                       0x8D00
#define GL_STENCIL_ATTACHMENT                                         0x8D20
#define GL_STENCIL_ATTACHMENT_EXT                                     0x8D20
#define GL_FRAMEBUFFER                                                0x8D40
#define GL_FRAMEBUFFER_EXT                                            0x8D40
#define GL_RENDERBUFFER                                               0x8D41
#define GL_RENDERBUFFER_EXT                                           0x8D41
#define GL_RENDERBUFFER_WIDTH                                         0x8D42
#define GL_RENDERBUFFER_WIDTH_EXT                                     0x8D42
#define GL_RENDERBUFFER_HEIGHT                                        0x8D43
#define GL_RENDERBUFFER_HEIGHT_EXT                                    0x8D43
#define GL_RENDERBUFFER_INTERNAL_FORMAT                               0x8D44
#define GL_RENDERBUFFER_INTERNAL_FORMAT_EXT                           0x8D44
#define GL_STENCIL_INDEX1                                             0x8D46
#define GL_STENCIL_INDEX1_EXT                                         0x8D46
#define GL_STENCIL_INDEX4                                             0x8D47
#define GL_STENCIL_INDEX4_EXT                                         0x8D47
#define GL_STENCIL_INDEX8                                             0x8D48
#define GL_STENCIL_INDEX8_EXT                                         0x8D48
#define GL_STENCIL_INDEX16                                            0x8D49
#define GL_STENCIL_INDEX16_EXT                                        0x8D49
#define GL_RENDERBUFFER_RED_SIZE                                      0x8D50
#define GL_RENDERBUFFER_RED_SIZE_EXT                                  0x8D50
#define GL_RENDERBUFFER_GREEN_SIZE                                    0x8D51
#define GL_RENDERBUFFER_GREEN_SIZE_EXT                                0x8D51
#define GL_RENDERBUFFER_BLUE_SIZE                                     0x8D52
#define GL_RENDERBUFFER_BLUE_SIZE_EXT                                 0x8D52
#define GL_RENDERBUFFER_ALPHA_SIZE                                    0x8D53
#define GL_RENDERBUFFER_ALPHA_SIZE_EXT                                0x8D53
#define GL_RENDERBUFFER_DEPTH_SIZE                                    0x8D54
#define GL_RENDERBUFFER_DEPTH_SIZE_EXT                                0x8D54
#define GL_RENDERBUFFER_STENCIL_SIZE                                  0x8D55
#define GL_RENDERBUFFER_STENCIL_SIZE_EXT                              0x8D55
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE                         0x8D56
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_EXT                     0x8D56
#define GL_MAX_SAMPLES                                                0x8D57
#define GL_MAX_SAMPLES_EXT                                            0x8D57
#define GL_RGB565                                                     0x8D62
#define GL_PRIMITIVE_RESTART_FIXED_INDEX                              0x8D69
#define GL_ANY_SAMPLES_PASSED_CONSERVATIVE                            0x8D6A
#define GL_MAX_ELEMENT_INDEX                                          0x8D6B
#define GL_RGBA32UI                                                   0x8D70
#define GL_RGBA32UI_EXT                                               0x8D70
#define GL_RGB32UI                                                    0x8D71
#define GL_RGB32UI_EXT                                                0x8D71
#define GL_ALPHA32UI_EXT                                              0x8D72
#define GL_INTENSITY32UI_EXT                                          0x8D73
#define GL_LUMINANCE32UI_EXT                                          0x8D74
#define GL_LUMINANCE_ALPHA32UI_EXT                                    0x8D75
#define GL_RGBA16UI                                                   0x8D76
#define GL_RGBA16UI_EXT                                               0x8D76
#define GL_RGB16UI                                                    0x8D77
#define GL_RGB16UI_EXT                                                0x8D77
#define GL_ALPHA16UI_EXT                                              0x8D78
#define GL_INTENSITY16UI_EXT                                          0x8D79
#define GL_LUMINANCE16UI_EXT                                          0x8D7A
#define GL_LUMINANCE_ALPHA16UI_EXT                                    0x8D7B
#define GL_RGBA8UI                                                    0x8D7C
#define GL_RGBA8UI_EXT                                                0x8D7C
#define GL_RGB8UI                                                     0x8D7D
#define GL_RGB8UI_EXT                                                 0x8D7D
#define GL_ALPHA8UI_EXT                                               0x8D7E
#define GL_INTENSITY8UI_EXT                                           0x8D7F
#define GL_LUMINANCE8UI_EXT                                           0x8D80
#define GL_LUMINANCE_ALPHA8UI_EXT                                     0x8D81
#define GL_RGBA32I                                                    0x8D82
#define GL_RGBA32I_EXT                                                0x8D82
#define GL_RGB32I                                                     0x8D83
#define GL_RGB32I_EXT                                                 0x8D83
#define GL_ALPHA32I_EXT                                               0x8D84
#define GL_INTENSITY32I_EXT                                           0x8D85
#define GL_LUMINANCE32I_EXT                                           0x8D86
#define GL_LUMINANCE_ALPHA32I_EXT                                     0x8D87
#define GL_RGBA16I                                                    0x8D88
#define GL_RGBA16I_EXT                                                0x8D88
#define GL_RGB16I                                                     0x8D89
#define GL_RGB16I_EXT                                                 0x8D89
#define GL_ALPHA16I_EXT                                               0x8D8A
#define GL_INTENSITY16I_EXT                                           0x8D8B
#define GL_LUMINANCE16I_EXT                                           0x8D8C
#define GL_LUMINANCE_ALPHA16I_EXT                                     0x8D8D
#define GL_RGBA8I                                                     0x8D8E
#define GL_RGBA8I_EXT                                                 0x8D8E
#define GL_RGB8I                                                      0x8D8F
#define GL_RGB8I_EXT                                                  0x8D8F
#define GL_ALPHA8I_EXT                                                0x8D90
#define GL_INTENSITY8I_EXT                                            0x8D91
#define GL_LUMINANCE8I_EXT                                            0x8D92
#define GL_LUMINANCE_ALPHA8I_EXT                                      0x8D93
#define GL_RED_INTEGER                                                0x8D94
#define GL_RED_INTEGER_EXT                                            0x8D94
#define GL_GREEN_INTEGER                                              0x8D95
#define GL_GREEN_INTEGER_EXT                                          0x8D95
#define GL_BLUE_INTEGER                                               0x8D96
#define GL_BLUE_INTEGER_EXT                                           0x8D96
#define GL_ALPHA_INTEGER                                              0x8D97
#define GL_ALPHA_INTEGER_EXT                                          0x8D97
#define GL_RGB_INTEGER                                                0x8D98
#define GL_RGB_INTEGER_EXT                                            0x8D98
#define GL_RGBA_INTEGER                                               0x8D99
#define GL_RGBA_INTEGER_EXT                                           0x8D99
#define GL_BGR_INTEGER                                                0x8D9A
#define GL_BGR_INTEGER_EXT                                            0x8D9A
#define GL_BGRA_INTEGER                                               0x8D9B
#define GL_BGRA_INTEGER_EXT                                           0x8D9B
#define GL_LUMINANCE_INTEGER_EXT                                      0x8D9C
#define GL_LUMINANCE_ALPHA_INTEGER_EXT                                0x8D9D
#define GL_RGBA_INTEGER_MODE_EXT                                      0x8D9E
#define GL_INT_2_10_10_10_REV                                         0x8D9F
#define GL_MAX_PROGRAM_PARAMETER_BUFFER_BINDINGS_NV                   0x8DA0
#define GL_MAX_PROGRAM_PARAMETER_BUFFER_SIZE_NV                       0x8DA1
#define GL_VERTEX_PROGRAM_PARAMETER_BUFFER_NV                         0x8DA2
#define GL_GEOMETRY_PROGRAM_PARAMETER_BUFFER_NV                       0x8DA3
#define GL_FRAGMENT_PROGRAM_PARAMETER_BUFFER_NV                       0x8DA4
#define GL_MAX_PROGRAM_GENERIC_ATTRIBS_NV                             0x8DA5
#define GL_MAX_PROGRAM_GENERIC_RESULTS_NV                             0x8DA6
#define GL_FRAMEBUFFER_ATTACHMENT_LAYERED                             0x8DA7
#define GL_FRAMEBUFFER_ATTACHMENT_LAYERED_ARB                         0x8DA7
#define GL_FRAMEBUFFER_ATTACHMENT_LAYERED_EXT                         0x8DA7
#define GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS                       0x8DA8
#define GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_ARB                   0x8DA8
#define GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT                   0x8DA8
#define GL_FRAMEBUFFER_INCOMPLETE_LAYER_COUNT_ARB                     0x8DA9
#define GL_FRAMEBUFFER_INCOMPLETE_LAYER_COUNT_EXT                     0x8DA9
#define GL_LAYER_NV                                                   0x8DAA
#define GL_DEPTH_COMPONENT32F_NV                                      0x8DAB
#define GL_DEPTH32F_STENCIL8_NV                                       0x8DAC
#define GL_FLOAT_32_UNSIGNED_INT_24_8_REV                             0x8DAD
#define GL_FLOAT_32_UNSIGNED_INT_24_8_REV_NV                          0x8DAD
#define GL_SHADER_INCLUDE_ARB                                         0x8DAE
#define GL_DEPTH_BUFFER_FLOAT_MODE_NV                                 0x8DAF
#define GL_FRAMEBUFFER_SRGB                                           0x8DB9
#define GL_FRAMEBUFFER_SRGB_EXT                                       0x8DB9
#define GL_FRAMEBUFFER_SRGB_CAPABLE_EXT                               0x8DBA
#define GL_COMPRESSED_RED_RGTC1                                       0x8DBB
#define GL_COMPRESSED_RED_RGTC1_EXT                                   0x8DBB
#define GL_COMPRESSED_SIGNED_RED_RGTC1                                0x8DBC
#define GL_COMPRESSED_SIGNED_RED_RGTC1_EXT                            0x8DBC
#define GL_COMPRESSED_RED_GREEN_RGTC2_EXT                             0x8DBD
#define GL_COMPRESSED_RG_RGTC2                                        0x8DBD
#define GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT                      0x8DBE
#define GL_COMPRESSED_SIGNED_RG_RGTC2                                 0x8DBE
#define GL_SAMPLER_1D_ARRAY                                           0x8DC0
#define GL_SAMPLER_1D_ARRAY_EXT                                       0x8DC0
#define GL_SAMPLER_2D_ARRAY                                           0x8DC1
#define GL_SAMPLER_2D_ARRAY_EXT                                       0x8DC1
#define GL_SAMPLER_BUFFER                                             0x8DC2
#define GL_SAMPLER_BUFFER_EXT                                         0x8DC2
#define GL_SAMPLER_1D_ARRAY_SHADOW                                    0x8DC3
#define GL_SAMPLER_1D_ARRAY_SHADOW_EXT                                0x8DC3
#define GL_SAMPLER_2D_ARRAY_SHADOW                                    0x8DC4
#define GL_SAMPLER_2D_ARRAY_SHADOW_EXT                                0x8DC4
#define GL_SAMPLER_CUBE_SHADOW                                        0x8DC5
#define GL_SAMPLER_CUBE_SHADOW_EXT                                    0x8DC5
#define GL_UNSIGNED_INT_VEC2                                          0x8DC6
#define GL_UNSIGNED_INT_VEC2_EXT                                      0x8DC6
#define GL_UNSIGNED_INT_VEC3                                          0x8DC7
#define GL_UNSIGNED_INT_VEC3_EXT                                      0x8DC7
#define GL_UNSIGNED_INT_VEC4                                          0x8DC8
#define GL_UNSIGNED_INT_VEC4_EXT                                      0x8DC8
#define GL_INT_SAMPLER_1D                                             0x8DC9
#define GL_INT_SAMPLER_1D_EXT                                         0x8DC9
#define GL_INT_SAMPLER_2D                                             0x8DCA
#define GL_INT_SAMPLER_2D_EXT                                         0x8DCA
#define GL_INT_SAMPLER_3D                                             0x8DCB
#define GL_INT_SAMPLER_3D_EXT                                         0x8DCB
#define GL_INT_SAMPLER_CUBE                                           0x8DCC
#define GL_INT_SAMPLER_CUBE_EXT                                       0x8DCC
#define GL_INT_SAMPLER_2D_RECT                                        0x8DCD
#define GL_INT_SAMPLER_2D_RECT_EXT                                    0x8DCD
#define GL_INT_SAMPLER_1D_ARRAY                                       0x8DCE
#define GL_INT_SAMPLER_1D_ARRAY_EXT                                   0x8DCE
#define GL_INT_SAMPLER_2D_ARRAY                                       0x8DCF
#define GL_INT_SAMPLER_2D_ARRAY_EXT                                   0x8DCF
#define GL_INT_SAMPLER_BUFFER                                         0x8DD0
#define GL_INT_SAMPLER_BUFFER_EXT                                     0x8DD0
#define GL_UNSIGNED_INT_SAMPLER_1D                                    0x8DD1
#define GL_UNSIGNED_INT_SAMPLER_1D_EXT                                0x8DD1
#define GL_UNSIGNED_INT_SAMPLER_2D                                    0x8DD2
#define GL_UNSIGNED_INT_SAMPLER_2D_EXT                                0x8DD2
#define GL_UNSIGNED_INT_SAMPLER_3D                                    0x8DD3
#define GL_UNSIGNED_INT_SAMPLER_3D_EXT                                0x8DD3
#define GL_UNSIGNED_INT_SAMPLER_CUBE                                  0x8DD4
#define GL_UNSIGNED_INT_SAMPLER_CUBE_EXT                              0x8DD4
#define GL_UNSIGNED_INT_SAMPLER_2D_RECT                               0x8DD5
#define GL_UNSIGNED_INT_SAMPLER_2D_RECT_EXT                           0x8DD5
#define GL_UNSIGNED_INT_SAMPLER_1D_ARRAY                              0x8DD6
#define GL_UNSIGNED_INT_SAMPLER_1D_ARRAY_EXT                          0x8DD6
#define GL_UNSIGNED_INT_SAMPLER_2D_ARRAY                              0x8DD7
#define GL_UNSIGNED_INT_SAMPLER_2D_ARRAY_EXT                          0x8DD7
#define GL_UNSIGNED_INT_SAMPLER_BUFFER                                0x8DD8
#define GL_UNSIGNED_INT_SAMPLER_BUFFER_EXT                            0x8DD8
#define GL_GEOMETRY_SHADER                                            0x8DD9
#define GL_GEOMETRY_SHADER_ARB                                        0x8DD9
#define GL_GEOMETRY_SHADER_EXT                                        0x8DD9
#define GL_GEOMETRY_VERTICES_OUT_ARB                                  0x8DDA
#define GL_GEOMETRY_VERTICES_OUT_EXT                                  0x8DDA
#define GL_GEOMETRY_INPUT_TYPE_ARB                                    0x8DDB
#define GL_GEOMETRY_INPUT_TYPE_EXT                                    0x8DDB
#define GL_GEOMETRY_OUTPUT_TYPE_ARB                                   0x8DDC
#define GL_GEOMETRY_OUTPUT_TYPE_EXT                                   0x8DDC
#define GL_MAX_GEOMETRY_VARYING_COMPONENTS_ARB                        0x8DDD
#define GL_MAX_GEOMETRY_VARYING_COMPONENTS_EXT                        0x8DDD
#define GL_MAX_VERTEX_VARYING_COMPONENTS_ARB                          0x8DDE
#define GL_MAX_VERTEX_VARYING_COMPONENTS_EXT                          0x8DDE
#define GL_MAX_GEOMETRY_UNIFORM_COMPONENTS                            0x8DDF
#define GL_MAX_GEOMETRY_UNIFORM_COMPONENTS_ARB                        0x8DDF
#define GL_MAX_GEOMETRY_UNIFORM_COMPONENTS_EXT                        0x8DDF
#define GL_MAX_GEOMETRY_OUTPUT_VERTICES                               0x8DE0
#define GL_MAX_GEOMETRY_OUTPUT_VERTICES_ARB                           0x8DE0
#define GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT                           0x8DE0
#define GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS                       0x8DE1
#define GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS_ARB                   0x8DE1
#define GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS_EXT                   0x8DE1
#define GL_MAX_VERTEX_BINDABLE_UNIFORMS_EXT                           0x8DE2
#define GL_MAX_FRAGMENT_BINDABLE_UNIFORMS_EXT                         0x8DE3
#define GL_MAX_GEOMETRY_BINDABLE_UNIFORMS_EXT                         0x8DE4
#define GL_ACTIVE_SUBROUTINES                                         0x8DE5
#define GL_ACTIVE_SUBROUTINE_UNIFORMS                                 0x8DE6
#define GL_MAX_SUBROUTINES                                            0x8DE7
#define GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS                           0x8DE8
#define GL_NAMED_STRING_LENGTH_ARB                                    0x8DE9
#define GL_NAMED_STRING_TYPE_ARB                                      0x8DEA
#define GL_MAX_BINDABLE_UNIFORM_SIZE_EXT                              0x8DED
#define GL_UNIFORM_BUFFER_EXT                                         0x8DEE
#define GL_UNIFORM_BUFFER_BINDING_EXT                                 0x8DEF
#define GL_LOW_FLOAT                                                  0x8DF0
#define GL_MEDIUM_FLOAT                                               0x8DF1
#define GL_HIGH_FLOAT                                                 0x8DF2
#define GL_LOW_INT                                                    0x8DF3
#define GL_MEDIUM_INT                                                 0x8DF4
#define GL_HIGH_INT                                                   0x8DF5
#define GL_SHADER_BINARY_FORMATS                                      0x8DF8
#define GL_NUM_SHADER_BINARY_FORMATS                                  0x8DF9
#define GL_SHADER_COMPILER                                            0x8DFA
#define GL_MAX_VERTEX_UNIFORM_VECTORS                                 0x8DFB
#define GL_MAX_VARYING_VECTORS                                        0x8DFC
#define GL_MAX_FRAGMENT_UNIFORM_VECTORS                               0x8DFD
#define GL_RENDERBUFFER_COLOR_SAMPLES_NV                              0x8E10
#define GL_MAX_MULTISAMPLE_COVERAGE_MODES_NV                          0x8E11
#define GL_MULTISAMPLE_COVERAGE_MODES_NV                              0x8E12
#define GL_QUERY_WAIT                                                 0x8E13
#define GL_QUERY_WAIT_NV                                              0x8E13
#define GL_QUERY_NO_WAIT                                              0x8E14
#define GL_QUERY_NO_WAIT_NV                                           0x8E14
#define GL_QUERY_BY_REGION_WAIT                                       0x8E15
#define GL_QUERY_BY_REGION_WAIT_NV                                    0x8E15
#define GL_QUERY_BY_REGION_NO_WAIT                                    0x8E16
#define GL_QUERY_BY_REGION_NO_WAIT_NV                                 0x8E16
#define GL_QUERY_WAIT_INVERTED                                        0x8E17
#define GL_QUERY_NO_WAIT_INVERTED                                     0x8E18
#define GL_QUERY_BY_REGION_WAIT_INVERTED                              0x8E19
#define GL_QUERY_BY_REGION_NO_WAIT_INVERTED                           0x8E1A
#define GL_POLYGON_OFFSET_CLAMP                                       0x8E1B
#define GL_POLYGON_OFFSET_CLAMP_EXT                                   0x8E1B
#define GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS               0x8E1E
#define GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS            0x8E1F
#define GL_COLOR_SAMPLES_NV                                           0x8E20
#define GL_TRANSFORM_FEEDBACK                                         0x8E22
#define GL_TRANSFORM_FEEDBACK_NV                                      0x8E22
#define GL_TRANSFORM_FEEDBACK_BUFFER_PAUSED                           0x8E23
#define GL_TRANSFORM_FEEDBACK_PAUSED                                  0x8E23
#define GL_TRANSFORM_FEEDBACK_BUFFER_PAUSED_NV                        0x8E23
#define GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE                           0x8E24
#define GL_TRANSFORM_FEEDBACK_ACTIVE                                  0x8E24
#define GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE_NV                        0x8E24
#define GL_TRANSFORM_FEEDBACK_BINDING                                 0x8E25
#define GL_TRANSFORM_FEEDBACK_BINDING_NV                              0x8E25
#define GL_FRAME_NV                                                   0x8E26
#define GL_FIELDS_NV                                                  0x8E27
#define GL_CURRENT_TIME_NV                                            0x8E28
#define GL_TIMESTAMP                                                  0x8E28
#define GL_NUM_FILL_STREAMS_NV                                        0x8E29
#define GL_PRESENT_TIME_NV                                            0x8E2A
#define GL_PRESENT_DURATION_NV                                        0x8E2B
#define GL_PROGRAM_MATRIX_EXT                                         0x8E2D
#define GL_TRANSPOSE_PROGRAM_MATRIX_EXT                               0x8E2E
#define GL_PROGRAM_MATRIX_STACK_DEPTH_EXT                             0x8E2F
#define GL_TEXTURE_SWIZZLE_R                                          0x8E42
#define GL_TEXTURE_SWIZZLE_R_EXT                                      0x8E42
#define GL_TEXTURE_SWIZZLE_G                                          0x8E43
#define GL_TEXTURE_SWIZZLE_G_EXT                                      0x8E43
#define GL_TEXTURE_SWIZZLE_B                                          0x8E44
#define GL_TEXTURE_SWIZZLE_B_EXT                                      0x8E44
#define GL_TEXTURE_SWIZZLE_A                                          0x8E45
#define GL_TEXTURE_SWIZZLE_A_EXT                                      0x8E45
#define GL_TEXTURE_SWIZZLE_RGBA                                       0x8E46
#define GL_TEXTURE_SWIZZLE_RGBA_EXT                                   0x8E46
#define GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS                        0x8E47
#define GL_ACTIVE_SUBROUTINE_MAX_LENGTH                               0x8E48
#define GL_ACTIVE_SUBROUTINE_UNIFORM_MAX_LENGTH                       0x8E49
#define GL_NUM_COMPATIBLE_SUBROUTINES                                 0x8E4A
#define GL_COMPATIBLE_SUBROUTINES                                     0x8E4B
#define GL_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION                   0x8E4C
#define GL_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION_EXT               0x8E4C
#define GL_FIRST_VERTEX_CONVENTION                                    0x8E4D
#define GL_FIRST_VERTEX_CONVENTION_EXT                                0x8E4D
#define GL_LAST_VERTEX_CONVENTION                                     0x8E4E
#define GL_LAST_VERTEX_CONVENTION_EXT                                 0x8E4E
#define GL_PROVOKING_VERTEX                                           0x8E4F
#define GL_PROVOKING_VERTEX_EXT                                       0x8E4F
#define GL_SAMPLE_POSITION                                            0x8E50
#define GL_SAMPLE_POSITION_NV                                         0x8E50
#define GL_SAMPLE_LOCATION_ARB                                        0x8E50
#define GL_SAMPLE_LOCATION_NV                                         0x8E50
#define GL_SAMPLE_MASK                                                0x8E51
#define GL_SAMPLE_MASK_NV                                             0x8E51
#define GL_SAMPLE_MASK_VALUE                                          0x8E52
#define GL_SAMPLE_MASK_VALUE_NV                                       0x8E52
#define GL_TEXTURE_BINDING_RENDERBUFFER_NV                            0x8E53
#define GL_TEXTURE_RENDERBUFFER_DATA_STORE_BINDING_NV                 0x8E54
#define GL_TEXTURE_RENDERBUFFER_NV                                    0x8E55
#define GL_SAMPLER_RENDERBUFFER_NV                                    0x8E56
#define GL_INT_SAMPLER_RENDERBUFFER_NV                                0x8E57
#define GL_UNSIGNED_INT_SAMPLER_RENDERBUFFER_NV                       0x8E58
#define GL_MAX_SAMPLE_MASK_WORDS                                      0x8E59
#define GL_MAX_SAMPLE_MASK_WORDS_NV                                   0x8E59
#define GL_MAX_GEOMETRY_PROGRAM_INVOCATIONS_NV                        0x8E5A
#define GL_MAX_GEOMETRY_SHADER_INVOCATIONS                            0x8E5A
#define GL_MIN_FRAGMENT_INTERPOLATION_OFFSET                          0x8E5B
#define GL_MIN_FRAGMENT_INTERPOLATION_OFFSET_NV                       0x8E5B
#define GL_MAX_FRAGMENT_INTERPOLATION_OFFSET                          0x8E5C
#define GL_MAX_FRAGMENT_INTERPOLATION_OFFSET_NV                       0x8E5C
#define GL_FRAGMENT_INTERPOLATION_OFFSET_BITS                         0x8E5D
#define GL_FRAGMENT_PROGRAM_INTERPOLATION_OFFSET_BITS_NV              0x8E5D
#define GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET                          0x8E5E
#define GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET_ARB                      0x8E5E
#define GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET_NV                       0x8E5E
#define GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET                          0x8E5F
#define GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET_ARB                      0x8E5F
#define GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET_NV                       0x8E5F
#define GL_MAX_MESH_UNIFORM_BLOCKS_NV                                 0x8E60
#define GL_MAX_MESH_TEXTURE_IMAGE_UNITS_NV                            0x8E61
#define GL_MAX_MESH_IMAGE_UNIFORMS_NV                                 0x8E62
#define GL_MAX_MESH_UNIFORM_COMPONENTS_NV                             0x8E63
#define GL_MAX_MESH_ATOMIC_COUNTER_BUFFERS_NV                         0x8E64
#define GL_MAX_MESH_ATOMIC_COUNTERS_NV                                0x8E65
#define GL_MAX_MESH_SHADER_STORAGE_BLOCKS_NV                          0x8E66
#define GL_MAX_COMBINED_MESH_UNIFORM_COMPONENTS_NV                    0x8E67
#define GL_MAX_TASK_UNIFORM_BLOCKS_NV                                 0x8E68
#define GL_MAX_TASK_TEXTURE_IMAGE_UNITS_NV                            0x8E69
#define GL_MAX_TASK_IMAGE_UNIFORMS_NV                                 0x8E6A
#define GL_MAX_TASK_UNIFORM_COMPONENTS_NV                             0x8E6B
#define GL_MAX_TASK_ATOMIC_COUNTER_BUFFERS_NV                         0x8E6C
#define GL_MAX_TASK_ATOMIC_COUNTERS_NV                                0x8E6D
#define GL_MAX_TASK_SHADER_STORAGE_BLOCKS_NV                          0x8E6E
#define GL_MAX_COMBINED_TASK_UNIFORM_COMPONENTS_NV                    0x8E6F
#define GL_MAX_TRANSFORM_FEEDBACK_BUFFERS                             0x8E70
#define GL_MAX_VERTEX_STREAMS                                         0x8E71
#define GL_PATCH_VERTICES                                             0x8E72
#define GL_PATCH_DEFAULT_INNER_LEVEL                                  0x8E73
#define GL_PATCH_DEFAULT_OUTER_LEVEL                                  0x8E74
#define GL_TESS_CONTROL_OUTPUT_VERTICES                               0x8E75
#define GL_TESS_GEN_MODE                                              0x8E76
#define GL_TESS_GEN_SPACING                                           0x8E77
#define GL_TESS_GEN_VERTEX_ORDER                                      0x8E78
#define GL_TESS_GEN_POINT_MODE                                        0x8E79
#define GL_ISOLINES                                                   0x8E7A
#define GL_FRACTIONAL_ODD                                             0x8E7B
#define GL_FRACTIONAL_EVEN                                            0x8E7C
#define GL_MAX_PATCH_VERTICES                                         0x8E7D
#define GL_MAX_TESS_GEN_LEVEL                                         0x8E7E
#define GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS                        0x8E7F
#define GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS                     0x8E80
#define GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS                       0x8E81
#define GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS                    0x8E82
#define GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS                         0x8E83
#define GL_MAX_TESS_PATCH_COMPONENTS                                  0x8E84
#define GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS                   0x8E85
#define GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS                      0x8E86
#define GL_TESS_EVALUATION_SHADER                                     0x8E87
#define GL_TESS_CONTROL_SHADER                                        0x8E88
#define GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS                            0x8E89
#define GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS                         0x8E8A
#define GL_COMPRESSED_RGBA_BPTC_UNORM                                 0x8E8C
#define GL_COMPRESSED_RGBA_BPTC_UNORM_ARB                             0x8E8C
#define GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM                           0x8E8D
#define GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB                       0x8E8D
#define GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT                           0x8E8E
#define GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB                       0x8E8E
#define GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT                         0x8E8F
#define GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB                     0x8E8F
#define GL_INCLUSIVE_EXT                                              0x8F10
#define GL_EXCLUSIVE_EXT                                              0x8F11
#define GL_WINDOW_RECTANGLE_EXT                                       0x8F12
#define GL_WINDOW_RECTANGLE_MODE_EXT                                  0x8F13
#define GL_MAX_WINDOW_RECTANGLES_EXT                                  0x8F14
#define GL_NUM_WINDOW_RECTANGLES_EXT                                  0x8F15
#define GL_BUFFER_GPU_ADDRESS_NV                                      0x8F1D
#define GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV                             0x8F1E
#define GL_ELEMENT_ARRAY_UNIFIED_NV                                   0x8F1F
#define GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV                             0x8F20
#define GL_VERTEX_ARRAY_ADDRESS_NV                                    0x8F21
#define GL_NORMAL_ARRAY_ADDRESS_NV                                    0x8F22
#define GL_COLOR_ARRAY_ADDRESS_NV                                     0x8F23
#define GL_INDEX_ARRAY_ADDRESS_NV                                     0x8F24
#define GL_TEXTURE_COORD_ARRAY_ADDRESS_NV                             0x8F25
#define GL_EDGE_FLAG_ARRAY_ADDRESS_NV                                 0x8F26
#define GL_SECONDARY_COLOR_ARRAY_ADDRESS_NV                           0x8F27
#define GL_FOG_COORD_ARRAY_ADDRESS_NV                                 0x8F28
#define GL_ELEMENT_ARRAY_ADDRESS_NV                                   0x8F29
#define GL_VERTEX_ATTRIB_ARRAY_LENGTH_NV                              0x8F2A
#define GL_VERTEX_ARRAY_LENGTH_NV                                     0x8F2B
#define GL_NORMAL_ARRAY_LENGTH_NV                                     0x8F2C
#define GL_COLOR_ARRAY_LENGTH_NV                                      0x8F2D
#define GL_INDEX_ARRAY_LENGTH_NV                                      0x8F2E
#define GL_TEXTURE_COORD_ARRAY_LENGTH_NV                              0x8F2F
#define GL_EDGE_FLAG_ARRAY_LENGTH_NV                                  0x8F30
#define GL_SECONDARY_COLOR_ARRAY_LENGTH_NV                            0x8F31
#define GL_FOG_COORD_ARRAY_LENGTH_NV                                  0x8F32
#define GL_ELEMENT_ARRAY_LENGTH_NV                                    0x8F33
#define GL_GPU_ADDRESS_NV                                             0x8F34
#define GL_MAX_SHADER_BUFFER_ADDRESS_NV                               0x8F35
#define GL_COPY_READ_BUFFER                                           0x8F36
#define GL_COPY_READ_BUFFER_BINDING                                   0x8F36
#define GL_COPY_WRITE_BUFFER                                          0x8F37
#define GL_COPY_WRITE_BUFFER_BINDING                                  0x8F37
#define GL_MAX_IMAGE_UNITS                                            0x8F38
#define GL_MAX_IMAGE_UNITS_EXT                                        0x8F38
#define GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS              0x8F39
#define GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS_EXT          0x8F39
#define GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES                       0x8F39
#define GL_IMAGE_BINDING_NAME                                         0x8F3A
#define GL_IMAGE_BINDING_NAME_EXT                                     0x8F3A
#define GL_IMAGE_BINDING_LEVEL                                        0x8F3B
#define GL_IMAGE_BINDING_LEVEL_EXT                                    0x8F3B
#define GL_IMAGE_BINDING_LAYERED                                      0x8F3C
#define GL_IMAGE_BINDING_LAYERED_EXT                                  0x8F3C
#define GL_IMAGE_BINDING_LAYER                                        0x8F3D
#define GL_IMAGE_BINDING_LAYER_EXT                                    0x8F3D
#define GL_IMAGE_BINDING_ACCESS                                       0x8F3E
#define GL_IMAGE_BINDING_ACCESS_EXT                                   0x8F3E
#define GL_DRAW_INDIRECT_BUFFER                                       0x8F3F
#define GL_DRAW_INDIRECT_UNIFIED_NV                                   0x8F40
#define GL_DRAW_INDIRECT_ADDRESS_NV                                   0x8F41
#define GL_DRAW_INDIRECT_LENGTH_NV                                    0x8F42
#define GL_DRAW_INDIRECT_BUFFER_BINDING                               0x8F43
#define GL_MAX_PROGRAM_SUBROUTINE_PARAMETERS_NV                       0x8F44
#define GL_MAX_PROGRAM_SUBROUTINE_NUM_NV                              0x8F45
#define GL_DOUBLE_MAT2                                                0x8F46
#define GL_DOUBLE_MAT2_EXT                                            0x8F46
#define GL_DOUBLE_MAT3                                                0x8F47
#define GL_DOUBLE_MAT3_EXT                                            0x8F47
#define GL_DOUBLE_MAT4                                                0x8F48
#define GL_DOUBLE_MAT4_EXT                                            0x8F48
#define GL_DOUBLE_MAT2x3                                              0x8F49
#define GL_DOUBLE_MAT2x3_EXT                                          0x8F49
#define GL_DOUBLE_MAT2x4                                              0x8F4A
#define GL_DOUBLE_MAT2x4_EXT                                          0x8F4A
#define GL_DOUBLE_MAT3x2                                              0x8F4B
#define GL_DOUBLE_MAT3x2_EXT                                          0x8F4B
#define GL_DOUBLE_MAT3x4                                              0x8F4C
#define GL_DOUBLE_MAT3x4_EXT                                          0x8F4C
#define GL_DOUBLE_MAT4x2                                              0x8F4D
#define GL_DOUBLE_MAT4x2_EXT                                          0x8F4D
#define GL_DOUBLE_MAT4x3                                              0x8F4E
#define GL_DOUBLE_MAT4x3_EXT                                          0x8F4E
#define GL_VERTEX_BINDING_BUFFER                                      0x8F4F
#define GL_RED_SNORM                                                  0x8F90
#define GL_RG_SNORM                                                   0x8F91
#define GL_RGB_SNORM                                                  0x8F92
#define GL_RGBA_SNORM                                                 0x8F93
#define GL_R8_SNORM                                                   0x8F94
#define GL_RG8_SNORM                                                  0x8F95
#define GL_RGB8_SNORM                                                 0x8F96
#define GL_RGBA8_SNORM                                                0x8F97
#define GL_R16_SNORM                                                  0x8F98
#define GL_RG16_SNORM                                                 0x8F99
#define GL_RGB16_SNORM                                                0x8F9A
#define GL_RGBA16_SNORM                                               0x8F9B
#define GL_SIGNED_NORMALIZED                                          0x8F9C
#define GL_PRIMITIVE_RESTART                                          0x8F9D
#define GL_PRIMITIVE_RESTART_INDEX                                    0x8F9E
#define GL_MAX_PROGRAM_TEXTURE_GATHER_COMPONENTS_ARB                  0x8F9F
#define GL_SR8_EXT                                                    0x8FBD
#define GL_INT8_NV                                                    0x8FE0
#define GL_INT8_VEC2_NV                                               0x8FE1
#define GL_INT8_VEC3_NV                                               0x8FE2
#define GL_INT8_VEC4_NV                                               0x8FE3
#define GL_INT16_NV                                                   0x8FE4
#define GL_INT16_VEC2_NV                                              0x8FE5
#define GL_INT16_VEC3_NV                                              0x8FE6
#define GL_INT16_VEC4_NV                                              0x8FE7
#define GL_INT64_VEC2_ARB                                             0x8FE9
#define GL_INT64_VEC2_NV                                              0x8FE9
#define GL_INT64_VEC3_ARB                                             0x8FEA
#define GL_INT64_VEC3_NV                                              0x8FEA
#define GL_INT64_VEC4_ARB                                             0x8FEB
#define GL_INT64_VEC4_NV                                              0x8FEB
#define GL_UNSIGNED_INT8_NV                                           0x8FEC
#define GL_UNSIGNED_INT8_VEC2_NV                                      0x8FED
#define GL_UNSIGNED_INT8_VEC3_NV                                      0x8FEE
#define GL_UNSIGNED_INT8_VEC4_NV                                      0x8FEF
#define GL_UNSIGNED_INT16_NV                                          0x8FF0
#define GL_UNSIGNED_INT16_VEC2_NV                                     0x8FF1
#define GL_UNSIGNED_INT16_VEC3_NV                                     0x8FF2
#define GL_UNSIGNED_INT16_VEC4_NV                                     0x8FF3
#define GL_UNSIGNED_INT64_VEC2_ARB                                    0x8FF5
#define GL_UNSIGNED_INT64_VEC2_NV                                     0x8FF5
#define GL_UNSIGNED_INT64_VEC3_ARB                                    0x8FF6
#define GL_UNSIGNED_INT64_VEC3_NV                                     0x8FF6
#define GL_UNSIGNED_INT64_VEC4_ARB                                    0x8FF7
#define GL_UNSIGNED_INT64_VEC4_NV                                     0x8FF7
#define GL_FLOAT16_NV                                                 0x8FF8
#define GL_FLOAT16_VEC2_NV                                            0x8FF9
#define GL_FLOAT16_VEC3_NV                                            0x8FFA
#define GL_FLOAT16_VEC4_NV                                            0x8FFB
#define GL_DOUBLE_VEC2                                                0x8FFC
#define GL_DOUBLE_VEC2_EXT                                            0x8FFC
#define GL_DOUBLE_VEC3                                                0x8FFD
#define GL_DOUBLE_VEC3_EXT                                            0x8FFD
#define GL_DOUBLE_VEC4                                                0x8FFE
#define GL_DOUBLE_VEC4_EXT                                            0x8FFE
#define GL_SAMPLER_BUFFER_AMD                                         0x9001
#define GL_INT_SAMPLER_BUFFER_AMD                                     0x9002
#define GL_UNSIGNED_INT_SAMPLER_BUFFER_AMD                            0x9003
#define GL_TESSELLATION_MODE_AMD                                      0x9004
#define GL_TESSELLATION_FACTOR_AMD                                    0x9005
#define GL_DISCRETE_AMD                                               0x9006
#define GL_CONTINUOUS_AMD                                             0x9007
#define GL_TEXTURE_CUBE_MAP_ARRAY                                     0x9009
#define GL_TEXTURE_CUBE_MAP_ARRAY_ARB                                 0x9009
#define GL_TEXTURE_BINDING_CUBE_MAP_ARRAY                             0x900A
#define GL_TEXTURE_BINDING_CUBE_MAP_ARRAY_ARB                         0x900A
#define GL_PROXY_TEXTURE_CUBE_MAP_ARRAY                               0x900B
#define GL_PROXY_TEXTURE_CUBE_MAP_ARRAY_ARB                           0x900B
#define GL_SAMPLER_CUBE_MAP_ARRAY                                     0x900C
#define GL_SAMPLER_CUBE_MAP_ARRAY_ARB                                 0x900C
#define GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW                              0x900D
#define GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW_ARB                          0x900D
#define GL_INT_SAMPLER_CUBE_MAP_ARRAY                                 0x900E
#define GL_INT_SAMPLER_CUBE_MAP_ARRAY_ARB                             0x900E
#define GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY                        0x900F
#define GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY_ARB                    0x900F
#define GL_ALPHA_SNORM                                                0x9010
#define GL_LUMINANCE_SNORM                                            0x9011
#define GL_LUMINANCE_ALPHA_SNORM                                      0x9012
#define GL_INTENSITY_SNORM                                            0x9013
#define GL_ALPHA8_SNORM                                               0x9014
#define GL_LUMINANCE8_SNORM                                           0x9015
#define GL_LUMINANCE8_ALPHA8_SNORM                                    0x9016
#define GL_INTENSITY8_SNORM                                           0x9017
#define GL_ALPHA16_SNORM                                              0x9018
#define GL_LUMINANCE16_SNORM                                          0x9019
#define GL_LUMINANCE16_ALPHA16_SNORM                                  0x901A
#define GL_INTENSITY16_SNORM                                          0x901B
#define GL_FACTOR_MIN_AMD                                             0x901C
#define GL_FACTOR_MAX_AMD                                             0x901D
#define GL_DEPTH_CLAMP_NEAR_AMD                                       0x901E
#define GL_DEPTH_CLAMP_FAR_AMD                                        0x901F
#define GL_VIDEO_BUFFER_NV                                            0x9020
#define GL_VIDEO_BUFFER_BINDING_NV                                    0x9021
#define GL_FIELD_UPPER_NV                                             0x9022
#define GL_FIELD_LOWER_NV                                             0x9023
#define GL_NUM_VIDEO_CAPTURE_STREAMS_NV                               0x9024
#define GL_NEXT_VIDEO_CAPTURE_BUFFER_STATUS_NV                        0x9025
#define GL_VIDEO_CAPTURE_TO_422_SUPPORTED_NV                          0x9026
#define GL_LAST_VIDEO_CAPTURE_STATUS_NV                               0x9027
#define GL_VIDEO_BUFFER_PITCH_NV                                      0x9028
#define GL_VIDEO_COLOR_CONVERSION_MATRIX_NV                           0x9029
#define GL_VIDEO_COLOR_CONVERSION_MAX_NV                              0x902A
#define GL_VIDEO_COLOR_CONVERSION_MIN_NV                              0x902B
#define GL_VIDEO_COLOR_CONVERSION_OFFSET_NV                           0x902C
#define GL_VIDEO_BUFFER_INTERNAL_FORMAT_NV                            0x902D
#define GL_PARTIAL_SUCCESS_NV                                         0x902E
#define GL_SUCCESS_NV                                                 0x902F
#define GL_FAILURE_NV                                                 0x9030
#define GL_YCBYCR8_422_NV                                             0x9031
#define GL_YCBAYCR8A_4224_NV                                          0x9032
#define GL_Z6Y10Z6CB10Z6Y10Z6CR10_422_NV                              0x9033
#define GL_Z6Y10Z6CB10Z6A10Z6Y10Z6CR10Z6A10_4224_NV                   0x9034
#define GL_Z4Y12Z4CB12Z4Y12Z4CR12_422_NV                              0x9035
#define GL_Z4Y12Z4CB12Z4A12Z4Y12Z4CR12Z4A12_4224_NV                   0x9036
#define GL_Z4Y12Z4CB12Z4CR12_444_NV                                   0x9037
#define GL_VIDEO_CAPTURE_FRAME_WIDTH_NV                               0x9038
#define GL_VIDEO_CAPTURE_FRAME_HEIGHT_NV                              0x9039
#define GL_VIDEO_CAPTURE_FIELD_UPPER_HEIGHT_NV                        0x903A
#define GL_VIDEO_CAPTURE_FIELD_LOWER_HEIGHT_NV                        0x903B
#define GL_VIDEO_CAPTURE_SURFACE_ORIGIN_NV                            0x903C
#define GL_TEXTURE_COVERAGE_SAMPLES_NV                                0x9045
#define GL_TEXTURE_COLOR_SAMPLES_NV                                   0x9046
#define GL_IMAGE_1D                                                   0x904C
#define GL_IMAGE_1D_EXT                                               0x904C
#define GL_IMAGE_2D                                                   0x904D
#define GL_IMAGE_2D_EXT                                               0x904D
#define GL_IMAGE_3D                                                   0x904E
#define GL_IMAGE_3D_EXT                                               0x904E
#define GL_IMAGE_2D_RECT                                              0x904F
#define GL_IMAGE_2D_RECT_EXT                                          0x904F
#define GL_IMAGE_CUBE                                                 0x9050
#define GL_IMAGE_CUBE_EXT                                             0x9050
#define GL_IMAGE_BUFFER                                               0x9051
#define GL_IMAGE_BUFFER_EXT                                           0x9051
#define GL_IMAGE_1D_ARRAY                                             0x9052
#define GL_IMAGE_1D_ARRAY_EXT                                         0x9052
#define GL_IMAGE_2D_ARRAY                                             0x9053
#define GL_IMAGE_2D_ARRAY_EXT                                         0x9053
#define GL_IMAGE_CUBE_MAP_ARRAY                                       0x9054
#define GL_IMAGE_CUBE_MAP_ARRAY_EXT                                   0x9054
#define GL_IMAGE_2D_MULTISAMPLE                                       0x9055
#define GL_IMAGE_2D_MULTISAMPLE_EXT                                   0x9055
#define GL_IMAGE_2D_MULTISAMPLE_ARRAY                                 0x9056
#define GL_IMAGE_2D_MULTISAMPLE_ARRAY_EXT                             0x9056
#define GL_INT_IMAGE_1D                                               0x9057
#define GL_INT_IMAGE_1D_EXT                                           0x9057
#define GL_INT_IMAGE_2D                                               0x9058
#define GL_INT_IMAGE_2D_EXT                                           0x9058
#define GL_INT_IMAGE_3D                                               0x9059
#define GL_INT_IMAGE_3D_EXT                                           0x9059
#define GL_INT_IMAGE_2D_RECT                                          0x905A
#define GL_INT_IMAGE_2D_RECT_EXT                                      0x905A
#define GL_INT_IMAGE_CUBE                                             0x905B
#define GL_INT_IMAGE_CUBE_EXT                                         0x905B
#define GL_INT_IMAGE_BUFFER                                           0x905C
#define GL_INT_IMAGE_BUFFER_EXT                                       0x905C
#define GL_INT_IMAGE_1D_ARRAY                                         0x905D
#define GL_INT_IMAGE_1D_ARRAY_EXT                                     0x905D
#define GL_INT_IMAGE_2D_ARRAY                                         0x905E
#define GL_INT_IMAGE_2D_ARRAY_EXT                                     0x905E
#define GL_INT_IMAGE_CUBE_MAP_ARRAY                                   0x905F
#define GL_INT_IMAGE_CUBE_MAP_ARRAY_EXT                               0x905F
#define GL_INT_IMAGE_2D_MULTISAMPLE                                   0x9060
#define GL_INT_IMAGE_2D_MULTISAMPLE_EXT                               0x9060
#define GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY                             0x9061
#define GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY_EXT                         0x9061
#define GL_UNSIGNED_INT_IMAGE_1D                                      0x9062
#define GL_UNSIGNED_INT_IMAGE_1D_EXT                                  0x9062
#define GL_UNSIGNED_INT_IMAGE_2D                                      0x9063
#define GL_UNSIGNED_INT_IMAGE_2D_EXT                                  0x9063
#define GL_UNSIGNED_INT_IMAGE_3D                                      0x9064
#define GL_UNSIGNED_INT_IMAGE_3D_EXT                                  0x9064
#define GL_UNSIGNED_INT_IMAGE_2D_RECT                                 0x9065
#define GL_UNSIGNED_INT_IMAGE_2D_RECT_EXT                             0x9065
#define GL_UNSIGNED_INT_IMAGE_CUBE                                    0x9066
#define GL_UNSIGNED_INT_IMAGE_CUBE_EXT                                0x9066
#define GL_UNSIGNED_INT_IMAGE_BUFFER                                  0x9067
#define GL_UNSIGNED_INT_IMAGE_BUFFER_EXT                              0x9067
#define GL_UNSIGNED_INT_IMAGE_1D_ARRAY                                0x9068
#define GL_UNSIGNED_INT_IMAGE_1D_ARRAY_EXT                            0x9068
#define GL_UNSIGNED_INT_IMAGE_2D_ARRAY                                0x9069
#define GL_UNSIGNED_INT_IMAGE_2D_ARRAY_EXT                            0x9069
#define GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY                          0x906A
#define GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY_EXT                      0x906A
#define GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE                          0x906B
#define GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_EXT                      0x906B
#define GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY                    0x906C
#define GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY_EXT                0x906C
#define GL_MAX_IMAGE_SAMPLES                                          0x906D
#define GL_MAX_IMAGE_SAMPLES_EXT                                      0x906D
#define GL_IMAGE_BINDING_FORMAT                                       0x906E
#define GL_IMAGE_BINDING_FORMAT_EXT                                   0x906E
#define GL_RGB10_A2UI                                                 0x906F
#define GL_PATH_FORMAT_SVG_NV                                         0x9070
#define GL_PATH_FORMAT_PS_NV                                          0x9071
#define GL_STANDARD_FONT_NAME_NV                                      0x9072
#define GL_SYSTEM_FONT_NAME_NV                                        0x9073
#define GL_FILE_NAME_NV                                               0x9074
#define GL_PATH_STROKE_WIDTH_NV                                       0x9075
#define GL_PATH_END_CAPS_NV                                           0x9076
#define GL_PATH_INITIAL_END_CAP_NV                                    0x9077
#define GL_PATH_TERMINAL_END_CAP_NV                                   0x9078
#define GL_PATH_JOIN_STYLE_NV                                         0x9079
#define GL_PATH_MITER_LIMIT_NV                                        0x907A
#define GL_PATH_DASH_CAPS_NV                                          0x907B
#define GL_PATH_INITIAL_DASH_CAP_NV                                   0x907C
#define GL_PATH_TERMINAL_DASH_CAP_NV                                  0x907D
#define GL_PATH_DASH_OFFSET_NV                                        0x907E
#define GL_PATH_CLIENT_LENGTH_NV                                      0x907F
#define GL_PATH_FILL_MODE_NV                                          0x9080
#define GL_PATH_FILL_MASK_NV                                          0x9081
#define GL_PATH_FILL_COVER_MODE_NV                                    0x9082
#define GL_PATH_STROKE_COVER_MODE_NV                                  0x9083
#define GL_PATH_STROKE_MASK_NV                                        0x9084
#define GL_COUNT_UP_NV                                                0x9088
#define GL_COUNT_DOWN_NV                                              0x9089
#define GL_PATH_OBJECT_BOUNDING_BOX_NV                                0x908A
#define GL_CONVEX_HULL_NV                                             0x908B
#define GL_BOUNDING_BOX_NV                                            0x908D
#define GL_TRANSLATE_X_NV                                             0x908E
#define GL_TRANSLATE_Y_NV                                             0x908F
#define GL_TRANSLATE_2D_NV                                            0x9090
#define GL_TRANSLATE_3D_NV                                            0x9091
#define GL_AFFINE_2D_NV                                               0x9092
#define GL_AFFINE_3D_NV                                               0x9094
#define GL_TRANSPOSE_AFFINE_2D_NV                                     0x9096
#define GL_TRANSPOSE_AFFINE_3D_NV                                     0x9098
#define GL_UTF8_NV                                                    0x909A
#define GL_UTF16_NV                                                   0x909B
#define GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV                          0x909C
#define GL_PATH_COMMAND_COUNT_NV                                      0x909D
#define GL_PATH_COORD_COUNT_NV                                        0x909E
#define GL_PATH_DASH_ARRAY_COUNT_NV                                   0x909F
#define GL_PATH_COMPUTED_LENGTH_NV                                    0x90A0
#define GL_PATH_FILL_BOUNDING_BOX_NV                                  0x90A1
#define GL_PATH_STROKE_BOUNDING_BOX_NV                                0x90A2
#define GL_SQUARE_NV                                                  0x90A3
#define GL_ROUND_NV                                                   0x90A4
#define GL_TRIANGULAR_NV                                              0x90A5
#define GL_BEVEL_NV                                                   0x90A6
#define GL_MITER_REVERT_NV                                            0x90A7
#define GL_MITER_TRUNCATE_NV                                          0x90A8
#define GL_SKIP_MISSING_GLYPH_NV                                      0x90A9
#define GL_USE_MISSING_GLYPH_NV                                       0x90AA
#define GL_PATH_ERROR_POSITION_NV                                     0x90AB
#define GL_PATH_FOG_GEN_MODE_NV                                       0x90AC
#define GL_ACCUM_ADJACENT_PAIRS_NV                                    0x90AD
#define GL_ADJACENT_PAIRS_NV                                          0x90AE
#define GL_FIRST_TO_REST_NV                                           0x90AF
#define GL_PATH_GEN_MODE_NV                                           0x90B0
#define GL_PATH_GEN_COEFF_NV                                          0x90B1
#define GL_PATH_GEN_COLOR_FORMAT_NV                                   0x90B2
#define GL_PATH_GEN_COMPONENTS_NV                                     0x90B3
#define GL_PATH_DASH_OFFSET_RESET_NV                                  0x90B4
#define GL_MOVE_TO_RESETS_NV                                          0x90B5
#define GL_MOVE_TO_CONTINUES_NV                                       0x90B6
#define GL_PATH_STENCIL_FUNC_NV                                       0x90B7
#define GL_PATH_STENCIL_REF_NV                                        0x90B8
#define GL_PATH_STENCIL_VALUE_MASK_NV                                 0x90B9
#define GL_SCALED_RESOLVE_FASTEST_EXT                                 0x90BA
#define GL_SCALED_RESOLVE_NICEST_EXT                                  0x90BB
#define GL_MIN_MAP_BUFFER_ALIGNMENT                                   0x90BC
#define GL_PATH_STENCIL_DEPTH_OFFSET_FACTOR_NV                        0x90BD
#define GL_PATH_STENCIL_DEPTH_OFFSET_UNITS_NV                         0x90BE
#define GL_PATH_COVER_DEPTH_FUNC_NV                                   0x90BF
#define GL_IMAGE_FORMAT_COMPATIBILITY_TYPE                            0x90C7
#define GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE                         0x90C8
#define GL_IMAGE_FORMAT_COMPATIBILITY_BY_CLASS                        0x90C9
#define GL_MAX_VERTEX_IMAGE_UNIFORMS                                  0x90CA
#define GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS                            0x90CB
#define GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS                         0x90CC
#define GL_MAX_GEOMETRY_IMAGE_UNIFORMS                                0x90CD
#define GL_MAX_FRAGMENT_IMAGE_UNIFORMS                                0x90CE
#define GL_MAX_COMBINED_IMAGE_UNIFORMS                                0x90CF
#define GL_MAX_DEEP_3D_TEXTURE_WIDTH_HEIGHT_NV                        0x90D0
#define GL_MAX_DEEP_3D_TEXTURE_DEPTH_NV                               0x90D1
#define GL_SHADER_STORAGE_BUFFER                                      0x90D2
#define GL_SHADER_STORAGE_BUFFER_BINDING                              0x90D3
#define GL_SHADER_STORAGE_BUFFER_START                                0x90D4
#define GL_SHADER_STORAGE_BUFFER_SIZE                                 0x90D5
#define GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS                           0x90D6
#define GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS                         0x90D7
#define GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS                     0x90D8
#define GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS                  0x90D9
#define GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS                         0x90DA
#define GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS                          0x90DB
#define GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS                         0x90DC
#define GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS                         0x90DD
#define GL_MAX_SHADER_STORAGE_BLOCK_SIZE                              0x90DE
#define GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT                     0x90DF
#define GL_SYNC_X11_FENCE_EXT                                         0x90E1
#define GL_DEPTH_STENCIL_TEXTURE_MODE                                 0x90EA
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS                         0x90EB
#define GL_MAX_COMPUTE_FIXED_GROUP_INVOCATIONS_ARB                    0x90EB
#define GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER                 0x90EC
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER         0x90ED
#define GL_DISPATCH_INDIRECT_BUFFER                                   0x90EE
#define GL_DISPATCH_INDIRECT_BUFFER_BINDING                           0x90EF
#define GL_CONTEXT_ROBUST_ACCESS                                      0x90F3
#define GL_CONTEXT_ROBUST_ACCESS_KHR                                  0x90F3
#define GL_COMPUTE_PROGRAM_NV                                         0x90FB
#define GL_COMPUTE_PROGRAM_PARAMETER_BUFFER_NV                        0x90FC
#define GL_TEXTURE_2D_MULTISAMPLE                                     0x9100
#define GL_PROXY_TEXTURE_2D_MULTISAMPLE                               0x9101
#define GL_TEXTURE_2D_MULTISAMPLE_ARRAY                               0x9102
#define GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY                         0x9103
#define GL_TEXTURE_BINDING_2D_MULTISAMPLE                             0x9104
#define GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY                       0x9105
#define GL_TEXTURE_SAMPLES                                            0x9106
#define GL_TEXTURE_FIXED_SAMPLE_LOCATIONS                             0x9107
#define GL_SAMPLER_2D_MULTISAMPLE                                     0x9108
#define GL_INT_SAMPLER_2D_MULTISAMPLE                                 0x9109
#define GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE                        0x910A
#define GL_SAMPLER_2D_MULTISAMPLE_ARRAY                               0x910B
#define GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY                           0x910C
#define GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY                  0x910D
#define GL_MAX_COLOR_TEXTURE_SAMPLES                                  0x910E
#define GL_MAX_DEPTH_TEXTURE_SAMPLES                                  0x910F
#define GL_MAX_INTEGER_SAMPLES                                        0x9110
#define GL_MAX_SERVER_WAIT_TIMEOUT                                    0x9111
#define GL_OBJECT_TYPE                                                0x9112
#define GL_SYNC_CONDITION                                             0x9113
#define GL_SYNC_STATUS                                                0x9114
#define GL_SYNC_FLAGS                                                 0x9115
#define GL_SYNC_FENCE                                                 0x9116
#define GL_SYNC_GPU_COMMANDS_COMPLETE                                 0x9117
#define GL_UNSIGNALED                                                 0x9118
#define GL_SIGNALED                                                   0x9119
#define GL_ALREADY_SIGNALED                                           0x911A
#define GL_TIMEOUT_EXPIRED                                            0x911B
#define GL_CONDITION_SATISFIED                                        0x911C
#define GL_WAIT_FAILED                                                0x911D
#define GL_BUFFER_ACCESS_FLAGS                                        0x911F
#define GL_BUFFER_MAP_LENGTH                                          0x9120
#define GL_BUFFER_MAP_OFFSET                                          0x9121
#define GL_MAX_VERTEX_OUTPUT_COMPONENTS                               0x9122
#define GL_MAX_GEOMETRY_INPUT_COMPONENTS                              0x9123
#define GL_MAX_GEOMETRY_OUTPUT_COMPONENTS                             0x9124
#define GL_MAX_FRAGMENT_INPUT_COMPONENTS                              0x9125
#define GL_CONTEXT_PROFILE_MASK                                       0x9126
#define GL_UNPACK_COMPRESSED_BLOCK_WIDTH                              0x9127
#define GL_UNPACK_COMPRESSED_BLOCK_HEIGHT                             0x9128
#define GL_UNPACK_COMPRESSED_BLOCK_DEPTH                              0x9129
#define GL_UNPACK_COMPRESSED_BLOCK_SIZE                               0x912A
#define GL_PACK_COMPRESSED_BLOCK_WIDTH                                0x912B
#define GL_PACK_COMPRESSED_BLOCK_HEIGHT                               0x912C
#define GL_PACK_COMPRESSED_BLOCK_DEPTH                                0x912D
#define GL_PACK_COMPRESSED_BLOCK_SIZE                                 0x912E
#define GL_TEXTURE_IMMUTABLE_FORMAT                                   0x912F
#define GL_MAX_DEBUG_MESSAGE_LENGTH                                   0x9143
#define GL_MAX_DEBUG_MESSAGE_LENGTH_AMD                               0x9143
#define GL_MAX_DEBUG_MESSAGE_LENGTH_ARB                               0x9143
#define GL_MAX_DEBUG_MESSAGE_LENGTH_KHR                               0x9143
#define GL_MAX_DEBUG_LOGGED_MESSAGES                                  0x9144
#define GL_MAX_DEBUG_LOGGED_MESSAGES_AMD                              0x9144
#define GL_MAX_DEBUG_LOGGED_MESSAGES_ARB                              0x9144
#define GL_MAX_DEBUG_LOGGED_MESSAGES_KHR                              0x9144
#define GL_DEBUG_LOGGED_MESSAGES                                      0x9145
#define GL_DEBUG_LOGGED_MESSAGES_AMD                                  0x9145
#define GL_DEBUG_LOGGED_MESSAGES_ARB                                  0x9145
#define GL_DEBUG_LOGGED_MESSAGES_KHR                                  0x9145
#define GL_DEBUG_SEVERITY_HIGH                                        0x9146
#define GL_DEBUG_SEVERITY_HIGH_AMD                                    0x9146
#define GL_DEBUG_SEVERITY_HIGH_ARB                                    0x9146
#define GL_DEBUG_SEVERITY_HIGH_KHR                                    0x9146
#define GL_DEBUG_SEVERITY_MEDIUM                                      0x9147
#define GL_DEBUG_SEVERITY_MEDIUM_AMD                                  0x9147
#define GL_DEBUG_SEVERITY_MEDIUM_ARB                                  0x9147
#define GL_DEBUG_SEVERITY_MEDIUM_KHR                                  0x9147
#define GL_DEBUG_SEVERITY_LOW                                         0x9148
#define GL_DEBUG_SEVERITY_LOW_AMD                                     0x9148
#define GL_DEBUG_SEVERITY_LOW_ARB                                     0x9148
#define GL_DEBUG_SEVERITY_LOW_KHR                                     0x9148
#define GL_DEBUG_CATEGORY_API_ERROR_AMD                               0x9149
#define GL_DEBUG_CATEGORY_WINDOW_SYSTEM_AMD                           0x914A
#define GL_DEBUG_CATEGORY_DEPRECATION_AMD                             0x914B
#define GL_DEBUG_CATEGORY_UNDEFINED_BEHAVIOR_AMD                      0x914C
#define GL_DEBUG_CATEGORY_PERFORMANCE_AMD                             0x914D
#define GL_DEBUG_CATEGORY_SHADER_COMPILER_AMD                         0x914E
#define GL_DEBUG_CATEGORY_APPLICATION_AMD                             0x914F
#define GL_DEBUG_CATEGORY_OTHER_AMD                                   0x9150
#define GL_BUFFER_OBJECT_EXT                                          0x9151
#define GL_DATA_BUFFER_AMD                                            0x9151
#define GL_PERFORMANCE_MONITOR_AMD                                    0x9152
#define GL_QUERY_OBJECT_AMD                                           0x9153
#define GL_QUERY_OBJECT_EXT                                           0x9153
#define GL_VERTEX_ARRAY_OBJECT_AMD                                    0x9154
#define GL_VERTEX_ARRAY_OBJECT_EXT                                    0x9154
#define GL_SAMPLER_OBJECT_AMD                                         0x9155
#define GL_EXTERNAL_VIRTUAL_MEMORY_BUFFER_AMD                         0x9160
#define GL_QUERY_BUFFER                                               0x9192
#define GL_QUERY_BUFFER_AMD                                           0x9192
#define GL_QUERY_BUFFER_BINDING                                       0x9193
#define GL_QUERY_BUFFER_BINDING_AMD                                   0x9193
#define GL_QUERY_RESULT_NO_WAIT                                       0x9194
#define GL_QUERY_RESULT_NO_WAIT_AMD                                   0x9194
#define GL_VIRTUAL_PAGE_SIZE_X_ARB                                    0x9195
#define GL_VIRTUAL_PAGE_SIZE_X_AMD                                    0x9195
#define GL_VIRTUAL_PAGE_SIZE_Y_ARB                                    0x9196
#define GL_VIRTUAL_PAGE_SIZE_Y_AMD                                    0x9196
#define GL_VIRTUAL_PAGE_SIZE_Z_ARB                                    0x9197
#define GL_VIRTUAL_PAGE_SIZE_Z_AMD                                    0x9197
#define GL_MAX_SPARSE_TEXTURE_SIZE_ARB                                0x9198
#define GL_MAX_SPARSE_TEXTURE_SIZE_AMD                                0x9198
#define GL_MAX_SPARSE_3D_TEXTURE_SIZE_ARB                             0x9199
#define GL_MAX_SPARSE_3D_TEXTURE_SIZE_AMD                             0x9199
#define GL_MAX_SPARSE_ARRAY_TEXTURE_LAYERS                            0x919A
#define GL_MAX_SPARSE_ARRAY_TEXTURE_LAYERS_ARB                        0x919A
#define GL_MIN_SPARSE_LEVEL_AMD                                       0x919B
#define GL_MIN_LOD_WARNING_AMD                                        0x919C
#define GL_TEXTURE_BUFFER_OFFSET                                      0x919D
#define GL_TEXTURE_BUFFER_SIZE                                        0x919E
#define GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT                            0x919F
#define GL_STREAM_RASTERIZATION_AMD                                   0x91A0
#define GL_VERTEX_ELEMENT_SWIZZLE_AMD                                 0x91A4
#define GL_VERTEX_ID_SWIZZLE_AMD                                      0x91A5
#define GL_TEXTURE_SPARSE_ARB                                         0x91A6
#define GL_VIRTUAL_PAGE_SIZE_INDEX_ARB                                0x91A7
#define GL_NUM_VIRTUAL_PAGE_SIZES_ARB                                 0x91A8
#define GL_SPARSE_TEXTURE_FULL_ARRAY_CUBE_MIPMAPS_ARB                 0x91A9
#define GL_NUM_SPARSE_LEVELS_ARB                                      0x91AA
#define GL_PIXELS_PER_SAMPLE_PATTERN_X_AMD                            0x91AE
#define GL_PIXELS_PER_SAMPLE_PATTERN_Y_AMD                            0x91AF
#define GL_MAX_SHADER_COMPILER_THREADS_KHR                            0x91B0
#define GL_MAX_SHADER_COMPILER_THREADS_ARB                            0x91B0
#define GL_COMPLETION_STATUS_KHR                                      0x91B1
#define GL_COMPLETION_STATUS_ARB                                      0x91B1
#define GL_RENDERBUFFER_STORAGE_SAMPLES_AMD                           0x91B2
#define GL_MAX_COLOR_FRAMEBUFFER_SAMPLES_AMD                          0x91B3
#define GL_MAX_COLOR_FRAMEBUFFER_STORAGE_SAMPLES_AMD                  0x91B4
#define GL_MAX_DEPTH_STENCIL_FRAMEBUFFER_SAMPLES_AMD                  0x91B5
#define GL_NUM_SUPPORTED_MULTISAMPLE_MODES_AMD                        0x91B6
#define GL_SUPPORTED_MULTISAMPLE_MODES_AMD                            0x91B7
#define GL_COMPUTE_SHADER                                             0x91B9
#define GL_MAX_COMPUTE_UNIFORM_BLOCKS                                 0x91BB
#define GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS                            0x91BC
#define GL_MAX_COMPUTE_IMAGE_UNIFORMS                                 0x91BD
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT                               0x91BE
#define GL_MAX_COMPUTE_WORK_GROUP_SIZE                                0x91BF
#define GL_MAX_COMPUTE_FIXED_GROUP_SIZE_ARB                           0x91BF
#define GL_FLOAT16_MAT2_AMD                                           0x91C5
#define GL_FLOAT16_MAT3_AMD                                           0x91C6
#define GL_FLOAT16_MAT4_AMD                                           0x91C7
#define GL_FLOAT16_MAT2x3_AMD                                         0x91C8
#define GL_FLOAT16_MAT2x4_AMD                                         0x91C9
#define GL_FLOAT16_MAT3x2_AMD                                         0x91CA
#define GL_FLOAT16_MAT3x4_AMD                                         0x91CB
#define GL_FLOAT16_MAT4x2_AMD                                         0x91CC
#define GL_FLOAT16_MAT4x3_AMD                                         0x91CD
#define GL_COMPRESSED_R11_EAC                                         0x9270
#define GL_COMPRESSED_SIGNED_R11_EAC                                  0x9271
#define GL_COMPRESSED_RG11_EAC                                        0x9272
#define GL_COMPRESSED_SIGNED_RG11_EAC                                 0x9273
#define GL_COMPRESSED_RGB8_ETC2                                       0x9274
#define GL_COMPRESSED_SRGB8_ETC2                                      0x9275
#define GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2                   0x9276
#define GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2                  0x9277
#define GL_COMPRESSED_RGBA8_ETC2_EAC                                  0x9278
#define GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC                           0x9279
#define GL_BLEND_PREMULTIPLIED_SRC_NV                                 0x9280
#define GL_BLEND_OVERLAP_NV                                           0x9281
#define GL_UNCORRELATED_NV                                            0x9282
#define GL_DISJOINT_NV                                                0x9283
#define GL_CONJOINT_NV                                                0x9284
#define GL_BLEND_ADVANCED_COHERENT_KHR                                0x9285
#define GL_BLEND_ADVANCED_COHERENT_NV                                 0x9285
#define GL_SRC_NV                                                     0x9286
#define GL_DST_NV                                                     0x9287
#define GL_SRC_OVER_NV                                                0x9288
#define GL_DST_OVER_NV                                                0x9289
#define GL_SRC_IN_NV                                                  0x928A
#define GL_DST_IN_NV                                                  0x928B
#define GL_SRC_OUT_NV                                                 0x928C
#define GL_DST_OUT_NV                                                 0x928D
#define GL_SRC_ATOP_NV                                                0x928E
#define GL_DST_ATOP_NV                                                0x928F
#define GL_PLUS_NV                                                    0x9291
#define GL_PLUS_DARKER_NV                                             0x9292
#define GL_MULTIPLY_KHR                                               0x9294
#define GL_MULTIPLY_NV                                                0x9294
#define GL_SCREEN_KHR                                                 0x9295
#define GL_SCREEN_NV                                                  0x9295
#define GL_OVERLAY_KHR                                                0x9296
#define GL_OVERLAY_NV                                                 0x9296
#define GL_DARKEN_KHR                                                 0x9297
#define GL_DARKEN_NV                                                  0x9297
#define GL_LIGHTEN_KHR                                                0x9298
#define GL_LIGHTEN_NV                                                 0x9298
#define GL_COLORDODGE_KHR                                             0x9299
#define GL_COLORDODGE_NV                                              0x9299
#define GL_COLORBURN_KHR                                              0x929A
#define GL_COLORBURN_NV                                               0x929A
#define GL_HARDLIGHT_KHR                                              0x929B
#define GL_HARDLIGHT_NV                                               0x929B
#define GL_SOFTLIGHT_KHR                                              0x929C
#define GL_SOFTLIGHT_NV                                               0x929C
#define GL_DIFFERENCE_KHR                                             0x929E
#define GL_DIFFERENCE_NV                                              0x929E
#define GL_MINUS_NV                                                   0x929F
#define GL_EXCLUSION_KHR                                              0x92A0
#define GL_EXCLUSION_NV                                               0x92A0
#define GL_CONTRAST_NV                                                0x92A1
#define GL_INVERT_RGB_NV                                              0x92A3
#define GL_LINEARDODGE_NV                                             0x92A4
#define GL_LINEARBURN_NV                                              0x92A5
#define GL_VIVIDLIGHT_NV                                              0x92A6
#define GL_LINEARLIGHT_NV                                             0x92A7
#define GL_PINLIGHT_NV                                                0x92A8
#define GL_HARDMIX_NV                                                 0x92A9
#define GL_HSL_HUE_KHR                                                0x92AD
#define GL_HSL_HUE_NV                                                 0x92AD
#define GL_HSL_SATURATION_KHR                                         0x92AE
#define GL_HSL_SATURATION_NV                                          0x92AE
#define GL_HSL_COLOR_KHR                                              0x92AF
#define GL_HSL_COLOR_NV                                               0x92AF
#define GL_HSL_LUMINOSITY_KHR                                         0x92B0
#define GL_HSL_LUMINOSITY_NV                                          0x92B0
#define GL_PLUS_CLAMPED_NV                                            0x92B1
#define GL_PLUS_CLAMPED_ALPHA_NV                                      0x92B2
#define GL_MINUS_CLAMPED_NV                                           0x92B3
#define GL_INVERT_OVG_NV                                              0x92B4
#define GL_MULTICAST_GPUS_NV                                          0x92BA
#define GL_PURGED_CONTEXT_RESET_NV                                    0x92BB
#define GL_PRIMITIVE_BOUNDING_BOX_ARB                                 0x92BE
#define GL_ALPHA_TO_COVERAGE_DITHER_MODE_NV                           0x92BF
#define GL_ATOMIC_COUNTER_BUFFER                                      0x92C0
#define GL_ATOMIC_COUNTER_BUFFER_BINDING                              0x92C1
#define GL_ATOMIC_COUNTER_BUFFER_START                                0x92C2
#define GL_ATOMIC_COUNTER_BUFFER_SIZE                                 0x92C3
#define GL_ATOMIC_COUNTER_BUFFER_DATA_SIZE                            0x92C4
#define GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS               0x92C5
#define GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES        0x92C6
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER          0x92C7
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER    0x92C8
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER 0x92C9
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER        0x92CA
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER        0x92CB
#define GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS                          0x92CC
#define GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS                    0x92CD
#define GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS                 0x92CE
#define GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS                        0x92CF
#define GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS                        0x92D0
#define GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS                        0x92D1
#define GL_MAX_VERTEX_ATOMIC_COUNTERS                                 0x92D2
#define GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS                           0x92D3
#define GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS                        0x92D4
#define GL_MAX_GEOMETRY_ATOMIC_COUNTERS                               0x92D5
#define GL_MAX_FRAGMENT_ATOMIC_COUNTERS                               0x92D6
#define GL_MAX_COMBINED_ATOMIC_COUNTERS                               0x92D7
#define GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE                             0x92D8
#define GL_ACTIVE_ATOMIC_COUNTER_BUFFERS                              0x92D9
#define GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX                        0x92DA
#define GL_UNSIGNED_INT_ATOMIC_COUNTER                                0x92DB
#define GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS                         0x92DC
#define GL_FRAGMENT_COVERAGE_TO_COLOR_NV                              0x92DD
#define GL_FRAGMENT_COVERAGE_COLOR_NV                                 0x92DE
#define GL_MESH_OUTPUT_PER_VERTEX_GRANULARITY_NV                      0x92DF
#define GL_DEBUG_OUTPUT                                               0x92E0
#define GL_DEBUG_OUTPUT_KHR                                           0x92E0
#define GL_UNIFORM                                                    0x92E1
#define GL_UNIFORM_BLOCK                                              0x92E2
#define GL_PROGRAM_INPUT                                              0x92E3
#define GL_PROGRAM_OUTPUT                                             0x92E4
#define GL_BUFFER_VARIABLE                                            0x92E5
#define GL_SHADER_STORAGE_BLOCK                                       0x92E6
#define GL_IS_PER_PATCH                                               0x92E7
#define GL_VERTEX_SUBROUTINE                                          0x92E8
#define GL_TESS_CONTROL_SUBROUTINE                                    0x92E9
#define GL_TESS_EVALUATION_SUBROUTINE                                 0x92EA
#define GL_GEOMETRY_SUBROUTINE                                        0x92EB
#define GL_FRAGMENT_SUBROUTINE                                        0x92EC
#define GL_COMPUTE_SUBROUTINE                                         0x92ED
#define GL_VERTEX_SUBROUTINE_UNIFORM                                  0x92EE
#define GL_TESS_CONTROL_SUBROUTINE_UNIFORM                            0x92EF
#define GL_TESS_EVALUATION_SUBROUTINE_UNIFORM                         0x92F0
#define GL_GEOMETRY_SUBROUTINE_UNIFORM                                0x92F1
#define GL_FRAGMENT_SUBROUTINE_UNIFORM                                0x92F2
#define GL_COMPUTE_SUBROUTINE_UNIFORM                                 0x92F3
#define GL_TRANSFORM_FEEDBACK_VARYING                                 0x92F4
#define GL_ACTIVE_RESOURCES                                           0x92F5
#define GL_MAX_NAME_LENGTH                                            0x92F6
#define GL_MAX_NUM_ACTIVE_VARIABLES                                   0x92F7
#define GL_MAX_NUM_COMPATIBLE_SUBROUTINES                             0x92F8
#define GL_NAME_LENGTH                                                0x92F9
#define GL_TYPE                                                       0x92FA
#define GL_ARRAY_SIZE                                                 0x92FB
#define GL_OFFSET                                                     0x92FC
#define GL_BLOCK_INDEX                                                0x92FD
#define GL_ARRAY_STRIDE                                               0x92FE
#define GL_MATRIX_STRIDE                                              0x92FF
#define GL_IS_ROW_MAJOR                                               0x9300
#define GL_ATOMIC_COUNTER_BUFFER_INDEX                                0x9301
#define GL_BUFFER_BINDING                                             0x9302
#define GL_BUFFER_DATA_SIZE                                           0x9303
#define GL_NUM_ACTIVE_VARIABLES                                       0x9304
#define GL_ACTIVE_VARIABLES                                           0x9305
#define GL_REFERENCED_BY_VERTEX_SHADER                                0x9306
#define GL_REFERENCED_BY_TESS_CONTROL_SHADER                          0x9307
#define GL_REFERENCED_BY_TESS_EVALUATION_SHADER                       0x9308
#define GL_REFERENCED_BY_GEOMETRY_SHADER                              0x9309
#define GL_REFERENCED_BY_FRAGMENT_SHADER                              0x930A
#define GL_REFERENCED_BY_COMPUTE_SHADER                               0x930B
#define GL_TOP_LEVEL_ARRAY_SIZE                                       0x930C
#define GL_TOP_LEVEL_ARRAY_STRIDE                                     0x930D
#define GL_LOCATION                                                   0x930E
#define GL_LOCATION_INDEX                                             0x930F
#define GL_FRAMEBUFFER_DEFAULT_WIDTH                                  0x9310
#define GL_FRAMEBUFFER_DEFAULT_HEIGHT                                 0x9311
#define GL_FRAMEBUFFER_DEFAULT_LAYERS                                 0x9312
#define GL_FRAMEBUFFER_DEFAULT_SAMPLES                                0x9313
#define GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS                 0x9314
#define GL_MAX_FRAMEBUFFER_WIDTH                                      0x9315
#define GL_MAX_FRAMEBUFFER_HEIGHT                                     0x9316
#define GL_MAX_FRAMEBUFFER_LAYERS                                     0x9317
#define GL_MAX_FRAMEBUFFER_SAMPLES                                    0x9318
#define GL_RASTER_MULTISAMPLE_EXT                                     0x9327
#define GL_RASTER_SAMPLES_EXT                                         0x9328
#define GL_MAX_RASTER_SAMPLES_EXT                                     0x9329
#define GL_RASTER_FIXED_SAMPLE_LOCATIONS_EXT                          0x932A
#define GL_MULTISAMPLE_RASTERIZATION_ALLOWED_EXT                      0x932B
#define GL_EFFECTIVE_RASTER_SAMPLES_EXT                               0x932C
#define GL_DEPTH_SAMPLES_NV                                           0x932D
#define GL_STENCIL_SAMPLES_NV                                         0x932E
#define GL_MIXED_DEPTH_SAMPLES_SUPPORTED_NV                           0x932F
#define GL_MIXED_STENCIL_SAMPLES_SUPPORTED_NV                         0x9330
#define GL_COVERAGE_MODULATION_TABLE_NV                               0x9331
#define GL_COVERAGE_MODULATION_NV                                     0x9332
#define GL_COVERAGE_MODULATION_TABLE_SIZE_NV                          0x9333
#define GL_WARP_SIZE_NV                                               0x9339
#define GL_WARPS_PER_SM_NV                                            0x933A
#define GL_SM_COUNT_NV                                                0x933B
#define GL_FILL_RECTANGLE_NV                                          0x933C
#define GL_SAMPLE_LOCATION_SUBPIXEL_BITS_ARB                          0x933D
#define GL_SAMPLE_LOCATION_SUBPIXEL_BITS_NV                           0x933D
#define GL_SAMPLE_LOCATION_PIXEL_GRID_WIDTH_ARB                       0x933E
#define GL_SAMPLE_LOCATION_PIXEL_GRID_WIDTH_NV                        0x933E
#define GL_SAMPLE_LOCATION_PIXEL_GRID_HEIGHT_ARB                      0x933F
#define GL_SAMPLE_LOCATION_PIXEL_GRID_HEIGHT_NV                       0x933F
#define GL_PROGRAMMABLE_SAMPLE_LOCATION_TABLE_SIZE_ARB                0x9340
#define GL_PROGRAMMABLE_SAMPLE_LOCATION_TABLE_SIZE_NV                 0x9340
#define GL_PROGRAMMABLE_SAMPLE_LOCATION_ARB                           0x9341
#define GL_PROGRAMMABLE_SAMPLE_LOCATION_NV                            0x9341
#define GL_FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_ARB              0x9342
#define GL_FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_NV               0x9342
#define GL_FRAMEBUFFER_SAMPLE_LOCATION_PIXEL_GRID_ARB                 0x9343
#define GL_FRAMEBUFFER_SAMPLE_LOCATION_PIXEL_GRID_NV                  0x9343
#define GL_MAX_COMPUTE_VARIABLE_GROUP_INVOCATIONS_ARB                 0x9344
#define GL_MAX_COMPUTE_VARIABLE_GROUP_SIZE_ARB                        0x9345
#define GL_CONSERVATIVE_RASTERIZATION_NV                              0x9346
#define GL_SUBPIXEL_PRECISION_BIAS_X_BITS_NV                          0x9347
#define GL_SUBPIXEL_PRECISION_BIAS_Y_BITS_NV                          0x9348
#define GL_MAX_SUBPIXEL_PRECISION_BIAS_BITS_NV                        0x9349
#define GL_LOCATION_COMPONENT                                         0x934A
#define GL_TRANSFORM_FEEDBACK_BUFFER_INDEX                            0x934B
#define GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE                           0x934C
#define GL_ALPHA_TO_COVERAGE_DITHER_DEFAULT_NV                        0x934D
#define GL_ALPHA_TO_COVERAGE_DITHER_ENABLE_NV                         0x934E
#define GL_ALPHA_TO_COVERAGE_DITHER_DISABLE_NV                        0x934F
#define GL_VIEWPORT_SWIZZLE_POSITIVE_X_NV                             0x9350
#define GL_VIEWPORT_SWIZZLE_NEGATIVE_X_NV                             0x9351
#define GL_VIEWPORT_SWIZZLE_POSITIVE_Y_NV                             0x9352
#define GL_VIEWPORT_SWIZZLE_NEGATIVE_Y_NV                             0x9353
#define GL_VIEWPORT_SWIZZLE_POSITIVE_Z_NV                             0x9354
#define GL_VIEWPORT_SWIZZLE_NEGATIVE_Z_NV                             0x9355
#define GL_VIEWPORT_SWIZZLE_POSITIVE_W_NV                             0x9356
#define GL_VIEWPORT_SWIZZLE_NEGATIVE_W_NV                             0x9357
#define GL_VIEWPORT_SWIZZLE_X_NV                                      0x9358
#define GL_VIEWPORT_SWIZZLE_Y_NV                                      0x9359
#define GL_VIEWPORT_SWIZZLE_Z_NV                                      0x935A
#define GL_VIEWPORT_SWIZZLE_W_NV                                      0x935B
#define GL_CLIP_ORIGIN                                                0x935C
#define GL_CLIP_DEPTH_MODE                                            0x935D
#define GL_NEGATIVE_ONE_TO_ONE                                        0x935E
#define GL_ZERO_TO_ONE                                                0x935F
#define GL_CLEAR_TEXTURE                                              0x9365
#define GL_TEXTURE_REDUCTION_MODE_ARB                                 0x9366
#define GL_TEXTURE_REDUCTION_MODE_EXT                                 0x9366
#define GL_WEIGHTED_AVERAGE_ARB                                       0x9367
#define GL_WEIGHTED_AVERAGE_EXT                                       0x9367
#define GL_FONT_GLYPHS_AVAILABLE_NV                                   0x9368
#define GL_FONT_TARGET_UNAVAILABLE_NV                                 0x9369
#define GL_FONT_UNAVAILABLE_NV                                        0x936A
#define GL_FONT_UNINTELLIGIBLE_NV                                     0x936B
#define GL_STANDARD_FONT_FORMAT_NV                                    0x936C
#define GL_FRAGMENT_INPUT_NV                                          0x936D
#define GL_UNIFORM_BUFFER_UNIFIED_NV                                  0x936E
#define GL_UNIFORM_BUFFER_ADDRESS_NV                                  0x936F
#define GL_UNIFORM_BUFFER_LENGTH_NV                                   0x9370
#define GL_MULTISAMPLES_NV                                            0x9371
#define GL_SUPERSAMPLE_SCALE_X_NV                                     0x9372
#define GL_SUPERSAMPLE_SCALE_Y_NV                                     0x9373
#define GL_CONFORMANT_NV                                              0x9374
#define GL_CONSERVATIVE_RASTER_DILATE_NV                              0x9379
#define GL_CONSERVATIVE_RASTER_DILATE_RANGE_NV                        0x937A
#define GL_CONSERVATIVE_RASTER_DILATE_GRANULARITY_NV                  0x937B
#define GL_VIEWPORT_POSITION_W_SCALE_NV                               0x937C
#define GL_VIEWPORT_POSITION_W_SCALE_X_COEFF_NV                       0x937D
#define GL_VIEWPORT_POSITION_W_SCALE_Y_COEFF_NV                       0x937E
#define GL_REPRESENTATIVE_FRAGMENT_TEST_NV                            0x937F
#define GL_NUM_SAMPLE_COUNTS                                          0x9380
#define GL_MULTISAMPLE_LINE_WIDTH_RANGE_ARB                           0x9381
#define GL_MULTISAMPLE_LINE_WIDTH_GRANULARITY_ARB                     0x9382
#define GL_VIEW_CLASS_EAC_R11                                         0x9383
#define GL_VIEW_CLASS_EAC_RG11                                        0x9384
#define GL_VIEW_CLASS_ETC2_RGB                                        0x9385
#define GL_VIEW_CLASS_ETC2_RGBA                                       0x9386
#define GL_VIEW_CLASS_ETC2_EAC_RGBA                                   0x9387
#define GL_VIEW_CLASS_ASTC_4x4_RGBA                                   0x9388
#define GL_VIEW_CLASS_ASTC_5x4_RGBA                                   0x9389
#define GL_VIEW_CLASS_ASTC_5x5_RGBA                                   0x938A
#define GL_VIEW_CLASS_ASTC_6x5_RGBA                                   0x938B
#define GL_VIEW_CLASS_ASTC_6x6_RGBA                                   0x938C
#define GL_VIEW_CLASS_ASTC_8x5_RGBA                                   0x938D
#define GL_VIEW_CLASS_ASTC_8x6_RGBA                                   0x938E
#define GL_VIEW_CLASS_ASTC_8x8_RGBA                                   0x938F
#define GL_VIEW_CLASS_ASTC_10x5_RGBA                                  0x9390
#define GL_VIEW_CLASS_ASTC_10x6_RGBA                                  0x9391
#define GL_VIEW_CLASS_ASTC_10x8_RGBA                                  0x9392
#define GL_VIEW_CLASS_ASTC_10x10_RGBA                                 0x9393
#define GL_VIEW_CLASS_ASTC_12x10_RGBA                                 0x9394
#define GL_VIEW_CLASS_ASTC_12x12_RGBA                                 0x9395
#define GL_COMPRESSED_RGBA_ASTC_4x4_KHR                               0x93B0
#define GL_COMPRESSED_RGBA_ASTC_5x4_KHR                               0x93B1
#define GL_COMPRESSED_RGBA_ASTC_5x5_KHR                               0x93B2
#define GL_COMPRESSED_RGBA_ASTC_6x5_KHR                               0x93B3
#define GL_COMPRESSED_RGBA_ASTC_6x6_KHR                               0x93B4
#define GL_COMPRESSED_RGBA_ASTC_8x5_KHR                               0x93B5
#define GL_COMPRESSED_RGBA_ASTC_8x6_KHR                               0x93B6
#define GL_COMPRESSED_RGBA_ASTC_8x8_KHR                               0x93B7
#define GL_COMPRESSED_RGBA_ASTC_10x5_KHR                              0x93B8
#define GL_COMPRESSED_RGBA_ASTC_10x6_KHR                              0x93B9
#define GL_COMPRESSED_RGBA_ASTC_10x8_KHR                              0x93BA
#define GL_COMPRESSED_RGBA_ASTC_10x10_KHR                             0x93BB
#define GL_COMPRESSED_RGBA_ASTC_12x10_KHR                             0x93BC
#define GL_COMPRESSED_RGBA_ASTC_12x12_KHR                             0x93BD
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR                       0x93D0
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR                       0x93D1
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR                       0x93D2
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR                       0x93D3
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR                       0x93D4
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR                       0x93D5
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR                       0x93D6
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR                       0x93D7
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR                      0x93D8
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR                      0x93D9
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR                      0x93DA
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR                     0x93DB
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR                     0x93DC
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR                     0x93DD
#define GL_PERFQUERY_COUNTER_EVENT_INTEL                              0x94F0
#define GL_PERFQUERY_COUNTER_DURATION_NORM_INTEL                      0x94F1
#define GL_PERFQUERY_COUNTER_DURATION_RAW_INTEL                       0x94F2
#define GL_PERFQUERY_COUNTER_THROUGHPUT_INTEL                         0x94F3
#define GL_PERFQUERY_COUNTER_RAW_INTEL                                0x94F4
#define GL_PERFQUERY_COUNTER_TIMESTAMP_INTEL                          0x94F5
#define GL_PERFQUERY_COUNTER_DATA_UINT32_INTEL                        0x94F8
#define GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL                        0x94F9
#define GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL                         0x94FA
#define GL_PERFQUERY_COUNTER_DATA_DOUBLE_INTEL                        0x94FB
#define GL_PERFQUERY_COUNTER_DATA_BOOL32_INTEL                        0x94FC
#define GL_PERFQUERY_QUERY_NAME_LENGTH_MAX_INTEL                      0x94FD
#define GL_PERFQUERY_COUNTER_NAME_LENGTH_MAX_INTEL                    0x94FE
#define GL_PERFQUERY_COUNTER_DESC_LENGTH_MAX_INTEL                    0x94FF
#define GL_PERFQUERY_GPA_EXTENDED_COUNTERS_INTEL                      0x9500
#define GL_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_EXT              0x9530
#define GL_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_EXT              0x9531
#define GL_SUBGROUP_SIZE_KHR                                          0x9532
#define GL_SUBGROUP_SUPPORTED_STAGES_KHR                              0x9533
#define GL_SUBGROUP_SUPPORTED_FEATURES_KHR                            0x9534
#define GL_SUBGROUP_QUAD_ALL_STAGES_KHR                               0x9535
#define GL_MAX_MESH_TOTAL_MEMORY_SIZE_NV                              0x9536
#define GL_MAX_TASK_TOTAL_MEMORY_SIZE_NV                              0x9537
#define GL_MAX_MESH_OUTPUT_VERTICES_NV                                0x9538
#define GL_MAX_MESH_OUTPUT_PRIMITIVES_NV                              0x9539
#define GL_MAX_TASK_OUTPUT_COUNT_NV                                   0x953A
#define GL_MAX_MESH_WORK_GROUP_SIZE_NV                                0x953B
#define GL_MAX_TASK_WORK_GROUP_SIZE_NV                                0x953C
#define GL_MAX_DRAW_MESH_TASKS_COUNT_NV                               0x953D
#define GL_MESH_WORK_GROUP_SIZE_NV                                    0x953E
#define GL_TASK_WORK_GROUP_SIZE_NV                                    0x953F
#define GL_QUERY_RESOURCE_TYPE_VIDMEM_ALLOC_NV                        0x9540
#define GL_QUERY_RESOURCE_MEMTYPE_VIDMEM_NV                           0x9542
#define GL_MESH_OUTPUT_PER_PRIMITIVE_GRANULARITY_NV                   0x9543
#define GL_QUERY_RESOURCE_SYS_RESERVED_NV                             0x9544
#define GL_QUERY_RESOURCE_TEXTURE_NV                                  0x9545
#define GL_QUERY_RESOURCE_RENDERBUFFER_NV                             0x9546
#define GL_QUERY_RESOURCE_BUFFEROBJECT_NV                             0x9547
#define GL_PER_GPU_STORAGE_NV                                         0x9548
#define GL_MULTICAST_PROGRAMMABLE_SAMPLE_LOCATION_NV                  0x9549
#define GL_CONSERVATIVE_RASTER_MODE_NV                                0x954D
#define GL_CONSERVATIVE_RASTER_MODE_POST_SNAP_NV                      0x954E
#define GL_CONSERVATIVE_RASTER_MODE_PRE_SNAP_TRIANGLES_NV             0x954F
#define GL_CONSERVATIVE_RASTER_MODE_PRE_SNAP_NV                       0x9550
#define GL_SHADER_BINARY_FORMAT_SPIR_V                                0x9551
#define GL_SHADER_BINARY_FORMAT_SPIR_V_ARB                            0x9551
#define GL_SPIR_V_BINARY                                              0x9552
#define GL_SPIR_V_BINARY_ARB                                          0x9552
#define GL_SPIR_V_EXTENSIONS                                          0x9553
#define GL_NUM_SPIR_V_EXTENSIONS                                      0x9554
#define GL_SCISSOR_TEST_EXCLUSIVE_NV                                  0x9555
#define GL_SCISSOR_BOX_EXCLUSIVE_NV                                   0x9556
#define GL_MAX_MESH_VIEWS_NV                                          0x9557
#define GL_RENDER_GPU_MASK_NV                                         0x9558
#define GL_MESH_SHADER_NV                                             0x9559
#define GL_TASK_SHADER_NV                                             0x955A
#define GL_SHADING_RATE_IMAGE_BINDING_NV                              0x955B
#define GL_SHADING_RATE_IMAGE_TEXEL_WIDTH_NV                          0x955C
#define GL_SHADING_RATE_IMAGE_TEXEL_HEIGHT_NV                         0x955D
#define GL_SHADING_RATE_IMAGE_PALETTE_SIZE_NV                         0x955E
#define GL_MAX_COARSE_FRAGMENT_SAMPLES_NV                             0x955F
#define GL_SHADING_RATE_IMAGE_NV                                      0x9563
#define GL_SHADING_RATE_NO_INVOCATIONS_NV                             0x9564
#define GL_SHADING_RATE_1_INVOCATION_PER_PIXEL_NV                     0x9565
#define GL_SHADING_RATE_1_INVOCATION_PER_1X2_PIXELS_NV                0x9566
#define GL_SHADING_RATE_1_INVOCATION_PER_2X1_PIXELS_NV                0x9567
#define GL_SHADING_RATE_1_INVOCATION_PER_2X2_PIXELS_NV                0x9568
#define GL_SHADING_RATE_1_INVOCATION_PER_2X4_PIXELS_NV                0x9569
#define GL_SHADING_RATE_1_INVOCATION_PER_4X2_PIXELS_NV                0x956A
#define GL_SHADING_RATE_1_INVOCATION_PER_4X4_PIXELS_NV                0x956B
#define GL_SHADING_RATE_2_INVOCATIONS_PER_PIXEL_NV                    0x956C
#define GL_SHADING_RATE_4_INVOCATIONS_PER_PIXEL_NV                    0x956D
#define GL_SHADING_RATE_8_INVOCATIONS_PER_PIXEL_NV                    0x956E
#define GL_SHADING_RATE_16_INVOCATIONS_PER_PIXEL_NV                   0x956F
#define GL_MESH_VERTICES_OUT_NV                                       0x9579
#define GL_MESH_PRIMITIVES_OUT_NV                                     0x957A
#define GL_MESH_OUTPUT_TYPE_NV                                        0x957B
#define GL_MESH_SUBROUTINE_NV                                         0x957C
#define GL_TASK_SUBROUTINE_NV                                         0x957D
#define GL_MESH_SUBROUTINE_UNIFORM_NV                                 0x957E
#define GL_TASK_SUBROUTINE_UNIFORM_NV                                 0x957F
#define GL_TEXTURE_TILING_EXT                                         0x9580
#define GL_DEDICATED_MEMORY_OBJECT_EXT                                0x9581
#define GL_NUM_TILING_TYPES_EXT                                       0x9582
#define GL_TILING_TYPES_EXT                                           0x9583
#define GL_OPTIMAL_TILING_EXT                                         0x9584
#define GL_LINEAR_TILING_EXT                                          0x9585
#define GL_HANDLE_TYPE_OPAQUE_FD_EXT                                  0x9586
#define GL_HANDLE_TYPE_OPAQUE_WIN32_EXT                               0x9587
#define GL_HANDLE_TYPE_OPAQUE_WIN32_KMT_EXT                           0x9588
#define GL_HANDLE_TYPE_D3D12_TILEPOOL_EXT                             0x9589
#define GL_HANDLE_TYPE_D3D12_RESOURCE_EXT                             0x958A
#define GL_HANDLE_TYPE_D3D11_IMAGE_EXT                                0x958B
#define GL_HANDLE_TYPE_D3D11_IMAGE_KMT_EXT                            0x958C
#define GL_LAYOUT_GENERAL_EXT                                         0x958D
#define GL_LAYOUT_COLOR_ATTACHMENT_EXT                                0x958E
#define GL_LAYOUT_DEPTH_STENCIL_ATTACHMENT_EXT                        0x958F
#define GL_LAYOUT_DEPTH_STENCIL_READ_ONLY_EXT                         0x9590
#define GL_LAYOUT_SHADER_READ_ONLY_EXT                                0x9591
#define GL_LAYOUT_TRANSFER_SRC_EXT                                    0x9592
#define GL_LAYOUT_TRANSFER_DST_EXT                                    0x9593
#define GL_HANDLE_TYPE_D3D12_FENCE_EXT                                0x9594
#define GL_D3D12_FENCE_VALUE_EXT                                      0x9595
#define GL_NUM_DEVICE_UUIDS_EXT                                       0x9596
#define GL_DEVICE_UUID_EXT                                            0x9597
#define GL_DRIVER_UUID_EXT                                            0x9598
#define GL_DEVICE_LUID_EXT                                            0x9599
#define GL_DEVICE_NODE_MASK_EXT                                       0x959A
#define GL_PROTECTED_MEMORY_OBJECT_EXT                                0x959B
#define GL_UNIFORM_BLOCK_REFERENCED_BY_MESH_SHADER_NV                 0x959C
#define GL_UNIFORM_BLOCK_REFERENCED_BY_TASK_SHADER_NV                 0x959D
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_MESH_SHADER_NV         0x959E
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TASK_SHADER_NV         0x959F
#define GL_REFERENCED_BY_MESH_SHADER_NV                               0x95A0
#define GL_REFERENCED_BY_TASK_SHADER_NV                               0x95A1
#define GL_MAX_MESH_WORK_GROUP_INVOCATIONS_NV                         0x95A2
#define GL_MAX_TASK_WORK_GROUP_INVOCATIONS_NV                         0x95A3
#define GL_ATTACHED_MEMORY_OBJECT_NV                                  0x95A4
#define GL_ATTACHED_MEMORY_OFFSET_NV                                  0x95A5
#define GL_MEMORY_ATTACHABLE_ALIGNMENT_NV                             0x95A6
#define GL_MEMORY_ATTACHABLE_SIZE_NV                                  0x95A7
#define GL_MEMORY_ATTACHABLE_NV                                       0x95A8
#define GL_DETACHED_MEMORY_INCARNATION_NV                             0x95A9
#define GL_DETACHED_TEXTURES_NV                                       0x95AA
#define GL_DETACHED_BUFFERS_NV                                        0x95AB
#define GL_MAX_DETACHED_TEXTURES_NV                                   0x95AC
#define GL_MAX_DETACHED_BUFFERS_NV                                    0x95AD
#define GL_SHADING_RATE_SAMPLE_ORDER_DEFAULT_NV                       0x95AE
#define GL_SHADING_RATE_SAMPLE_ORDER_PIXEL_MAJOR_NV                   0x95AF
#define GL_SHADING_RATE_SAMPLE_ORDER_SAMPLE_MAJOR_NV                  0x95B0


namespace OpenSubdiv {
namespace internal {
namespace GLApi {


typedef void (GLAPIENTRY *PFNGLACCUMPROC) (GLenum  op, GLfloat  value);
typedef void (GLAPIENTRY *PFNGLACTIVEPROGRAMEXTPROC) (GLuint  program);
typedef void (GLAPIENTRY *PFNGLACTIVESHADERPROGRAMPROC) (GLuint  pipeline, GLuint  program);
typedef void (GLAPIENTRY *PFNGLACTIVESHADERPROGRAMEXTPROC) (GLuint  pipeline, GLuint  program);
typedef void (GLAPIENTRY *PFNGLACTIVESTENCILFACEEXTPROC) (GLenum  face);
typedef void (GLAPIENTRY *PFNGLACTIVETEXTUREPROC) (GLenum  texture);
typedef void (GLAPIENTRY *PFNGLACTIVETEXTUREARBPROC) (GLenum  texture);
typedef void (GLAPIENTRY *PFNGLACTIVEVARYINGNVPROC) (GLuint  program, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLALPHAFUNCPROC) (GLenum  func, GLfloat  ref);
typedef void (GLAPIENTRY *PFNGLALPHATOCOVERAGEDITHERCONTROLNVPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLAPPLYFRAMEBUFFERATTACHMENTCMAAINTELPROC) ();
typedef void (GLAPIENTRY *PFNGLAPPLYTEXTUREEXTPROC) (GLenum  mode);
typedef GLboolean (GLAPIENTRY *PFNGLACQUIREKEYEDMUTEXWIN32EXTPROC) (GLuint  memory, GLuint64  key, GLuint  timeout);
typedef GLboolean (GLAPIENTRY *PFNGLAREPROGRAMSRESIDENTNVPROC) (GLsizei  n, const GLuint * programs, GLboolean * residences);
typedef GLboolean (GLAPIENTRY *PFNGLARETEXTURESRESIDENTPROC) (GLsizei  n, const GLuint * textures, GLboolean * residences);
typedef GLboolean (GLAPIENTRY *PFNGLARETEXTURESRESIDENTEXTPROC) (GLsizei  n, const GLuint * textures, GLboolean * residences);
typedef void (GLAPIENTRY *PFNGLARRAYELEMENTPROC) (GLint  i);
typedef void (GLAPIENTRY *PFNGLARRAYELEMENTEXTPROC) (GLint  i);
typedef void (GLAPIENTRY *PFNGLATTACHOBJECTARBPROC) (GLhandleARB  containerObj, GLhandleARB  obj);
typedef void (GLAPIENTRY *PFNGLATTACHSHADERPROC) (GLuint  program, GLuint  shader);
typedef void (GLAPIENTRY *PFNGLBEGINPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLBEGINCONDITIONALRENDERPROC) (GLuint  id, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLBEGINCONDITIONALRENDERNVPROC) (GLuint  id, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLBEGINOCCLUSIONQUERYNVPROC) (GLuint  id);
typedef void (GLAPIENTRY *PFNGLBEGINPERFMONITORAMDPROC) (GLuint  monitor);
typedef void (GLAPIENTRY *PFNGLBEGINPERFQUERYINTELPROC) (GLuint  queryHandle);
typedef void (GLAPIENTRY *PFNGLBEGINQUERYPROC) (GLenum  target, GLuint  id);
typedef void (GLAPIENTRY *PFNGLBEGINQUERYARBPROC) (GLenum  target, GLuint  id);
typedef void (GLAPIENTRY *PFNGLBEGINQUERYINDEXEDPROC) (GLenum  target, GLuint  index, GLuint  id);
typedef void (GLAPIENTRY *PFNGLBEGINTRANSFORMFEEDBACKPROC) (GLenum  primitiveMode);
typedef void (GLAPIENTRY *PFNGLBEGINTRANSFORMFEEDBACKEXTPROC) (GLenum  primitiveMode);
typedef void (GLAPIENTRY *PFNGLBEGINTRANSFORMFEEDBACKNVPROC) (GLenum  primitiveMode);
typedef void (GLAPIENTRY *PFNGLBEGINVERTEXSHADEREXTPROC) ();
typedef void (GLAPIENTRY *PFNGLBEGINVIDEOCAPTURENVPROC) (GLuint  video_capture_slot);
typedef void (GLAPIENTRY *PFNGLBINDATTRIBLOCATIONPROC) (GLuint  program, GLuint  index, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLBINDATTRIBLOCATIONARBPROC) (GLhandleARB  programObj, GLuint  index, const GLcharARB * name);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERPROC) (GLenum  target, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERARBPROC) (GLenum  target, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERBASEPROC) (GLenum  target, GLuint  index, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERBASEEXTPROC) (GLenum  target, GLuint  index, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERBASENVPROC) (GLenum  target, GLuint  index, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLBINDBUFFEROFFSETEXTPROC) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLBINDBUFFEROFFSETNVPROC) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERRANGEPROC) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERRANGEEXTPROC) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERRANGENVPROC) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERSBASEPROC) (GLenum  target, GLuint  first, GLsizei  count, const GLuint * buffers);
typedef void (GLAPIENTRY *PFNGLBINDBUFFERSRANGEPROC) (GLenum  target, GLuint  first, GLsizei  count, const GLuint * buffers, const GLintptr * offsets, const GLsizeiptr * sizes);
typedef void (GLAPIENTRY *PFNGLBINDFRAGDATALOCATIONPROC) (GLuint  program, GLuint  color, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLBINDFRAGDATALOCATIONEXTPROC) (GLuint  program, GLuint  color, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLBINDFRAGDATALOCATIONINDEXEDPROC) (GLuint  program, GLuint  colorNumber, GLuint  index, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLBINDFRAMEBUFFERPROC) (GLenum  target, GLuint  framebuffer);
typedef void (GLAPIENTRY *PFNGLBINDFRAMEBUFFEREXTPROC) (GLenum  target, GLuint  framebuffer);
typedef void (GLAPIENTRY *PFNGLBINDIMAGETEXTUREPROC) (GLuint  unit, GLuint  texture, GLint  level, GLboolean  layered, GLint  layer, GLenum  access, GLenum  format);
typedef void (GLAPIENTRY *PFNGLBINDIMAGETEXTUREEXTPROC) (GLuint  index, GLuint  texture, GLint  level, GLboolean  layered, GLint  layer, GLenum  access, GLint  format);
typedef void (GLAPIENTRY *PFNGLBINDIMAGETEXTURESPROC) (GLuint  first, GLsizei  count, const GLuint * textures);
typedef GLuint (GLAPIENTRY *PFNGLBINDLIGHTPARAMETEREXTPROC) (GLenum  light, GLenum  value);
typedef GLuint (GLAPIENTRY *PFNGLBINDMATERIALPARAMETEREXTPROC) (GLenum  face, GLenum  value);
typedef void (GLAPIENTRY *PFNGLBINDMULTITEXTUREEXTPROC) (GLenum  texunit, GLenum  target, GLuint  texture);
typedef GLuint (GLAPIENTRY *PFNGLBINDPARAMETEREXTPROC) (GLenum  value);
typedef void (GLAPIENTRY *PFNGLBINDPROGRAMARBPROC) (GLenum  target, GLuint  program);
typedef void (GLAPIENTRY *PFNGLBINDPROGRAMNVPROC) (GLenum  target, GLuint  id);
typedef void (GLAPIENTRY *PFNGLBINDPROGRAMPIPELINEPROC) (GLuint  pipeline);
typedef void (GLAPIENTRY *PFNGLBINDPROGRAMPIPELINEEXTPROC) (GLuint  pipeline);
typedef void (GLAPIENTRY *PFNGLBINDRENDERBUFFERPROC) (GLenum  target, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLBINDRENDERBUFFEREXTPROC) (GLenum  target, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLBINDSAMPLERPROC) (GLuint  unit, GLuint  sampler);
typedef void (GLAPIENTRY *PFNGLBINDSAMPLERSPROC) (GLuint  first, GLsizei  count, const GLuint * samplers);
typedef void (GLAPIENTRY *PFNGLBINDSHADINGRATEIMAGENVPROC) (GLuint  texture);
typedef GLuint (GLAPIENTRY *PFNGLBINDTEXGENPARAMETEREXTPROC) (GLenum  unit, GLenum  coord, GLenum  value);
typedef void (GLAPIENTRY *PFNGLBINDTEXTUREPROC) (GLenum  target, GLuint  texture);
typedef void (GLAPIENTRY *PFNGLBINDTEXTUREEXTPROC) (GLenum  target, GLuint  texture);
typedef void (GLAPIENTRY *PFNGLBINDTEXTUREUNITPROC) (GLuint  unit, GLuint  texture);
typedef GLuint (GLAPIENTRY *PFNGLBINDTEXTUREUNITPARAMETEREXTPROC) (GLenum  unit, GLenum  value);
typedef void (GLAPIENTRY *PFNGLBINDTEXTURESPROC) (GLuint  first, GLsizei  count, const GLuint * textures);
typedef void (GLAPIENTRY *PFNGLBINDTRANSFORMFEEDBACKPROC) (GLenum  target, GLuint  id);
typedef void (GLAPIENTRY *PFNGLBINDTRANSFORMFEEDBACKNVPROC) (GLenum  target, GLuint  id);
typedef void (GLAPIENTRY *PFNGLBINDVERTEXARRAYPROC) (GLuint  array);
typedef void (GLAPIENTRY *PFNGLBINDVERTEXARRAYAPPLEPROC) (GLuint  array);
typedef void (GLAPIENTRY *PFNGLBINDVERTEXBUFFERPROC) (GLuint  bindingindex, GLuint  buffer, GLintptr  offset, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLBINDVERTEXBUFFERSPROC) (GLuint  first, GLsizei  count, const GLuint * buffers, const GLintptr * offsets, const GLsizei * strides);
typedef void (GLAPIENTRY *PFNGLBINDVERTEXSHADEREXTPROC) (GLuint  id);
typedef void (GLAPIENTRY *PFNGLBINDVIDEOCAPTURESTREAMBUFFERNVPROC) (GLuint  video_capture_slot, GLuint  stream, GLenum  frame_region, GLintptrARB  offset);
typedef void (GLAPIENTRY *PFNGLBINDVIDEOCAPTURESTREAMTEXTURENVPROC) (GLuint  video_capture_slot, GLuint  stream, GLenum  frame_region, GLenum  target, GLuint  texture);
typedef void (GLAPIENTRY *PFNGLBINORMAL3BEXTPROC) (GLbyte  bx, GLbyte  by, GLbyte  bz);
typedef void (GLAPIENTRY *PFNGLBINORMAL3BVEXTPROC) (const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLBINORMAL3DEXTPROC) (GLdouble  bx, GLdouble  by, GLdouble  bz);
typedef void (GLAPIENTRY *PFNGLBINORMAL3DVEXTPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLBINORMAL3FEXTPROC) (GLfloat  bx, GLfloat  by, GLfloat  bz);
typedef void (GLAPIENTRY *PFNGLBINORMAL3FVEXTPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLBINORMAL3IEXTPROC) (GLint  bx, GLint  by, GLint  bz);
typedef void (GLAPIENTRY *PFNGLBINORMAL3IVEXTPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLBINORMAL3SEXTPROC) (GLshort  bx, GLshort  by, GLshort  bz);
typedef void (GLAPIENTRY *PFNGLBINORMAL3SVEXTPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLBINORMALPOINTEREXTPROC) (GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLBITMAPPROC) (GLsizei  width, GLsizei  height, GLfloat  xorig, GLfloat  yorig, GLfloat  xmove, GLfloat  ymove, const GLubyte * bitmap);
typedef void (GLAPIENTRY *PFNGLBLENDBARRIERKHRPROC) ();
typedef void (GLAPIENTRY *PFNGLBLENDBARRIERNVPROC) ();
typedef void (GLAPIENTRY *PFNGLBLENDCOLORPROC) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
typedef void (GLAPIENTRY *PFNGLBLENDCOLOREXTPROC) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONEXTPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONINDEXEDAMDPROC) (GLuint  buf, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONSEPARATEPROC) (GLenum  modeRGB, GLenum  modeAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONSEPARATEEXTPROC) (GLenum  modeRGB, GLenum  modeAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONSEPARATEINDEXEDAMDPROC) (GLuint  buf, GLenum  modeRGB, GLenum  modeAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONSEPARATEIPROC) (GLuint  buf, GLenum  modeRGB, GLenum  modeAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONSEPARATEIARBPROC) (GLuint  buf, GLenum  modeRGB, GLenum  modeAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONIPROC) (GLuint  buf, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLBLENDEQUATIONIARBPROC) (GLuint  buf, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCPROC) (GLenum  sfactor, GLenum  dfactor);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCINDEXEDAMDPROC) (GLuint  buf, GLenum  src, GLenum  dst);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCSEPARATEPROC) (GLenum  sfactorRGB, GLenum  dfactorRGB, GLenum  sfactorAlpha, GLenum  dfactorAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCSEPARATEEXTPROC) (GLenum  sfactorRGB, GLenum  dfactorRGB, GLenum  sfactorAlpha, GLenum  dfactorAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCSEPARATEINDEXEDAMDPROC) (GLuint  buf, GLenum  srcRGB, GLenum  dstRGB, GLenum  srcAlpha, GLenum  dstAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCSEPARATEIPROC) (GLuint  buf, GLenum  srcRGB, GLenum  dstRGB, GLenum  srcAlpha, GLenum  dstAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCSEPARATEIARBPROC) (GLuint  buf, GLenum  srcRGB, GLenum  dstRGB, GLenum  srcAlpha, GLenum  dstAlpha);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCIPROC) (GLuint  buf, GLenum  src, GLenum  dst);
typedef void (GLAPIENTRY *PFNGLBLENDFUNCIARBPROC) (GLuint  buf, GLenum  src, GLenum  dst);
typedef void (GLAPIENTRY *PFNGLBLENDPARAMETERINVPROC) (GLenum  pname, GLint  value);
typedef void (GLAPIENTRY *PFNGLBLITFRAMEBUFFERPROC) (GLint  srcX0, GLint  srcY0, GLint  srcX1, GLint  srcY1, GLint  dstX0, GLint  dstY0, GLint  dstX1, GLint  dstY1, GLbitfield  mask, GLenum  filter);
typedef void (GLAPIENTRY *PFNGLBLITFRAMEBUFFEREXTPROC) (GLint  srcX0, GLint  srcY0, GLint  srcX1, GLint  srcY1, GLint  dstX0, GLint  dstY0, GLint  dstX1, GLint  dstY1, GLbitfield  mask, GLenum  filter);
typedef void (GLAPIENTRY *PFNGLBLITNAMEDFRAMEBUFFERPROC) (GLuint  readFramebuffer, GLuint  drawFramebuffer, GLint  srcX0, GLint  srcY0, GLint  srcX1, GLint  srcY1, GLint  dstX0, GLint  dstY0, GLint  dstX1, GLint  dstY1, GLbitfield  mask, GLenum  filter);
typedef void (GLAPIENTRY *PFNGLBUFFERADDRESSRANGENVPROC) (GLenum  pname, GLuint  index, GLuint64EXT  address, GLsizeiptr  length);
typedef void (GLAPIENTRY *PFNGLBUFFERATTACHMEMORYNVPROC) (GLenum  target, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLBUFFERDATAPROC) (GLenum  target, GLsizeiptr  size, const void * data, GLenum  usage);
typedef void (GLAPIENTRY *PFNGLBUFFERDATAARBPROC) (GLenum  target, GLsizeiptrARB  size, const void * data, GLenum  usage);
typedef void (GLAPIENTRY *PFNGLBUFFERPAGECOMMITMENTARBPROC) (GLenum  target, GLintptr  offset, GLsizeiptr  size, GLboolean  commit);
typedef void (GLAPIENTRY *PFNGLBUFFERPARAMETERIAPPLEPROC) (GLenum  target, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLBUFFERSTORAGEPROC) (GLenum  target, GLsizeiptr  size, const void * data, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLBUFFERSTORAGEEXTERNALEXTPROC) (GLenum  target, GLintptr  offset, GLsizeiptr  size, GLeglClientBufferEXT  clientBuffer, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLBUFFERSTORAGEMEMEXTPROC) (GLenum  target, GLsizeiptr  size, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLBUFFERSUBDATAPROC) (GLenum  target, GLintptr  offset, GLsizeiptr  size, const void * data);
typedef void (GLAPIENTRY *PFNGLBUFFERSUBDATAARBPROC) (GLenum  target, GLintptrARB  offset, GLsizeiptrARB  size, const void * data);
typedef void (GLAPIENTRY *PFNGLCALLCOMMANDLISTNVPROC) (GLuint  list);
typedef void (GLAPIENTRY *PFNGLCALLLISTPROC) (GLuint  list);
typedef void (GLAPIENTRY *PFNGLCALLLISTSPROC) (GLsizei  n, GLenum  type, const void * lists);
typedef GLenum (GLAPIENTRY *PFNGLCHECKFRAMEBUFFERSTATUSPROC) (GLenum  target);
typedef GLenum (GLAPIENTRY *PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC) (GLenum  target);
typedef GLenum (GLAPIENTRY *PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC) (GLuint  framebuffer, GLenum  target);
typedef GLenum (GLAPIENTRY *PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC) (GLuint  framebuffer, GLenum  target);
typedef void (GLAPIENTRY *PFNGLCLAMPCOLORPROC) (GLenum  target, GLenum  clamp);
typedef void (GLAPIENTRY *PFNGLCLAMPCOLORARBPROC) (GLenum  target, GLenum  clamp);
typedef void (GLAPIENTRY *PFNGLCLEARPROC) (GLbitfield  mask);
typedef void (GLAPIENTRY *PFNGLCLEARACCUMPROC) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
typedef void (GLAPIENTRY *PFNGLCLEARBUFFERDATAPROC) (GLenum  target, GLenum  internalformat, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCLEARBUFFERSUBDATAPROC) (GLenum  target, GLenum  internalformat, GLintptr  offset, GLsizeiptr  size, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCLEARBUFFERFIPROC) (GLenum  buffer, GLint  drawbuffer, GLfloat  depth, GLint  stencil);
typedef void (GLAPIENTRY *PFNGLCLEARBUFFERFVPROC) (GLenum  buffer, GLint  drawbuffer, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLCLEARBUFFERIVPROC) (GLenum  buffer, GLint  drawbuffer, const GLint * value);
typedef void (GLAPIENTRY *PFNGLCLEARBUFFERUIVPROC) (GLenum  buffer, GLint  drawbuffer, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLCLEARCOLORPROC) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
typedef void (GLAPIENTRY *PFNGLCLEARCOLORIIEXTPROC) (GLint  red, GLint  green, GLint  blue, GLint  alpha);
typedef void (GLAPIENTRY *PFNGLCLEARCOLORIUIEXTPROC) (GLuint  red, GLuint  green, GLuint  blue, GLuint  alpha);
typedef void (GLAPIENTRY *PFNGLCLEARDEPTHPROC) (GLdouble  depth);
typedef void (GLAPIENTRY *PFNGLCLEARDEPTHDNVPROC) (GLdouble  depth);
typedef void (GLAPIENTRY *PFNGLCLEARDEPTHFPROC) (GLfloat  d);
typedef void (GLAPIENTRY *PFNGLCLEARINDEXPROC) (GLfloat  c);
typedef void (GLAPIENTRY *PFNGLCLEARNAMEDBUFFERDATAPROC) (GLuint  buffer, GLenum  internalformat, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCLEARNAMEDBUFFERDATAEXTPROC) (GLuint  buffer, GLenum  internalformat, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCLEARNAMEDBUFFERSUBDATAPROC) (GLuint  buffer, GLenum  internalformat, GLintptr  offset, GLsizeiptr  size, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC) (GLuint  buffer, GLenum  internalformat, GLsizeiptr  offset, GLsizeiptr  size, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCLEARNAMEDFRAMEBUFFERFIPROC) (GLuint  framebuffer, GLenum  buffer, GLint  drawbuffer, GLfloat  depth, GLint  stencil);
typedef void (GLAPIENTRY *PFNGLCLEARNAMEDFRAMEBUFFERFVPROC) (GLuint  framebuffer, GLenum  buffer, GLint  drawbuffer, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLCLEARNAMEDFRAMEBUFFERIVPROC) (GLuint  framebuffer, GLenum  buffer, GLint  drawbuffer, const GLint * value);
typedef void (GLAPIENTRY *PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC) (GLuint  framebuffer, GLenum  buffer, GLint  drawbuffer, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLCLEARSTENCILPROC) (GLint  s);
typedef void (GLAPIENTRY *PFNGLCLEARTEXIMAGEPROC) (GLuint  texture, GLint  level, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCLEARTEXSUBIMAGEPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCLIENTACTIVETEXTUREPROC) (GLenum  texture);
typedef void (GLAPIENTRY *PFNGLCLIENTACTIVETEXTUREARBPROC) (GLenum  texture);
typedef void (GLAPIENTRY *PFNGLCLIENTATTRIBDEFAULTEXTPROC) (GLbitfield  mask);
typedef GLenum (GLAPIENTRY *PFNGLCLIENTWAITSYNCPROC) (GLsync  sync, GLbitfield  flags, GLuint64  timeout);
typedef void (GLAPIENTRY *PFNGLCLIPCONTROLPROC) (GLenum  origin, GLenum  depth);
typedef void (GLAPIENTRY *PFNGLCLIPPLANEPROC) (GLenum  plane, const GLdouble * equation);
typedef void (GLAPIENTRY *PFNGLCOLOR3BPROC) (GLbyte  red, GLbyte  green, GLbyte  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3BVPROC) (const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLCOLOR3DPROC) (GLdouble  red, GLdouble  green, GLdouble  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLCOLOR3FPROC) (GLfloat  red, GLfloat  green, GLfloat  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLCOLOR3HNVPROC) (GLhalfNV  red, GLhalfNV  green, GLhalfNV  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLCOLOR3IPROC) (GLint  red, GLint  green, GLint  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLCOLOR3SPROC) (GLshort  red, GLshort  green, GLshort  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLCOLOR3UBPROC) (GLubyte  red, GLubyte  green, GLubyte  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3UBVPROC) (const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLCOLOR3UIPROC) (GLuint  red, GLuint  green, GLuint  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3UIVPROC) (const GLuint * v);
typedef void (GLAPIENTRY *PFNGLCOLOR3USPROC) (GLushort  red, GLushort  green, GLushort  blue);
typedef void (GLAPIENTRY *PFNGLCOLOR3USVPROC) (const GLushort * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4BPROC) (GLbyte  red, GLbyte  green, GLbyte  blue, GLbyte  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4BVPROC) (const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4DPROC) (GLdouble  red, GLdouble  green, GLdouble  blue, GLdouble  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4FPROC) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4HNVPROC) (GLhalfNV  red, GLhalfNV  green, GLhalfNV  blue, GLhalfNV  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4IPROC) (GLint  red, GLint  green, GLint  blue, GLint  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4SPROC) (GLshort  red, GLshort  green, GLshort  blue, GLshort  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4UBPROC) (GLubyte  red, GLubyte  green, GLubyte  blue, GLubyte  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4UBVPROC) (const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4UIPROC) (GLuint  red, GLuint  green, GLuint  blue, GLuint  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4UIVPROC) (const GLuint * v);
typedef void (GLAPIENTRY *PFNGLCOLOR4USPROC) (GLushort  red, GLushort  green, GLushort  blue, GLushort  alpha);
typedef void (GLAPIENTRY *PFNGLCOLOR4USVPROC) (const GLushort * v);
typedef void (GLAPIENTRY *PFNGLCOLORFORMATNVPROC) (GLint  size, GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLCOLORMASKPROC) (GLboolean  red, GLboolean  green, GLboolean  blue, GLboolean  alpha);
typedef void (GLAPIENTRY *PFNGLCOLORMASKINDEXEDEXTPROC) (GLuint  index, GLboolean  r, GLboolean  g, GLboolean  b, GLboolean  a);
typedef void (GLAPIENTRY *PFNGLCOLORMASKIPROC) (GLuint  index, GLboolean  r, GLboolean  g, GLboolean  b, GLboolean  a);
typedef void (GLAPIENTRY *PFNGLCOLORMATERIALPROC) (GLenum  face, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLCOLORP3UIPROC) (GLenum  type, GLuint  color);
typedef void (GLAPIENTRY *PFNGLCOLORP3UIVPROC) (GLenum  type, const GLuint * color);
typedef void (GLAPIENTRY *PFNGLCOLORP4UIPROC) (GLenum  type, GLuint  color);
typedef void (GLAPIENTRY *PFNGLCOLORP4UIVPROC) (GLenum  type, const GLuint * color);
typedef void (GLAPIENTRY *PFNGLCOLORPOINTERPROC) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLCOLORPOINTEREXTPROC) (GLint  size, GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
typedef void (GLAPIENTRY *PFNGLCOLORPOINTERVINTELPROC) (GLint  size, GLenum  type, const void ** pointer);
typedef void (GLAPIENTRY *PFNGLCOLORSUBTABLEPROC) (GLenum  target, GLsizei  start, GLsizei  count, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCOLORSUBTABLEEXTPROC) (GLenum  target, GLsizei  start, GLsizei  count, GLenum  format, GLenum  type, const void * data);
typedef void (GLAPIENTRY *PFNGLCOLORTABLEPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLenum  format, GLenum  type, const void * table);
typedef void (GLAPIENTRY *PFNGLCOLORTABLEEXTPROC) (GLenum  target, GLenum  internalFormat, GLsizei  width, GLenum  format, GLenum  type, const void * table);
typedef void (GLAPIENTRY *PFNGLCOLORTABLEPARAMETERFVPROC) (GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLCOLORTABLEPARAMETERIVPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLCOMBINERINPUTNVPROC) (GLenum  stage, GLenum  portion, GLenum  variable, GLenum  input, GLenum  mapping, GLenum  componentUsage);
typedef void (GLAPIENTRY *PFNGLCOMBINEROUTPUTNVPROC) (GLenum  stage, GLenum  portion, GLenum  abOutput, GLenum  cdOutput, GLenum  sumOutput, GLenum  scale, GLenum  bias, GLboolean  abDotProduct, GLboolean  cdDotProduct, GLboolean  muxSum);
typedef void (GLAPIENTRY *PFNGLCOMBINERPARAMETERFNVPROC) (GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLCOMBINERPARAMETERFVNVPROC) (GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLCOMBINERPARAMETERINVPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLCOMBINERPARAMETERIVNVPROC) (GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLCOMBINERSTAGEPARAMETERFVNVPROC) (GLenum  stage, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLCOMMANDLISTSEGMENTSNVPROC) (GLuint  list, GLuint  segments);
typedef void (GLAPIENTRY *PFNGLCOMPILECOMMANDLISTNVPROC) (GLuint  list);
typedef void (GLAPIENTRY *PFNGLCOMPILESHADERPROC) (GLuint  shader);
typedef void (GLAPIENTRY *PFNGLCOMPILESHADERARBPROC) (GLhandleARB  shaderObj);
typedef void (GLAPIENTRY *PFNGLCOMPILESHADERINCLUDEARBPROC) (GLuint  shader, GLsizei  count, const GLchar *const* path, const GLint * length);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDMULTITEXIMAGE1DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLint  border, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDMULTITEXIMAGE2DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDMULTITEXIMAGE3DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDMULTITEXSUBIMAGE1DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDMULTITEXSUBIMAGE2DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDMULTITEXSUBIMAGE3DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXIMAGE1DPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLint  border, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXIMAGE1DARBPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLint  border, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXIMAGE2DPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXIMAGE2DARBPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXIMAGE3DPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXIMAGE3DARBPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC) (GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTUREIMAGE1DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLint  border, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTUREIMAGE2DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTUREIMAGE3DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * data);
typedef void (GLAPIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * bits);
typedef void (GLAPIENTRY *PFNGLCONSERVATIVERASTERPARAMETERFNVPROC) (GLenum  pname, GLfloat  value);
typedef void (GLAPIENTRY *PFNGLCONSERVATIVERASTERPARAMETERINVPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONFILTER1DPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLenum  format, GLenum  type, const void * image);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONFILTER1DEXTPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLenum  format, GLenum  type, const void * image);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONFILTER2DPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * image);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONFILTER2DEXTPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * image);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONPARAMETERFPROC) (GLenum  target, GLenum  pname, GLfloat  params);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONPARAMETERFEXTPROC) (GLenum  target, GLenum  pname, GLfloat  params);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONPARAMETERFVPROC) (GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONPARAMETERFVEXTPROC) (GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONPARAMETERIPROC) (GLenum  target, GLenum  pname, GLint  params);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONPARAMETERIEXTPROC) (GLenum  target, GLenum  pname, GLint  params);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONPARAMETERIVPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLCONVOLUTIONPARAMETERIVEXTPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLCOPYBUFFERSUBDATAPROC) (GLenum  readTarget, GLenum  writeTarget, GLintptr  readOffset, GLintptr  writeOffset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLCOPYCOLORSUBTABLEPROC) (GLenum  target, GLsizei  start, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYCOLORSUBTABLEEXTPROC) (GLenum  target, GLsizei  start, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYCOLORTABLEPROC) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYCONVOLUTIONFILTER1DPROC) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYCONVOLUTIONFILTER1DEXTPROC) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYCONVOLUTIONFILTER2DPROC) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYCONVOLUTIONFILTER2DEXTPROC) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYIMAGESUBDATAPROC) (GLuint  srcName, GLenum  srcTarget, GLint  srcLevel, GLint  srcX, GLint  srcY, GLint  srcZ, GLuint  dstName, GLenum  dstTarget, GLint  dstLevel, GLint  dstX, GLint  dstY, GLint  dstZ, GLsizei  srcWidth, GLsizei  srcHeight, GLsizei  srcDepth);
typedef void (GLAPIENTRY *PFNGLCOPYIMAGESUBDATANVPROC) (GLuint  srcName, GLenum  srcTarget, GLint  srcLevel, GLint  srcX, GLint  srcY, GLint  srcZ, GLuint  dstName, GLenum  dstTarget, GLint  dstLevel, GLint  dstX, GLint  dstY, GLint  dstZ, GLsizei  width, GLsizei  height, GLsizei  depth);
typedef void (GLAPIENTRY *PFNGLCOPYMULTITEXIMAGE1DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLint  border);
typedef void (GLAPIENTRY *PFNGLCOPYMULTITEXIMAGE2DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLint  border);
typedef void (GLAPIENTRY *PFNGLCOPYMULTITEXSUBIMAGE1DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYMULTITEXSUBIMAGE2DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYMULTITEXSUBIMAGE3DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYNAMEDBUFFERSUBDATAPROC) (GLuint  readBuffer, GLuint  writeBuffer, GLintptr  readOffset, GLintptr  writeOffset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLCOPYPATHNVPROC) (GLuint  resultPath, GLuint  srcPath);
typedef void (GLAPIENTRY *PFNGLCOPYPIXELSPROC) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  type);
typedef void (GLAPIENTRY *PFNGLCOPYTEXIMAGE1DPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLint  border);
typedef void (GLAPIENTRY *PFNGLCOPYTEXIMAGE1DEXTPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLint  border);
typedef void (GLAPIENTRY *PFNGLCOPYTEXIMAGE2DPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLint  border);
typedef void (GLAPIENTRY *PFNGLCOPYTEXIMAGE2DEXTPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLint  border);
typedef void (GLAPIENTRY *PFNGLCOPYTEXSUBIMAGE1DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYTEXSUBIMAGE1DEXTPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYTEXSUBIMAGE2DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYTEXSUBIMAGE2DEXTPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYTEXSUBIMAGE3DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYTEXSUBIMAGE3DEXTPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYTEXTUREIMAGE1DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLint  border);
typedef void (GLAPIENTRY *PFNGLCOPYTEXTUREIMAGE2DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLint  border);
typedef void (GLAPIENTRY *PFNGLCOPYTEXTURESUBIMAGE1DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYTEXTURESUBIMAGE1DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLCOPYTEXTURESUBIMAGE2DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYTEXTURESUBIMAGE2DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYTEXTURESUBIMAGE3DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOPYTEXTURESUBIMAGE3DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLCOVERFILLPATHINSTANCEDNVPROC) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLenum  coverMode, GLenum  transformType, const GLfloat * transformValues);
typedef void (GLAPIENTRY *PFNGLCOVERFILLPATHNVPROC) (GLuint  path, GLenum  coverMode);
typedef void (GLAPIENTRY *PFNGLCOVERSTROKEPATHINSTANCEDNVPROC) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLenum  coverMode, GLenum  transformType, const GLfloat * transformValues);
typedef void (GLAPIENTRY *PFNGLCOVERSTROKEPATHNVPROC) (GLuint  path, GLenum  coverMode);
typedef void (GLAPIENTRY *PFNGLCOVERAGEMODULATIONNVPROC) (GLenum  components);
typedef void (GLAPIENTRY *PFNGLCOVERAGEMODULATIONTABLENVPROC) (GLsizei  n, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLCREATEBUFFERSPROC) (GLsizei  n, GLuint * buffers);
typedef void (GLAPIENTRY *PFNGLCREATECOMMANDLISTSNVPROC) (GLsizei  n, GLuint * lists);
typedef void (GLAPIENTRY *PFNGLCREATEFRAMEBUFFERSPROC) (GLsizei  n, GLuint * framebuffers);
typedef void (GLAPIENTRY *PFNGLCREATEMEMORYOBJECTSEXTPROC) (GLsizei  n, GLuint * memoryObjects);
typedef void (GLAPIENTRY *PFNGLCREATEPERFQUERYINTELPROC) (GLuint  queryId, GLuint * queryHandle);
typedef GLuint (GLAPIENTRY *PFNGLCREATEPROGRAMPROC) ();
typedef GLhandleARB (GLAPIENTRY *PFNGLCREATEPROGRAMOBJECTARBPROC) ();
typedef void (GLAPIENTRY *PFNGLCREATEPROGRAMPIPELINESPROC) (GLsizei  n, GLuint * pipelines);
typedef void (GLAPIENTRY *PFNGLCREATEQUERIESPROC) (GLenum  target, GLsizei  n, GLuint * ids);
typedef void (GLAPIENTRY *PFNGLCREATERENDERBUFFERSPROC) (GLsizei  n, GLuint * renderbuffers);
typedef void (GLAPIENTRY *PFNGLCREATESAMPLERSPROC) (GLsizei  n, GLuint * samplers);
typedef GLuint (GLAPIENTRY *PFNGLCREATESHADERPROC) (GLenum  type);
typedef GLhandleARB (GLAPIENTRY *PFNGLCREATESHADEROBJECTARBPROC) (GLenum  shaderType);
typedef GLuint (GLAPIENTRY *PFNGLCREATESHADERPROGRAMEXTPROC) (GLenum  type, const GLchar * string);
typedef GLuint (GLAPIENTRY *PFNGLCREATESHADERPROGRAMVPROC) (GLenum  type, GLsizei  count, const GLchar *const* strings);
typedef GLuint (GLAPIENTRY *PFNGLCREATESHADERPROGRAMVEXTPROC) (GLenum  type, GLsizei  count, const GLchar ** strings);
typedef void (GLAPIENTRY *PFNGLCREATESTATESNVPROC) (GLsizei  n, GLuint * states);
typedef GLsync (GLAPIENTRY *PFNGLCREATESYNCFROMCLEVENTARBPROC) (struct _cl_context * context, struct _cl_event * event, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLCREATETEXTURESPROC) (GLenum  target, GLsizei  n, GLuint * textures);
typedef void (GLAPIENTRY *PFNGLCREATETRANSFORMFEEDBACKSPROC) (GLsizei  n, GLuint * ids);
typedef void (GLAPIENTRY *PFNGLCREATEVERTEXARRAYSPROC) (GLsizei  n, GLuint * arrays);
typedef void (GLAPIENTRY *PFNGLCULLFACEPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLCULLPARAMETERDVEXTPROC) (GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLCULLPARAMETERFVEXTPROC) (GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLCURRENTPALETTEMATRIXARBPROC) (GLint  index);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGECALLBACKPROC) (GLDEBUGPROC  callback, const void * userParam);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGECALLBACKAMDPROC) (GLDEBUGPROCAMD  callback, void * userParam);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGECALLBACKARBPROC) (GLDEBUGPROCARB  callback, const void * userParam);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGECALLBACKKHRPROC) (GLDEBUGPROCKHR  callback, const void * userParam);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGECONTROLPROC) (GLenum  source, GLenum  type, GLenum  severity, GLsizei  count, const GLuint * ids, GLboolean  enabled);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGECONTROLARBPROC) (GLenum  source, GLenum  type, GLenum  severity, GLsizei  count, const GLuint * ids, GLboolean  enabled);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGECONTROLKHRPROC) (GLenum  source, GLenum  type, GLenum  severity, GLsizei  count, const GLuint * ids, GLboolean  enabled);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGEENABLEAMDPROC) (GLenum  category, GLenum  severity, GLsizei  count, const GLuint * ids, GLboolean  enabled);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGEINSERTPROC) (GLenum  source, GLenum  type, GLuint  id, GLenum  severity, GLsizei  length, const GLchar * buf);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGEINSERTAMDPROC) (GLenum  category, GLenum  severity, GLuint  id, GLsizei  length, const GLchar * buf);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGEINSERTARBPROC) (GLenum  source, GLenum  type, GLuint  id, GLenum  severity, GLsizei  length, const GLchar * buf);
typedef void (GLAPIENTRY *PFNGLDEBUGMESSAGEINSERTKHRPROC) (GLenum  source, GLenum  type, GLuint  id, GLenum  severity, GLsizei  length, const GLchar * buf);
typedef void (GLAPIENTRY *PFNGLDELETEBUFFERSPROC) (GLsizei  n, const GLuint * buffers);
typedef void (GLAPIENTRY *PFNGLDELETEBUFFERSARBPROC) (GLsizei  n, const GLuint * buffers);
typedef void (GLAPIENTRY *PFNGLDELETECOMMANDLISTSNVPROC) (GLsizei  n, const GLuint * lists);
typedef void (GLAPIENTRY *PFNGLDELETEFENCESAPPLEPROC) (GLsizei  n, const GLuint * fences);
typedef void (GLAPIENTRY *PFNGLDELETEFENCESNVPROC) (GLsizei  n, const GLuint * fences);
typedef void (GLAPIENTRY *PFNGLDELETEFRAMEBUFFERSPROC) (GLsizei  n, const GLuint * framebuffers);
typedef void (GLAPIENTRY *PFNGLDELETEFRAMEBUFFERSEXTPROC) (GLsizei  n, const GLuint * framebuffers);
typedef void (GLAPIENTRY *PFNGLDELETELISTSPROC) (GLuint  list, GLsizei  range);
typedef void (GLAPIENTRY *PFNGLDELETEMEMORYOBJECTSEXTPROC) (GLsizei  n, const GLuint * memoryObjects);
typedef void (GLAPIENTRY *PFNGLDELETENAMEDSTRINGARBPROC) (GLint  namelen, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLDELETENAMESAMDPROC) (GLenum  identifier, GLuint  num, const GLuint * names);
typedef void (GLAPIENTRY *PFNGLDELETEOBJECTARBPROC) (GLhandleARB  obj);
typedef void (GLAPIENTRY *PFNGLDELETEOCCLUSIONQUERIESNVPROC) (GLsizei  n, const GLuint * ids);
typedef void (GLAPIENTRY *PFNGLDELETEPATHSNVPROC) (GLuint  path, GLsizei  range);
typedef void (GLAPIENTRY *PFNGLDELETEPERFMONITORSAMDPROC) (GLsizei  n, GLuint * monitors);
typedef void (GLAPIENTRY *PFNGLDELETEPERFQUERYINTELPROC) (GLuint  queryHandle);
typedef void (GLAPIENTRY *PFNGLDELETEPROGRAMPROC) (GLuint  program);
typedef void (GLAPIENTRY *PFNGLDELETEPROGRAMPIPELINESPROC) (GLsizei  n, const GLuint * pipelines);
typedef void (GLAPIENTRY *PFNGLDELETEPROGRAMPIPELINESEXTPROC) (GLsizei  n, const GLuint * pipelines);
typedef void (GLAPIENTRY *PFNGLDELETEPROGRAMSARBPROC) (GLsizei  n, const GLuint * programs);
typedef void (GLAPIENTRY *PFNGLDELETEPROGRAMSNVPROC) (GLsizei  n, const GLuint * programs);
typedef void (GLAPIENTRY *PFNGLDELETEQUERIESPROC) (GLsizei  n, const GLuint * ids);
typedef void (GLAPIENTRY *PFNGLDELETEQUERIESARBPROC) (GLsizei  n, const GLuint * ids);
typedef void (GLAPIENTRY *PFNGLDELETEQUERYRESOURCETAGNVPROC) (GLsizei  n, const GLint * tagIds);
typedef void (GLAPIENTRY *PFNGLDELETERENDERBUFFERSPROC) (GLsizei  n, const GLuint * renderbuffers);
typedef void (GLAPIENTRY *PFNGLDELETERENDERBUFFERSEXTPROC) (GLsizei  n, const GLuint * renderbuffers);
typedef void (GLAPIENTRY *PFNGLDELETESAMPLERSPROC) (GLsizei  count, const GLuint * samplers);
typedef void (GLAPIENTRY *PFNGLDELETESEMAPHORESEXTPROC) (GLsizei  n, const GLuint * semaphores);
typedef void (GLAPIENTRY *PFNGLDELETESHADERPROC) (GLuint  shader);
typedef void (GLAPIENTRY *PFNGLDELETESTATESNVPROC) (GLsizei  n, const GLuint * states);
typedef void (GLAPIENTRY *PFNGLDELETESYNCPROC) (GLsync  sync);
typedef void (GLAPIENTRY *PFNGLDELETETEXTURESPROC) (GLsizei  n, const GLuint * textures);
typedef void (GLAPIENTRY *PFNGLDELETETEXTURESEXTPROC) (GLsizei  n, const GLuint * textures);
typedef void (GLAPIENTRY *PFNGLDELETETRANSFORMFEEDBACKSPROC) (GLsizei  n, const GLuint * ids);
typedef void (GLAPIENTRY *PFNGLDELETETRANSFORMFEEDBACKSNVPROC) (GLsizei  n, const GLuint * ids);
typedef void (GLAPIENTRY *PFNGLDELETEVERTEXARRAYSPROC) (GLsizei  n, const GLuint * arrays);
typedef void (GLAPIENTRY *PFNGLDELETEVERTEXARRAYSAPPLEPROC) (GLsizei  n, const GLuint * arrays);
typedef void (GLAPIENTRY *PFNGLDELETEVERTEXSHADEREXTPROC) (GLuint  id);
typedef void (GLAPIENTRY *PFNGLDEPTHBOUNDSEXTPROC) (GLclampd  zmin, GLclampd  zmax);
typedef void (GLAPIENTRY *PFNGLDEPTHBOUNDSDNVPROC) (GLdouble  zmin, GLdouble  zmax);
typedef void (GLAPIENTRY *PFNGLDEPTHFUNCPROC) (GLenum  func);
typedef void (GLAPIENTRY *PFNGLDEPTHMASKPROC) (GLboolean  flag);
typedef void (GLAPIENTRY *PFNGLDEPTHRANGEPROC) (GLdouble  n, GLdouble  f);
typedef void (GLAPIENTRY *PFNGLDEPTHRANGEARRAYDVNVPROC) (GLuint  first, GLsizei  count, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLDEPTHRANGEARRAYVPROC) (GLuint  first, GLsizei  count, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLDEPTHRANGEINDEXEDPROC) (GLuint  index, GLdouble  n, GLdouble  f);
typedef void (GLAPIENTRY *PFNGLDEPTHRANGEINDEXEDDNVPROC) (GLuint  index, GLdouble  n, GLdouble  f);
typedef void (GLAPIENTRY *PFNGLDEPTHRANGEDNVPROC) (GLdouble  zNear, GLdouble  zFar);
typedef void (GLAPIENTRY *PFNGLDEPTHRANGEFPROC) (GLfloat  n, GLfloat  f);
typedef void (GLAPIENTRY *PFNGLDETACHOBJECTARBPROC) (GLhandleARB  containerObj, GLhandleARB  attachedObj);
typedef void (GLAPIENTRY *PFNGLDETACHSHADERPROC) (GLuint  program, GLuint  shader);
typedef void (GLAPIENTRY *PFNGLDISABLEPROC) (GLenum  cap);
typedef void (GLAPIENTRY *PFNGLDISABLECLIENTSTATEPROC) (GLenum  array);
typedef void (GLAPIENTRY *PFNGLDISABLECLIENTSTATEINDEXEDEXTPROC) (GLenum  array, GLuint  index);
typedef void (GLAPIENTRY *PFNGLDISABLECLIENTSTATEIEXTPROC) (GLenum  array, GLuint  index);
typedef void (GLAPIENTRY *PFNGLDISABLEINDEXEDEXTPROC) (GLenum  target, GLuint  index);
typedef void (GLAPIENTRY *PFNGLDISABLEVARIANTCLIENTSTATEEXTPROC) (GLuint  id);
typedef void (GLAPIENTRY *PFNGLDISABLEVERTEXARRAYATTRIBPROC) (GLuint  vaobj, GLuint  index);
typedef void (GLAPIENTRY *PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC) (GLuint  vaobj, GLuint  index);
typedef void (GLAPIENTRY *PFNGLDISABLEVERTEXARRAYEXTPROC) (GLuint  vaobj, GLenum  array);
typedef void (GLAPIENTRY *PFNGLDISABLEVERTEXATTRIBAPPLEPROC) (GLuint  index, GLenum  pname);
typedef void (GLAPIENTRY *PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint  index);
typedef void (GLAPIENTRY *PFNGLDISABLEVERTEXATTRIBARRAYARBPROC) (GLuint  index);
typedef void (GLAPIENTRY *PFNGLDISABLEIPROC) (GLenum  target, GLuint  index);
typedef void (GLAPIENTRY *PFNGLDISPATCHCOMPUTEPROC) (GLuint  num_groups_x, GLuint  num_groups_y, GLuint  num_groups_z);
typedef void (GLAPIENTRY *PFNGLDISPATCHCOMPUTEGROUPSIZEARBPROC) (GLuint  num_groups_x, GLuint  num_groups_y, GLuint  num_groups_z, GLuint  group_size_x, GLuint  group_size_y, GLuint  group_size_z);
typedef void (GLAPIENTRY *PFNGLDISPATCHCOMPUTEINDIRECTPROC) (GLintptr  indirect);
typedef void (GLAPIENTRY *PFNGLDRAWARRAYSPROC) (GLenum  mode, GLint  first, GLsizei  count);
typedef void (GLAPIENTRY *PFNGLDRAWARRAYSEXTPROC) (GLenum  mode, GLint  first, GLsizei  count);
typedef void (GLAPIENTRY *PFNGLDRAWARRAYSINDIRECTPROC) (GLenum  mode, const void * indirect);
typedef void (GLAPIENTRY *PFNGLDRAWARRAYSINSTANCEDPROC) (GLenum  mode, GLint  first, GLsizei  count, GLsizei  instancecount);
typedef void (GLAPIENTRY *PFNGLDRAWARRAYSINSTANCEDARBPROC) (GLenum  mode, GLint  first, GLsizei  count, GLsizei  primcount);
typedef void (GLAPIENTRY *PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC) (GLenum  mode, GLint  first, GLsizei  count, GLsizei  instancecount, GLuint  baseinstance);
typedef void (GLAPIENTRY *PFNGLDRAWARRAYSINSTANCEDEXTPROC) (GLenum  mode, GLint  start, GLsizei  count, GLsizei  primcount);
typedef void (GLAPIENTRY *PFNGLDRAWBUFFERPROC) (GLenum  buf);
typedef void (GLAPIENTRY *PFNGLDRAWBUFFERSPROC) (GLsizei  n, const GLenum * bufs);
typedef void (GLAPIENTRY *PFNGLDRAWBUFFERSARBPROC) (GLsizei  n, const GLenum * bufs);
typedef void (GLAPIENTRY *PFNGLDRAWCOMMANDSADDRESSNVPROC) (GLenum  primitiveMode, const GLuint64 * indirects, const GLsizei * sizes, GLuint  count);
typedef void (GLAPIENTRY *PFNGLDRAWCOMMANDSNVPROC) (GLenum  primitiveMode, GLuint  buffer, const GLintptr * indirects, const GLsizei * sizes, GLuint  count);
typedef void (GLAPIENTRY *PFNGLDRAWCOMMANDSSTATESADDRESSNVPROC) (const GLuint64 * indirects, const GLsizei * sizes, const GLuint * states, const GLuint * fbos, GLuint  count);
typedef void (GLAPIENTRY *PFNGLDRAWCOMMANDSSTATESNVPROC) (GLuint  buffer, const GLintptr * indirects, const GLsizei * sizes, const GLuint * states, const GLuint * fbos, GLuint  count);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTARRAYAPPLEPROC) (GLenum  mode, GLint  first, GLsizei  count);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSPROC) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSBASEVERTEXPROC) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLint  basevertex);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSINDIRECTPROC) (GLenum  mode, GLenum  type, const void * indirect);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSINSTANCEDPROC) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  instancecount);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSINSTANCEDARBPROC) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  primcount);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  instancecount, GLuint  baseinstance);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  instancecount, GLint  basevertex);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  instancecount, GLint  basevertex, GLuint  baseinstance);
typedef void (GLAPIENTRY *PFNGLDRAWELEMENTSINSTANCEDEXTPROC) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  primcount);
typedef void (GLAPIENTRY *PFNGLDRAWMESHTASKSNVPROC) (GLuint  first, GLuint  count);
typedef void (GLAPIENTRY *PFNGLDRAWMESHTASKSINDIRECTNVPROC) (GLintptr  indirect);
typedef void (GLAPIENTRY *PFNGLDRAWPIXELSPROC) (GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLDRAWRANGEELEMENTARRAYAPPLEPROC) (GLenum  mode, GLuint  start, GLuint  end, GLint  first, GLsizei  count);
typedef void (GLAPIENTRY *PFNGLDRAWRANGEELEMENTSPROC) (GLenum  mode, GLuint  start, GLuint  end, GLsizei  count, GLenum  type, const void * indices);
typedef void (GLAPIENTRY *PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC) (GLenum  mode, GLuint  start, GLuint  end, GLsizei  count, GLenum  type, const void * indices, GLint  basevertex);
typedef void (GLAPIENTRY *PFNGLDRAWRANGEELEMENTSEXTPROC) (GLenum  mode, GLuint  start, GLuint  end, GLsizei  count, GLenum  type, const void * indices);
typedef void (GLAPIENTRY *PFNGLDRAWTEXTURENVPROC) (GLuint  texture, GLuint  sampler, GLfloat  x0, GLfloat  y0, GLfloat  x1, GLfloat  y1, GLfloat  z, GLfloat  s0, GLfloat  t0, GLfloat  s1, GLfloat  t1);
typedef void (GLAPIENTRY *PFNGLDRAWTRANSFORMFEEDBACKPROC) (GLenum  mode, GLuint  id);
typedef void (GLAPIENTRY *PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC) (GLenum  mode, GLuint  id, GLsizei  instancecount);
typedef void (GLAPIENTRY *PFNGLDRAWTRANSFORMFEEDBACKNVPROC) (GLenum  mode, GLuint  id);
typedef void (GLAPIENTRY *PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC) (GLenum  mode, GLuint  id, GLuint  stream);
typedef void (GLAPIENTRY *PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC) (GLenum  mode, GLuint  id, GLuint  stream, GLsizei  instancecount);
typedef void (GLAPIENTRY *PFNGLEGLIMAGETARGETTEXSTORAGEEXTPROC) (GLenum  target, GLeglImageOES  image, const GLint*  attrib_list);
typedef void (GLAPIENTRY *PFNGLEGLIMAGETARGETTEXTURESTORAGEEXTPROC) (GLuint  texture, GLeglImageOES  image, const GLint*  attrib_list);
typedef void (GLAPIENTRY *PFNGLEDGEFLAGPROC) (GLboolean  flag);
typedef void (GLAPIENTRY *PFNGLEDGEFLAGFORMATNVPROC) (GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLEDGEFLAGPOINTERPROC) (GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLEDGEFLAGPOINTEREXTPROC) (GLsizei  stride, GLsizei  count, const GLboolean * pointer);
typedef void (GLAPIENTRY *PFNGLEDGEFLAGVPROC) (const GLboolean * flag);
typedef void (GLAPIENTRY *PFNGLELEMENTPOINTERAPPLEPROC) (GLenum  type, const void * pointer);
typedef void (GLAPIENTRY *PFNGLENABLEPROC) (GLenum  cap);
typedef void (GLAPIENTRY *PFNGLENABLECLIENTSTATEPROC) (GLenum  array);
typedef void (GLAPIENTRY *PFNGLENABLECLIENTSTATEINDEXEDEXTPROC) (GLenum  array, GLuint  index);
typedef void (GLAPIENTRY *PFNGLENABLECLIENTSTATEIEXTPROC) (GLenum  array, GLuint  index);
typedef void (GLAPIENTRY *PFNGLENABLEINDEXEDEXTPROC) (GLenum  target, GLuint  index);
typedef void (GLAPIENTRY *PFNGLENABLEVARIANTCLIENTSTATEEXTPROC) (GLuint  id);
typedef void (GLAPIENTRY *PFNGLENABLEVERTEXARRAYATTRIBPROC) (GLuint  vaobj, GLuint  index);
typedef void (GLAPIENTRY *PFNGLENABLEVERTEXARRAYATTRIBEXTPROC) (GLuint  vaobj, GLuint  index);
typedef void (GLAPIENTRY *PFNGLENABLEVERTEXARRAYEXTPROC) (GLuint  vaobj, GLenum  array);
typedef void (GLAPIENTRY *PFNGLENABLEVERTEXATTRIBAPPLEPROC) (GLuint  index, GLenum  pname);
typedef void (GLAPIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint  index);
typedef void (GLAPIENTRY *PFNGLENABLEVERTEXATTRIBARRAYARBPROC) (GLuint  index);
typedef void (GLAPIENTRY *PFNGLENABLEIPROC) (GLenum  target, GLuint  index);
typedef void (GLAPIENTRY *PFNGLENDPROC) ();
typedef void (GLAPIENTRY *PFNGLENDCONDITIONALRENDERPROC) ();
typedef void (GLAPIENTRY *PFNGLENDCONDITIONALRENDERNVPROC) ();
typedef void (GLAPIENTRY *PFNGLENDLISTPROC) ();
typedef void (GLAPIENTRY *PFNGLENDOCCLUSIONQUERYNVPROC) ();
typedef void (GLAPIENTRY *PFNGLENDPERFMONITORAMDPROC) (GLuint  monitor);
typedef void (GLAPIENTRY *PFNGLENDPERFQUERYINTELPROC) (GLuint  queryHandle);
typedef void (GLAPIENTRY *PFNGLENDQUERYPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLENDQUERYARBPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLENDQUERYINDEXEDPROC) (GLenum  target, GLuint  index);
typedef void (GLAPIENTRY *PFNGLENDTRANSFORMFEEDBACKPROC) ();
typedef void (GLAPIENTRY *PFNGLENDTRANSFORMFEEDBACKEXTPROC) ();
typedef void (GLAPIENTRY *PFNGLENDTRANSFORMFEEDBACKNVPROC) ();
typedef void (GLAPIENTRY *PFNGLENDVERTEXSHADEREXTPROC) ();
typedef void (GLAPIENTRY *PFNGLENDVIDEOCAPTURENVPROC) (GLuint  video_capture_slot);
typedef void (GLAPIENTRY *PFNGLEVALCOORD1DPROC) (GLdouble  u);
typedef void (GLAPIENTRY *PFNGLEVALCOORD1DVPROC) (const GLdouble * u);
typedef void (GLAPIENTRY *PFNGLEVALCOORD1FPROC) (GLfloat  u);
typedef void (GLAPIENTRY *PFNGLEVALCOORD1FVPROC) (const GLfloat * u);
typedef void (GLAPIENTRY *PFNGLEVALCOORD2DPROC) (GLdouble  u, GLdouble  v);
typedef void (GLAPIENTRY *PFNGLEVALCOORD2DVPROC) (const GLdouble * u);
typedef void (GLAPIENTRY *PFNGLEVALCOORD2FPROC) (GLfloat  u, GLfloat  v);
typedef void (GLAPIENTRY *PFNGLEVALCOORD2FVPROC) (const GLfloat * u);
typedef void (GLAPIENTRY *PFNGLEVALMAPSNVPROC) (GLenum  target, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLEVALMESH1PROC) (GLenum  mode, GLint  i1, GLint  i2);
typedef void (GLAPIENTRY *PFNGLEVALMESH2PROC) (GLenum  mode, GLint  i1, GLint  i2, GLint  j1, GLint  j2);
typedef void (GLAPIENTRY *PFNGLEVALPOINT1PROC) (GLint  i);
typedef void (GLAPIENTRY *PFNGLEVALPOINT2PROC) (GLint  i, GLint  j);
typedef void (GLAPIENTRY *PFNGLEVALUATEDEPTHVALUESARBPROC) ();
typedef void (GLAPIENTRY *PFNGLEXECUTEPROGRAMNVPROC) (GLenum  target, GLuint  id, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLEXTRACTCOMPONENTEXTPROC) (GLuint  res, GLuint  src, GLuint  num);
typedef void (GLAPIENTRY *PFNGLFEEDBACKBUFFERPROC) (GLsizei  size, GLenum  type, GLfloat * buffer);
typedef GLsync (GLAPIENTRY *PFNGLFENCESYNCPROC) (GLenum  condition, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLFINALCOMBINERINPUTNVPROC) (GLenum  variable, GLenum  input, GLenum  mapping, GLenum  componentUsage);
typedef void (GLAPIENTRY *PFNGLFINISHPROC) ();
typedef void (GLAPIENTRY *PFNGLFINISHFENCEAPPLEPROC) (GLuint  fence);
typedef void (GLAPIENTRY *PFNGLFINISHFENCENVPROC) (GLuint  fence);
typedef void (GLAPIENTRY *PFNGLFINISHOBJECTAPPLEPROC) (GLenum  object, GLint  name);
typedef void (GLAPIENTRY *PFNGLFLUSHPROC) ();
typedef void (GLAPIENTRY *PFNGLFLUSHMAPPEDBUFFERRANGEPROC) (GLenum  target, GLintptr  offset, GLsizeiptr  length);
typedef void (GLAPIENTRY *PFNGLFLUSHMAPPEDBUFFERRANGEAPPLEPROC) (GLenum  target, GLintptr  offset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length);
typedef void (GLAPIENTRY *PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length);
typedef void (GLAPIENTRY *PFNGLFLUSHPIXELDATARANGENVPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLFLUSHVERTEXARRAYRANGEAPPLEPROC) (GLsizei  length, void * pointer);
typedef void (GLAPIENTRY *PFNGLFLUSHVERTEXARRAYRANGENVPROC) ();
typedef void (GLAPIENTRY *PFNGLFOGCOORDFORMATNVPROC) (GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLFOGCOORDPOINTERPROC) (GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLFOGCOORDPOINTEREXTPROC) (GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLFOGCOORDDPROC) (GLdouble  coord);
typedef void (GLAPIENTRY *PFNGLFOGCOORDDEXTPROC) (GLdouble  coord);
typedef void (GLAPIENTRY *PFNGLFOGCOORDDVPROC) (const GLdouble * coord);
typedef void (GLAPIENTRY *PFNGLFOGCOORDDVEXTPROC) (const GLdouble * coord);
typedef void (GLAPIENTRY *PFNGLFOGCOORDFPROC) (GLfloat  coord);
typedef void (GLAPIENTRY *PFNGLFOGCOORDFEXTPROC) (GLfloat  coord);
typedef void (GLAPIENTRY *PFNGLFOGCOORDFVPROC) (const GLfloat * coord);
typedef void (GLAPIENTRY *PFNGLFOGCOORDFVEXTPROC) (const GLfloat * coord);
typedef void (GLAPIENTRY *PFNGLFOGCOORDHNVPROC) (GLhalfNV  fog);
typedef void (GLAPIENTRY *PFNGLFOGCOORDHVNVPROC) (const GLhalfNV * fog);
typedef void (GLAPIENTRY *PFNGLFOGFPROC) (GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLFOGFVPROC) (GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLFOGIPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLFOGIVPROC) (GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLFRAGMENTCOVERAGECOLORNVPROC) (GLuint  color);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC) (GLuint  framebuffer, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC) (GLuint  framebuffer, GLsizei  n, const GLenum * bufs);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERFETCHBARRIEREXTPROC) ();
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERPARAMETERIPROC) (GLenum  target, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERREADBUFFEREXTPROC) (GLuint  framebuffer, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERRENDERBUFFERPROC) (GLenum  target, GLenum  attachment, GLenum  renderbuffertarget, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC) (GLenum  target, GLenum  attachment, GLenum  renderbuffertarget, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERSAMPLELOCATIONSFVARBPROC) (GLenum  target, GLuint  start, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERSAMPLELOCATIONSFVNVPROC) (GLenum  target, GLuint  start, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERSAMPLEPOSITIONSFVAMDPROC) (GLenum  target, GLuint  numsamples, GLuint  pixelindex, const GLfloat * values);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTUREPROC) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURE1DPROC) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURE1DEXTPROC) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURE2DPROC) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURE2DEXTPROC) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURE3DPROC) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level, GLint  zoffset);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURE3DEXTPROC) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level, GLint  zoffset);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTUREARBPROC) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTUREEXTPROC) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTUREFACEARBPROC) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLenum  face);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTUREFACEEXTPROC) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLenum  face);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURELAYERPROC) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURELAYERARBPROC) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
typedef void (GLAPIENTRY *PFNGLFRAMEBUFFERTEXTURELAYEREXTPROC) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
typedef void (GLAPIENTRY *PFNGLFRONTFACEPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLFRUSTUMPROC) (GLdouble  left, GLdouble  right, GLdouble  bottom, GLdouble  top, GLdouble  zNear, GLdouble  zFar);
typedef void (GLAPIENTRY *PFNGLGENBUFFERSPROC) (GLsizei  n, GLuint * buffers);
typedef void (GLAPIENTRY *PFNGLGENBUFFERSARBPROC) (GLsizei  n, GLuint * buffers);
typedef void (GLAPIENTRY *PFNGLGENFENCESAPPLEPROC) (GLsizei  n, GLuint * fences);
typedef void (GLAPIENTRY *PFNGLGENFENCESNVPROC) (GLsizei  n, GLuint * fences);
typedef void (GLAPIENTRY *PFNGLGENFRAMEBUFFERSPROC) (GLsizei  n, GLuint * framebuffers);
typedef void (GLAPIENTRY *PFNGLGENFRAMEBUFFERSEXTPROC) (GLsizei  n, GLuint * framebuffers);
typedef GLuint (GLAPIENTRY *PFNGLGENLISTSPROC) (GLsizei  range);
typedef void (GLAPIENTRY *PFNGLGENNAMESAMDPROC) (GLenum  identifier, GLuint  num, GLuint * names);
typedef void (GLAPIENTRY *PFNGLGENOCCLUSIONQUERIESNVPROC) (GLsizei  n, GLuint * ids);
typedef GLuint (GLAPIENTRY *PFNGLGENPATHSNVPROC) (GLsizei  range);
typedef void (GLAPIENTRY *PFNGLGENPERFMONITORSAMDPROC) (GLsizei  n, GLuint * monitors);
typedef void (GLAPIENTRY *PFNGLGENPROGRAMPIPELINESPROC) (GLsizei  n, GLuint * pipelines);
typedef void (GLAPIENTRY *PFNGLGENPROGRAMPIPELINESEXTPROC) (GLsizei  n, GLuint * pipelines);
typedef void (GLAPIENTRY *PFNGLGENPROGRAMSARBPROC) (GLsizei  n, GLuint * programs);
typedef void (GLAPIENTRY *PFNGLGENPROGRAMSNVPROC) (GLsizei  n, GLuint * programs);
typedef void (GLAPIENTRY *PFNGLGENQUERIESPROC) (GLsizei  n, GLuint * ids);
typedef void (GLAPIENTRY *PFNGLGENQUERIESARBPROC) (GLsizei  n, GLuint * ids);
typedef void (GLAPIENTRY *PFNGLGENQUERYRESOURCETAGNVPROC) (GLsizei  n, GLint * tagIds);
typedef void (GLAPIENTRY *PFNGLGENRENDERBUFFERSPROC) (GLsizei  n, GLuint * renderbuffers);
typedef void (GLAPIENTRY *PFNGLGENRENDERBUFFERSEXTPROC) (GLsizei  n, GLuint * renderbuffers);
typedef void (GLAPIENTRY *PFNGLGENSAMPLERSPROC) (GLsizei  count, GLuint * samplers);
typedef void (GLAPIENTRY *PFNGLGENSEMAPHORESEXTPROC) (GLsizei  n, GLuint * semaphores);
typedef GLuint (GLAPIENTRY *PFNGLGENSYMBOLSEXTPROC) (GLenum  datatype, GLenum  storagetype, GLenum  range, GLuint  components);
typedef void (GLAPIENTRY *PFNGLGENTEXTURESPROC) (GLsizei  n, GLuint * textures);
typedef void (GLAPIENTRY *PFNGLGENTEXTURESEXTPROC) (GLsizei  n, GLuint * textures);
typedef void (GLAPIENTRY *PFNGLGENTRANSFORMFEEDBACKSPROC) (GLsizei  n, GLuint * ids);
typedef void (GLAPIENTRY *PFNGLGENTRANSFORMFEEDBACKSNVPROC) (GLsizei  n, GLuint * ids);
typedef void (GLAPIENTRY *PFNGLGENVERTEXARRAYSPROC) (GLsizei  n, GLuint * arrays);
typedef void (GLAPIENTRY *PFNGLGENVERTEXARRAYSAPPLEPROC) (GLsizei  n, GLuint * arrays);
typedef GLuint (GLAPIENTRY *PFNGLGENVERTEXSHADERSEXTPROC) (GLuint  range);
typedef void (GLAPIENTRY *PFNGLGENERATEMIPMAPPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLGENERATEMIPMAPEXTPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLGENERATEMULTITEXMIPMAPEXTPROC) (GLenum  texunit, GLenum  target);
typedef void (GLAPIENTRY *PFNGLGENERATETEXTUREMIPMAPPROC) (GLuint  texture);
typedef void (GLAPIENTRY *PFNGLGENERATETEXTUREMIPMAPEXTPROC) (GLuint  texture, GLenum  target);
typedef void (GLAPIENTRY *PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC) (GLuint  program, GLuint  bufferIndex, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETACTIVEATTRIBPROC) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLint * size, GLenum * type, GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETACTIVEATTRIBARBPROC) (GLhandleARB  programObj, GLuint  index, GLsizei  maxLength, GLsizei * length, GLint * size, GLenum * type, GLcharARB * name);
typedef void (GLAPIENTRY *PFNGLGETACTIVESUBROUTINENAMEPROC) (GLuint  program, GLenum  shadertype, GLuint  index, GLsizei  bufSize, GLsizei * length, GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC) (GLuint  program, GLenum  shadertype, GLuint  index, GLsizei  bufSize, GLsizei * length, GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC) (GLuint  program, GLenum  shadertype, GLuint  index, GLenum  pname, GLint * values);
typedef void (GLAPIENTRY *PFNGLGETACTIVEUNIFORMPROC) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLint * size, GLenum * type, GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETACTIVEUNIFORMARBPROC) (GLhandleARB  programObj, GLuint  index, GLsizei  maxLength, GLsizei * length, GLint * size, GLenum * type, GLcharARB * name);
typedef void (GLAPIENTRY *PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC) (GLuint  program, GLuint  uniformBlockIndex, GLsizei  bufSize, GLsizei * length, GLchar * uniformBlockName);
typedef void (GLAPIENTRY *PFNGLGETACTIVEUNIFORMBLOCKIVPROC) (GLuint  program, GLuint  uniformBlockIndex, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETACTIVEUNIFORMNAMEPROC) (GLuint  program, GLuint  uniformIndex, GLsizei  bufSize, GLsizei * length, GLchar * uniformName);
typedef void (GLAPIENTRY *PFNGLGETACTIVEUNIFORMSIVPROC) (GLuint  program, GLsizei  uniformCount, const GLuint * uniformIndices, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETACTIVEVARYINGNVPROC) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETATTACHEDOBJECTSARBPROC) (GLhandleARB  containerObj, GLsizei  maxCount, GLsizei * count, GLhandleARB * obj);
typedef void (GLAPIENTRY *PFNGLGETATTACHEDSHADERSPROC) (GLuint  program, GLsizei  maxCount, GLsizei * count, GLuint * shaders);
typedef GLint (GLAPIENTRY *PFNGLGETATTRIBLOCATIONPROC) (GLuint  program, const GLchar * name);
typedef GLint (GLAPIENTRY *PFNGLGETATTRIBLOCATIONARBPROC) (GLhandleARB  programObj, const GLcharARB * name);
typedef void (GLAPIENTRY *PFNGLGETBOOLEANINDEXEDVEXTPROC) (GLenum  target, GLuint  index, GLboolean * data);
typedef void (GLAPIENTRY *PFNGLGETBOOLEANI_VPROC) (GLenum  target, GLuint  index, GLboolean * data);
typedef void (GLAPIENTRY *PFNGLGETBOOLEANVPROC) (GLenum  pname, GLboolean * data);
typedef void (GLAPIENTRY *PFNGLGETBUFFERPARAMETERI64VPROC) (GLenum  target, GLenum  pname, GLint64 * params);
typedef void (GLAPIENTRY *PFNGLGETBUFFERPARAMETERIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETBUFFERPARAMETERIVARBPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETBUFFERPARAMETERUI64VNVPROC) (GLenum  target, GLenum  pname, GLuint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETBUFFERPOINTERVPROC) (GLenum  target, GLenum  pname, void ** params);
typedef void (GLAPIENTRY *PFNGLGETBUFFERPOINTERVARBPROC) (GLenum  target, GLenum  pname, void ** params);
typedef void (GLAPIENTRY *PFNGLGETBUFFERSUBDATAPROC) (GLenum  target, GLintptr  offset, GLsizeiptr  size, void * data);
typedef void (GLAPIENTRY *PFNGLGETBUFFERSUBDATAARBPROC) (GLenum  target, GLintptrARB  offset, GLsizeiptrARB  size, void * data);
typedef void (GLAPIENTRY *PFNGLGETCLIPPLANEPROC) (GLenum  plane, GLdouble * equation);
typedef void (GLAPIENTRY *PFNGLGETCOLORTABLEPROC) (GLenum  target, GLenum  format, GLenum  type, void * table);
typedef void (GLAPIENTRY *PFNGLGETCOLORTABLEEXTPROC) (GLenum  target, GLenum  format, GLenum  type, void * data);
typedef void (GLAPIENTRY *PFNGLGETCOLORTABLEPARAMETERFVPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETCOLORTABLEPARAMETERFVEXTPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETCOLORTABLEPARAMETERIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETCOLORTABLEPARAMETERIVEXTPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC) (GLenum  stage, GLenum  portion, GLenum  variable, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC) (GLenum  stage, GLenum  portion, GLenum  variable, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC) (GLenum  stage, GLenum  portion, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC) (GLenum  stage, GLenum  portion, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC) (GLenum  stage, GLenum  pname, GLfloat * params);
typedef GLuint (GLAPIENTRY *PFNGLGETCOMMANDHEADERNVPROC) (GLenum  tokenID, GLuint  size);
typedef void (GLAPIENTRY *PFNGLGETCOMPRESSEDMULTITEXIMAGEEXTPROC) (GLenum  texunit, GLenum  target, GLint  lod, void * img);
typedef void (GLAPIENTRY *PFNGLGETCOMPRESSEDTEXIMAGEPROC) (GLenum  target, GLint  level, void * img);
typedef void (GLAPIENTRY *PFNGLGETCOMPRESSEDTEXIMAGEARBPROC) (GLenum  target, GLint  level, void * img);
typedef void (GLAPIENTRY *PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC) (GLuint  texture, GLint  level, GLsizei  bufSize, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETCOMPRESSEDTEXTUREIMAGEEXTPROC) (GLuint  texture, GLenum  target, GLint  lod, void * img);
typedef void (GLAPIENTRY *PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLsizei  bufSize, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETCONVOLUTIONFILTERPROC) (GLenum  target, GLenum  format, GLenum  type, void * image);
typedef void (GLAPIENTRY *PFNGLGETCONVOLUTIONFILTEREXTPROC) (GLenum  target, GLenum  format, GLenum  type, void * image);
typedef void (GLAPIENTRY *PFNGLGETCONVOLUTIONPARAMETERFVPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETCONVOLUTIONPARAMETERFVEXTPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETCONVOLUTIONPARAMETERIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETCONVOLUTIONPARAMETERIVEXTPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETCOVERAGEMODULATIONTABLENVPROC) (GLsizei  bufSize, GLfloat * v);
typedef GLuint (GLAPIENTRY *PFNGLGETDEBUGMESSAGELOGPROC) (GLuint  count, GLsizei  bufSize, GLenum * sources, GLenum * types, GLuint * ids, GLenum * severities, GLsizei * lengths, GLchar * messageLog);
typedef GLuint (GLAPIENTRY *PFNGLGETDEBUGMESSAGELOGAMDPROC) (GLuint  count, GLsizei  bufSize, GLenum * categories, GLuint * severities, GLuint * ids, GLsizei * lengths, GLchar * message);
typedef GLuint (GLAPIENTRY *PFNGLGETDEBUGMESSAGELOGARBPROC) (GLuint  count, GLsizei  bufSize, GLenum * sources, GLenum * types, GLuint * ids, GLenum * severities, GLsizei * lengths, GLchar * messageLog);
typedef GLuint (GLAPIENTRY *PFNGLGETDEBUGMESSAGELOGKHRPROC) (GLuint  count, GLsizei  bufSize, GLenum * sources, GLenum * types, GLuint * ids, GLenum * severities, GLsizei * lengths, GLchar * messageLog);
typedef void (GLAPIENTRY *PFNGLGETDOUBLEINDEXEDVEXTPROC) (GLenum  target, GLuint  index, GLdouble * data);
typedef void (GLAPIENTRY *PFNGLGETDOUBLEI_VPROC) (GLenum  target, GLuint  index, GLdouble * data);
typedef void (GLAPIENTRY *PFNGLGETDOUBLEI_VEXTPROC) (GLenum  pname, GLuint  index, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETDOUBLEVPROC) (GLenum  pname, GLdouble * data);
typedef GLenum (GLAPIENTRY *PFNGLGETERRORPROC) ();
typedef void (GLAPIENTRY *PFNGLGETFENCEIVNVPROC) (GLuint  fence, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC) (GLenum  variable, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC) (GLenum  variable, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETFIRSTPERFQUERYIDINTELPROC) (GLuint * queryId);
typedef void (GLAPIENTRY *PFNGLGETFLOATINDEXEDVEXTPROC) (GLenum  target, GLuint  index, GLfloat * data);
typedef void (GLAPIENTRY *PFNGLGETFLOATI_VPROC) (GLenum  target, GLuint  index, GLfloat * data);
typedef void (GLAPIENTRY *PFNGLGETFLOATI_VEXTPROC) (GLenum  pname, GLuint  index, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETFLOATVPROC) (GLenum  pname, GLfloat * data);
typedef GLint (GLAPIENTRY *PFNGLGETFRAGDATAINDEXPROC) (GLuint  program, const GLchar * name);
typedef GLint (GLAPIENTRY *PFNGLGETFRAGDATALOCATIONPROC) (GLuint  program, const GLchar * name);
typedef GLint (GLAPIENTRY *PFNGLGETFRAGDATALOCATIONEXTPROC) (GLuint  program, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC) (GLenum  target, GLenum  attachment, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC) (GLenum  target, GLenum  attachment, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETFRAMEBUFFERPARAMETERFVAMDPROC) (GLenum  target, GLenum  pname, GLuint  numsamples, GLuint  pixelindex, GLsizei  size, GLfloat * values);
typedef void (GLAPIENTRY *PFNGLGETFRAMEBUFFERPARAMETERIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETFRAMEBUFFERPARAMETERIVEXTPROC) (GLuint  framebuffer, GLenum  pname, GLint * params);
typedef GLenum (GLAPIENTRY *PFNGLGETGRAPHICSRESETSTATUSPROC) ();
typedef GLenum (GLAPIENTRY *PFNGLGETGRAPHICSRESETSTATUSARBPROC) ();
typedef GLenum (GLAPIENTRY *PFNGLGETGRAPHICSRESETSTATUSKHRPROC) ();
typedef GLhandleARB (GLAPIENTRY *PFNGLGETHANDLEARBPROC) (GLenum  pname);
typedef void (GLAPIENTRY *PFNGLGETHISTOGRAMPROC) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, void * values);
typedef void (GLAPIENTRY *PFNGLGETHISTOGRAMEXTPROC) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, void * values);
typedef void (GLAPIENTRY *PFNGLGETHISTOGRAMPARAMETERFVPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETHISTOGRAMPARAMETERFVEXTPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETHISTOGRAMPARAMETERIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETHISTOGRAMPARAMETERIVEXTPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef GLuint64 (GLAPIENTRY *PFNGLGETIMAGEHANDLEARBPROC) (GLuint  texture, GLint  level, GLboolean  layered, GLint  layer, GLenum  format);
typedef GLuint64 (GLAPIENTRY *PFNGLGETIMAGEHANDLENVPROC) (GLuint  texture, GLint  level, GLboolean  layered, GLint  layer, GLenum  format);
typedef void (GLAPIENTRY *PFNGLGETINFOLOGARBPROC) (GLhandleARB  obj, GLsizei  maxLength, GLsizei * length, GLcharARB * infoLog);
typedef void (GLAPIENTRY *PFNGLGETINTEGER64I_VPROC) (GLenum  target, GLuint  index, GLint64 * data);
typedef void (GLAPIENTRY *PFNGLGETINTEGER64VPROC) (GLenum  pname, GLint64 * data);
typedef void (GLAPIENTRY *PFNGLGETINTEGERINDEXEDVEXTPROC) (GLenum  target, GLuint  index, GLint * data);
typedef void (GLAPIENTRY *PFNGLGETINTEGERI_VPROC) (GLenum  target, GLuint  index, GLint * data);
typedef void (GLAPIENTRY *PFNGLGETINTEGERUI64I_VNVPROC) (GLenum  value, GLuint  index, GLuint64EXT * result);
typedef void (GLAPIENTRY *PFNGLGETINTEGERUI64VNVPROC) (GLenum  value, GLuint64EXT * result);
typedef void (GLAPIENTRY *PFNGLGETINTEGERVPROC) (GLenum  pname, GLint * data);
typedef void (GLAPIENTRY *PFNGLGETINTERNALFORMATSAMPLEIVNVPROC) (GLenum  target, GLenum  internalformat, GLsizei  samples, GLenum  pname, GLsizei  count, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETINTERNALFORMATI64VPROC) (GLenum  target, GLenum  internalformat, GLenum  pname, GLsizei  count, GLint64 * params);
typedef void (GLAPIENTRY *PFNGLGETINTERNALFORMATIVPROC) (GLenum  target, GLenum  internalformat, GLenum  pname, GLsizei  count, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETINVARIANTBOOLEANVEXTPROC) (GLuint  id, GLenum  value, GLboolean * data);
typedef void (GLAPIENTRY *PFNGLGETINVARIANTFLOATVEXTPROC) (GLuint  id, GLenum  value, GLfloat * data);
typedef void (GLAPIENTRY *PFNGLGETINVARIANTINTEGERVEXTPROC) (GLuint  id, GLenum  value, GLint * data);
typedef void (GLAPIENTRY *PFNGLGETLIGHTFVPROC) (GLenum  light, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETLIGHTIVPROC) (GLenum  light, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETLOCALCONSTANTBOOLEANVEXTPROC) (GLuint  id, GLenum  value, GLboolean * data);
typedef void (GLAPIENTRY *PFNGLGETLOCALCONSTANTFLOATVEXTPROC) (GLuint  id, GLenum  value, GLfloat * data);
typedef void (GLAPIENTRY *PFNGLGETLOCALCONSTANTINTEGERVEXTPROC) (GLuint  id, GLenum  value, GLint * data);
typedef void (GLAPIENTRY *PFNGLGETMAPATTRIBPARAMETERFVNVPROC) (GLenum  target, GLuint  index, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMAPATTRIBPARAMETERIVNVPROC) (GLenum  target, GLuint  index, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMAPCONTROLPOINTSNVPROC) (GLenum  target, GLuint  index, GLenum  type, GLsizei  ustride, GLsizei  vstride, GLboolean  packed, void * points);
typedef void (GLAPIENTRY *PFNGLGETMAPPARAMETERFVNVPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMAPPARAMETERIVNVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMAPDVPROC) (GLenum  target, GLenum  query, GLdouble * v);
typedef void (GLAPIENTRY *PFNGLGETMAPFVPROC) (GLenum  target, GLenum  query, GLfloat * v);
typedef void (GLAPIENTRY *PFNGLGETMAPIVPROC) (GLenum  target, GLenum  query, GLint * v);
typedef void (GLAPIENTRY *PFNGLGETMATERIALFVPROC) (GLenum  face, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMATERIALIVPROC) (GLenum  face, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMEMORYOBJECTDETACHEDRESOURCESUIVNVPROC) (GLuint  memory, GLenum  pname, GLint  first, GLsizei  count, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETMEMORYOBJECTPARAMETERIVEXTPROC) (GLuint  memoryObject, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMINMAXPROC) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, void * values);
typedef void (GLAPIENTRY *PFNGLGETMINMAXEXTPROC) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, void * values);
typedef void (GLAPIENTRY *PFNGLGETMINMAXPARAMETERFVPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMINMAXPARAMETERFVEXTPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMINMAXPARAMETERIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMINMAXPARAMETERIVEXTPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXENVFVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXENVIVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXGENDVEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXGENFVEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXGENIVEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXIMAGEEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLenum  format, GLenum  type, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXLEVELPARAMETERFVEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXLEVELPARAMETERIVEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXPARAMETERIIVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXPARAMETERIUIVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXPARAMETERFVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETMULTITEXPARAMETERIVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETMULTISAMPLEFVPROC) (GLenum  pname, GLuint  index, GLfloat * val);
typedef void (GLAPIENTRY *PFNGLGETMULTISAMPLEFVNVPROC) (GLenum  pname, GLuint  index, GLfloat * val);
typedef void (GLAPIENTRY *PFNGLGETNAMEDBUFFERPARAMETERI64VPROC) (GLuint  buffer, GLenum  pname, GLint64 * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDBUFFERPARAMETERIVPROC) (GLuint  buffer, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC) (GLuint  buffer, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDBUFFERPARAMETERUI64VNVPROC) (GLuint  buffer, GLenum  pname, GLuint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDBUFFERPOINTERVPROC) (GLuint  buffer, GLenum  pname, void ** params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDBUFFERPOINTERVEXTPROC) (GLuint  buffer, GLenum  pname, void ** params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDBUFFERSUBDATAPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, void * data);
typedef void (GLAPIENTRY *PFNGLGETNAMEDBUFFERSUBDATAEXTPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, void * data);
typedef void (GLAPIENTRY *PFNGLGETNAMEDFRAMEBUFFERPARAMETERFVAMDPROC) (GLuint  framebuffer, GLenum  pname, GLuint  numsamples, GLuint  pixelindex, GLsizei  size, GLfloat * values);
typedef void (GLAPIENTRY *PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVPROC) (GLuint  framebuffer, GLenum  attachment, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC) (GLuint  framebuffer, GLenum  attachment, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVPROC) (GLuint  framebuffer, GLenum  pname, GLint * param);
typedef void (GLAPIENTRY *PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVEXTPROC) (GLuint  framebuffer, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDPROGRAMLOCALPARAMETERIIVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDPROGRAMLOCALPARAMETERIUIVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDPROGRAMLOCALPARAMETERDVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDPROGRAMLOCALPARAMETERFVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDPROGRAMSTRINGEXTPROC) (GLuint  program, GLenum  target, GLenum  pname, void * string);
typedef void (GLAPIENTRY *PFNGLGETNAMEDPROGRAMIVEXTPROC) (GLuint  program, GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDRENDERBUFFERPARAMETERIVPROC) (GLuint  renderbuffer, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDRENDERBUFFERPARAMETERIVEXTPROC) (GLuint  renderbuffer, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNAMEDSTRINGARBPROC) (GLint  namelen, const GLchar * name, GLsizei  bufSize, GLint * stringlen, GLchar * string);
typedef void (GLAPIENTRY *PFNGLGETNAMEDSTRINGIVARBPROC) (GLint  namelen, const GLchar * name, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNEXTPERFQUERYIDINTELPROC) (GLuint  queryId, GLuint * nextQueryId);
typedef void (GLAPIENTRY *PFNGLGETOBJECTLABELPROC) (GLenum  identifier, GLuint  name, GLsizei  bufSize, GLsizei * length, GLchar * label);
typedef void (GLAPIENTRY *PFNGLGETOBJECTLABELEXTPROC) (GLenum  type, GLuint  object, GLsizei  bufSize, GLsizei * length, GLchar * label);
typedef void (GLAPIENTRY *PFNGLGETOBJECTLABELKHRPROC) (GLenum  identifier, GLuint  name, GLsizei  bufSize, GLsizei * length, GLchar * label);
typedef void (GLAPIENTRY *PFNGLGETOBJECTPARAMETERFVARBPROC) (GLhandleARB  obj, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETOBJECTPARAMETERIVAPPLEPROC) (GLenum  objectType, GLuint  name, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETOBJECTPARAMETERIVARBPROC) (GLhandleARB  obj, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETOBJECTPTRLABELPROC) (const void * ptr, GLsizei  bufSize, GLsizei * length, GLchar * label);
typedef void (GLAPIENTRY *PFNGLGETOBJECTPTRLABELKHRPROC) (const void * ptr, GLsizei  bufSize, GLsizei * length, GLchar * label);
typedef void (GLAPIENTRY *PFNGLGETOCCLUSIONQUERYIVNVPROC) (GLuint  id, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETOCCLUSIONQUERYUIVNVPROC) (GLuint  id, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETPATHCOLORGENFVNVPROC) (GLenum  color, GLenum  pname, GLfloat * value);
typedef void (GLAPIENTRY *PFNGLGETPATHCOLORGENIVNVPROC) (GLenum  color, GLenum  pname, GLint * value);
typedef void (GLAPIENTRY *PFNGLGETPATHCOMMANDSNVPROC) (GLuint  path, GLubyte * commands);
typedef void (GLAPIENTRY *PFNGLGETPATHCOORDSNVPROC) (GLuint  path, GLfloat * coords);
typedef void (GLAPIENTRY *PFNGLGETPATHDASHARRAYNVPROC) (GLuint  path, GLfloat * dashArray);
typedef GLfloat (GLAPIENTRY *PFNGLGETPATHLENGTHNVPROC) (GLuint  path, GLsizei  startSegment, GLsizei  numSegments);
typedef void (GLAPIENTRY *PFNGLGETPATHMETRICRANGENVPROC) (GLbitfield  metricQueryMask, GLuint  firstPathName, GLsizei  numPaths, GLsizei  stride, GLfloat * metrics);
typedef void (GLAPIENTRY *PFNGLGETPATHMETRICSNVPROC) (GLbitfield  metricQueryMask, GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLsizei  stride, GLfloat * metrics);
typedef void (GLAPIENTRY *PFNGLGETPATHPARAMETERFVNVPROC) (GLuint  path, GLenum  pname, GLfloat * value);
typedef void (GLAPIENTRY *PFNGLGETPATHPARAMETERIVNVPROC) (GLuint  path, GLenum  pname, GLint * value);
typedef void (GLAPIENTRY *PFNGLGETPATHSPACINGNVPROC) (GLenum  pathListMode, GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLfloat  advanceScale, GLfloat  kerningScale, GLenum  transformType, GLfloat * returnedSpacing);
typedef void (GLAPIENTRY *PFNGLGETPATHTEXGENFVNVPROC) (GLenum  texCoordSet, GLenum  pname, GLfloat * value);
typedef void (GLAPIENTRY *PFNGLGETPATHTEXGENIVNVPROC) (GLenum  texCoordSet, GLenum  pname, GLint * value);
typedef void (GLAPIENTRY *PFNGLGETPERFCOUNTERINFOINTELPROC) (GLuint  queryId, GLuint  counterId, GLuint  counterNameLength, GLchar * counterName, GLuint  counterDescLength, GLchar * counterDesc, GLuint * counterOffset, GLuint * counterDataSize, GLuint * counterTypeEnum, GLuint * counterDataTypeEnum, GLuint64 * rawCounterMaxValue);
typedef void (GLAPIENTRY *PFNGLGETPERFMONITORCOUNTERDATAAMDPROC) (GLuint  monitor, GLenum  pname, GLsizei  dataSize, GLuint * data, GLint * bytesWritten);
typedef void (GLAPIENTRY *PFNGLGETPERFMONITORCOUNTERINFOAMDPROC) (GLuint  group, GLuint  counter, GLenum  pname, void * data);
typedef void (GLAPIENTRY *PFNGLGETPERFMONITORCOUNTERSTRINGAMDPROC) (GLuint  group, GLuint  counter, GLsizei  bufSize, GLsizei * length, GLchar * counterString);
typedef void (GLAPIENTRY *PFNGLGETPERFMONITORCOUNTERSAMDPROC) (GLuint  group, GLint * numCounters, GLint * maxActiveCounters, GLsizei  counterSize, GLuint * counters);
typedef void (GLAPIENTRY *PFNGLGETPERFMONITORGROUPSTRINGAMDPROC) (GLuint  group, GLsizei  bufSize, GLsizei * length, GLchar * groupString);
typedef void (GLAPIENTRY *PFNGLGETPERFMONITORGROUPSAMDPROC) (GLint * numGroups, GLsizei  groupsSize, GLuint * groups);
typedef void (GLAPIENTRY *PFNGLGETPERFQUERYDATAINTELPROC) (GLuint  queryHandle, GLuint  flags, GLsizei  dataSize, void * data, GLuint * bytesWritten);
typedef void (GLAPIENTRY *PFNGLGETPERFQUERYIDBYNAMEINTELPROC) (GLchar * queryName, GLuint * queryId);
typedef void (GLAPIENTRY *PFNGLGETPERFQUERYINFOINTELPROC) (GLuint  queryId, GLuint  queryNameLength, GLchar * queryName, GLuint * dataSize, GLuint * noCounters, GLuint * noInstances, GLuint * capsMask);
typedef void (GLAPIENTRY *PFNGLGETPIXELMAPFVPROC) (GLenum  map, GLfloat * values);
typedef void (GLAPIENTRY *PFNGLGETPIXELMAPUIVPROC) (GLenum  map, GLuint * values);
typedef void (GLAPIENTRY *PFNGLGETPIXELMAPUSVPROC) (GLenum  map, GLushort * values);
typedef void (GLAPIENTRY *PFNGLGETPIXELTRANSFORMPARAMETERFVEXTPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETPIXELTRANSFORMPARAMETERIVEXTPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETPOINTERINDEXEDVEXTPROC) (GLenum  target, GLuint  index, void ** data);
typedef void (GLAPIENTRY *PFNGLGETPOINTERI_VEXTPROC) (GLenum  pname, GLuint  index, void ** params);
typedef void (GLAPIENTRY *PFNGLGETPOINTERVPROC) (GLenum  pname, void ** params);
typedef void (GLAPIENTRY *PFNGLGETPOINTERVEXTPROC) (GLenum  pname, void ** params);
typedef void (GLAPIENTRY *PFNGLGETPOINTERVKHRPROC) (GLenum  pname, void ** params);
typedef void (GLAPIENTRY *PFNGLGETPOLYGONSTIPPLEPROC) (GLubyte * mask);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMBINARYPROC) (GLuint  program, GLsizei  bufSize, GLsizei * length, GLenum * binaryFormat, void * binary);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMENVPARAMETERIIVNVPROC) (GLenum  target, GLuint  index, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMENVPARAMETERIUIVNVPROC) (GLenum  target, GLuint  index, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMENVPARAMETERDVARBPROC) (GLenum  target, GLuint  index, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMENVPARAMETERFVARBPROC) (GLenum  target, GLuint  index, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMINFOLOGPROC) (GLuint  program, GLsizei  bufSize, GLsizei * length, GLchar * infoLog);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMINTERFACEIVPROC) (GLuint  program, GLenum  programInterface, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMLOCALPARAMETERIIVNVPROC) (GLenum  target, GLuint  index, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMLOCALPARAMETERIUIVNVPROC) (GLenum  target, GLuint  index, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC) (GLenum  target, GLuint  index, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC) (GLenum  target, GLuint  index, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMNAMEDPARAMETERDVNVPROC) (GLuint  id, GLsizei  len, const GLubyte * name, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMNAMEDPARAMETERFVNVPROC) (GLuint  id, GLsizei  len, const GLubyte * name, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMPARAMETERDVNVPROC) (GLenum  target, GLuint  index, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMPARAMETERFVNVPROC) (GLenum  target, GLuint  index, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMPIPELINEINFOLOGPROC) (GLuint  pipeline, GLsizei  bufSize, GLsizei * length, GLchar * infoLog);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMPIPELINEINFOLOGEXTPROC) (GLuint  pipeline, GLsizei  bufSize, GLsizei * length, GLchar * infoLog);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMPIPELINEIVPROC) (GLuint  pipeline, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMPIPELINEIVEXTPROC) (GLuint  pipeline, GLenum  pname, GLint * params);
typedef GLuint (GLAPIENTRY *PFNGLGETPROGRAMRESOURCEINDEXPROC) (GLuint  program, GLenum  programInterface, const GLchar * name);
typedef GLint (GLAPIENTRY *PFNGLGETPROGRAMRESOURCELOCATIONPROC) (GLuint  program, GLenum  programInterface, const GLchar * name);
typedef GLint (GLAPIENTRY *PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC) (GLuint  program, GLenum  programInterface, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMRESOURCENAMEPROC) (GLuint  program, GLenum  programInterface, GLuint  index, GLsizei  bufSize, GLsizei * length, GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMRESOURCEFVNVPROC) (GLuint  program, GLenum  programInterface, GLuint  index, GLsizei  propCount, const GLenum * props, GLsizei  count, GLsizei * length, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMRESOURCEIVPROC) (GLuint  program, GLenum  programInterface, GLuint  index, GLsizei  propCount, const GLenum * props, GLsizei  count, GLsizei * length, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMSTAGEIVPROC) (GLuint  program, GLenum  shadertype, GLenum  pname, GLint * values);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMSTRINGARBPROC) (GLenum  target, GLenum  pname, void * string);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMSTRINGNVPROC) (GLuint  id, GLenum  pname, GLubyte * program);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMSUBROUTINEPARAMETERUIVNVPROC) (GLenum  target, GLuint  index, GLuint * param);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMIVPROC) (GLuint  program, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMIVARBPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETPROGRAMIVNVPROC) (GLuint  id, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYBUFFEROBJECTI64VPROC) (GLuint  id, GLuint  buffer, GLenum  pname, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLGETQUERYBUFFEROBJECTIVPROC) (GLuint  id, GLuint  buffer, GLenum  pname, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLGETQUERYBUFFEROBJECTUI64VPROC) (GLuint  id, GLuint  buffer, GLenum  pname, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLGETQUERYBUFFEROBJECTUIVPROC) (GLuint  id, GLuint  buffer, GLenum  pname, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLGETQUERYINDEXEDIVPROC) (GLenum  target, GLuint  index, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYOBJECTI64VPROC) (GLuint  id, GLenum  pname, GLint64 * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYOBJECTI64VEXTPROC) (GLuint  id, GLenum  pname, GLint64 * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYOBJECTIVPROC) (GLuint  id, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYOBJECTIVARBPROC) (GLuint  id, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYOBJECTUI64VPROC) (GLuint  id, GLenum  pname, GLuint64 * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYOBJECTUI64VEXTPROC) (GLuint  id, GLenum  pname, GLuint64 * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYOBJECTUIVPROC) (GLuint  id, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYOBJECTUIVARBPROC) (GLuint  id, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETQUERYIVARBPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETRENDERBUFFERPARAMETERIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETSAMPLERPARAMETERIIVPROC) (GLuint  sampler, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETSAMPLERPARAMETERIUIVPROC) (GLuint  sampler, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETSAMPLERPARAMETERFVPROC) (GLuint  sampler, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETSAMPLERPARAMETERIVPROC) (GLuint  sampler, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETSEMAPHOREPARAMETERUI64VEXTPROC) (GLuint  semaphore, GLenum  pname, GLuint64 * params);
typedef void (GLAPIENTRY *PFNGLGETSEPARABLEFILTERPROC) (GLenum  target, GLenum  format, GLenum  type, void * row, void * column, void * span);
typedef void (GLAPIENTRY *PFNGLGETSEPARABLEFILTEREXTPROC) (GLenum  target, GLenum  format, GLenum  type, void * row, void * column, void * span);
typedef void (GLAPIENTRY *PFNGLGETSHADERINFOLOGPROC) (GLuint  shader, GLsizei  bufSize, GLsizei * length, GLchar * infoLog);
typedef void (GLAPIENTRY *PFNGLGETSHADERPRECISIONFORMATPROC) (GLenum  shadertype, GLenum  precisiontype, GLint * range, GLint * precision);
typedef void (GLAPIENTRY *PFNGLGETSHADERSOURCEPROC) (GLuint  shader, GLsizei  bufSize, GLsizei * length, GLchar * source);
typedef void (GLAPIENTRY *PFNGLGETSHADERSOURCEARBPROC) (GLhandleARB  obj, GLsizei  maxLength, GLsizei * length, GLcharARB * source);
typedef void (GLAPIENTRY *PFNGLGETSHADERIVPROC) (GLuint  shader, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETSHADINGRATEIMAGEPALETTENVPROC) (GLuint  viewport, GLuint  entry, GLenum * rate);
typedef void (GLAPIENTRY *PFNGLGETSHADINGRATESAMPLELOCATIONIVNVPROC) (GLenum  rate, GLuint  samples, GLuint  index, GLint * location);
typedef GLushort (GLAPIENTRY *PFNGLGETSTAGEINDEXNVPROC) (GLenum  shadertype);
typedef const GLubyte *(GLAPIENTRY *PFNGLGETSTRINGPROC) (GLenum  name);
typedef const GLubyte *(GLAPIENTRY *PFNGLGETSTRINGIPROC) (GLenum  name, GLuint  index);
typedef GLuint (GLAPIENTRY *PFNGLGETSUBROUTINEINDEXPROC) (GLuint  program, GLenum  shadertype, const GLchar * name);
typedef GLint (GLAPIENTRY *PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC) (GLuint  program, GLenum  shadertype, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETSYNCIVPROC) (GLsync  sync, GLenum  pname, GLsizei  count, GLsizei * length, GLint * values);
typedef void (GLAPIENTRY *PFNGLGETTEXENVFVPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETTEXENVIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXGENDVPROC) (GLenum  coord, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETTEXGENFVPROC) (GLenum  coord, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETTEXGENIVPROC) (GLenum  coord, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXIMAGEPROC) (GLenum  target, GLint  level, GLenum  format, GLenum  type, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETTEXLEVELPARAMETERFVPROC) (GLenum  target, GLint  level, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETTEXLEVELPARAMETERIVPROC) (GLenum  target, GLint  level, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXPARAMETERIIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXPARAMETERIIVEXTPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXPARAMETERIUIVPROC) (GLenum  target, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXPARAMETERIUIVEXTPROC) (GLenum  target, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXPARAMETERPOINTERVAPPLEPROC) (GLenum  target, GLenum  pname, void ** params);
typedef void (GLAPIENTRY *PFNGLGETTEXPARAMETERFVPROC) (GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETTEXPARAMETERIVPROC) (GLenum  target, GLenum  pname, GLint * params);
typedef GLuint64 (GLAPIENTRY *PFNGLGETTEXTUREHANDLEARBPROC) (GLuint  texture);
typedef GLuint64 (GLAPIENTRY *PFNGLGETTEXTUREHANDLENVPROC) (GLuint  texture);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREIMAGEPROC) (GLuint  texture, GLint  level, GLenum  format, GLenum  type, GLsizei  bufSize, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREIMAGEEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLenum  format, GLenum  type, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETTEXTURELEVELPARAMETERFVPROC) (GLuint  texture, GLint  level, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTURELEVELPARAMETERFVEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTURELEVELPARAMETERIVPROC) (GLuint  texture, GLint  level, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTURELEVELPARAMETERIVEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREPARAMETERIIVPROC) (GLuint  texture, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREPARAMETERIIVEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREPARAMETERIUIVPROC) (GLuint  texture, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREPARAMETERIUIVEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREPARAMETERFVPROC) (GLuint  texture, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREPARAMETERFVEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREPARAMETERIVPROC) (GLuint  texture, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTEXTUREPARAMETERIVEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, GLint * params);
typedef GLuint64 (GLAPIENTRY *PFNGLGETTEXTURESAMPLERHANDLEARBPROC) (GLuint  texture, GLuint  sampler);
typedef GLuint64 (GLAPIENTRY *PFNGLGETTEXTURESAMPLERHANDLENVPROC) (GLuint  texture, GLuint  sampler);
typedef void (GLAPIENTRY *PFNGLGETTEXTURESUBIMAGEPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, GLsizei  bufSize, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETTRACKMATRIXIVNVPROC) (GLenum  target, GLuint  address, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETTRANSFORMFEEDBACKVARYINGPROC) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETTRANSFORMFEEDBACKVARYINGEXTPROC) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETTRANSFORMFEEDBACKVARYINGNVPROC) (GLuint  program, GLuint  index, GLint * location);
typedef void (GLAPIENTRY *PFNGLGETTRANSFORMFEEDBACKI64_VPROC) (GLuint  xfb, GLenum  pname, GLuint  index, GLint64 * param);
typedef void (GLAPIENTRY *PFNGLGETTRANSFORMFEEDBACKI_VPROC) (GLuint  xfb, GLenum  pname, GLuint  index, GLint * param);
typedef void (GLAPIENTRY *PFNGLGETTRANSFORMFEEDBACKIVPROC) (GLuint  xfb, GLenum  pname, GLint * param);
typedef GLuint (GLAPIENTRY *PFNGLGETUNIFORMBLOCKINDEXPROC) (GLuint  program, const GLchar * uniformBlockName);
typedef GLint (GLAPIENTRY *PFNGLGETUNIFORMBUFFERSIZEEXTPROC) (GLuint  program, GLint  location);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMINDICESPROC) (GLuint  program, GLsizei  uniformCount, const GLchar *const* uniformNames, GLuint * uniformIndices);
typedef GLint (GLAPIENTRY *PFNGLGETUNIFORMLOCATIONPROC) (GLuint  program, const GLchar * name);
typedef GLint (GLAPIENTRY *PFNGLGETUNIFORMLOCATIONARBPROC) (GLhandleARB  programObj, const GLcharARB * name);
typedef GLintptr (GLAPIENTRY *PFNGLGETUNIFORMOFFSETEXTPROC) (GLuint  program, GLint  location);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMSUBROUTINEUIVPROC) (GLenum  shadertype, GLint  location, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMDVPROC) (GLuint  program, GLint  location, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMFVPROC) (GLuint  program, GLint  location, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMFVARBPROC) (GLhandleARB  programObj, GLint  location, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMI64VARBPROC) (GLuint  program, GLint  location, GLint64 * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMI64VNVPROC) (GLuint  program, GLint  location, GLint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMIVPROC) (GLuint  program, GLint  location, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMIVARBPROC) (GLhandleARB  programObj, GLint  location, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMUI64VARBPROC) (GLuint  program, GLint  location, GLuint64 * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMUI64VNVPROC) (GLuint  program, GLint  location, GLuint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMUIVPROC) (GLuint  program, GLint  location, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETUNIFORMUIVEXTPROC) (GLuint  program, GLint  location, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETUNSIGNEDBYTEVEXTPROC) (GLenum  pname, GLubyte * data);
typedef void (GLAPIENTRY *PFNGLGETUNSIGNEDBYTEI_VEXTPROC) (GLenum  target, GLuint  index, GLubyte * data);
typedef void (GLAPIENTRY *PFNGLGETVARIANTBOOLEANVEXTPROC) (GLuint  id, GLenum  value, GLboolean * data);
typedef void (GLAPIENTRY *PFNGLGETVARIANTFLOATVEXTPROC) (GLuint  id, GLenum  value, GLfloat * data);
typedef void (GLAPIENTRY *PFNGLGETVARIANTINTEGERVEXTPROC) (GLuint  id, GLenum  value, GLint * data);
typedef void (GLAPIENTRY *PFNGLGETVARIANTPOINTERVEXTPROC) (GLuint  id, GLenum  value, void ** data);
typedef GLint (GLAPIENTRY *PFNGLGETVARYINGLOCATIONNVPROC) (GLuint  program, const GLchar * name);
typedef void (GLAPIENTRY *PFNGLGETVERTEXARRAYINDEXED64IVPROC) (GLuint  vaobj, GLuint  index, GLenum  pname, GLint64 * param);
typedef void (GLAPIENTRY *PFNGLGETVERTEXARRAYINDEXEDIVPROC) (GLuint  vaobj, GLuint  index, GLenum  pname, GLint * param);
typedef void (GLAPIENTRY *PFNGLGETVERTEXARRAYINTEGERI_VEXTPROC) (GLuint  vaobj, GLuint  index, GLenum  pname, GLint * param);
typedef void (GLAPIENTRY *PFNGLGETVERTEXARRAYINTEGERVEXTPROC) (GLuint  vaobj, GLenum  pname, GLint * param);
typedef void (GLAPIENTRY *PFNGLGETVERTEXARRAYPOINTERI_VEXTPROC) (GLuint  vaobj, GLuint  index, GLenum  pname, void ** param);
typedef void (GLAPIENTRY *PFNGLGETVERTEXARRAYPOINTERVEXTPROC) (GLuint  vaobj, GLenum  pname, void ** param);
typedef void (GLAPIENTRY *PFNGLGETVERTEXARRAYIVPROC) (GLuint  vaobj, GLenum  pname, GLint * param);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBIIVPROC) (GLuint  index, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBIIVEXTPROC) (GLuint  index, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBIUIVPROC) (GLuint  index, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBIUIVEXTPROC) (GLuint  index, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBLDVPROC) (GLuint  index, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBLDVEXTPROC) (GLuint  index, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBLI64VNVPROC) (GLuint  index, GLenum  pname, GLint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBLUI64VARBPROC) (GLuint  index, GLenum  pname, GLuint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBLUI64VNVPROC) (GLuint  index, GLenum  pname, GLuint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBPOINTERVPROC) (GLuint  index, GLenum  pname, void ** pointer);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBPOINTERVARBPROC) (GLuint  index, GLenum  pname, void ** pointer);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBPOINTERVNVPROC) (GLuint  index, GLenum  pname, void ** pointer);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBDVPROC) (GLuint  index, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBDVARBPROC) (GLuint  index, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBDVNVPROC) (GLuint  index, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBFVPROC) (GLuint  index, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBFVARBPROC) (GLuint  index, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBFVNVPROC) (GLuint  index, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBIVPROC) (GLuint  index, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBIVARBPROC) (GLuint  index, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETVERTEXATTRIBIVNVPROC) (GLuint  index, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETVIDEOCAPTURESTREAMDVNVPROC) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETVIDEOCAPTURESTREAMFVNVPROC) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETVIDEOCAPTURESTREAMIVNVPROC) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETVIDEOCAPTUREIVNVPROC) (GLuint  video_capture_slot, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETVIDEOI64VNVPROC) (GLuint  video_slot, GLenum  pname, GLint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETVIDEOIVNVPROC) (GLuint  video_slot, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETVIDEOUI64VNVPROC) (GLuint  video_slot, GLenum  pname, GLuint64EXT * params);
typedef void (GLAPIENTRY *PFNGLGETVIDEOUIVNVPROC) (GLuint  video_slot, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETNCOLORTABLEPROC) (GLenum  target, GLenum  format, GLenum  type, GLsizei  bufSize, void * table);
typedef void (GLAPIENTRY *PFNGLGETNCOLORTABLEARBPROC) (GLenum  target, GLenum  format, GLenum  type, GLsizei  bufSize, void * table);
typedef void (GLAPIENTRY *PFNGLGETNCOMPRESSEDTEXIMAGEPROC) (GLenum  target, GLint  lod, GLsizei  bufSize, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETNCOMPRESSEDTEXIMAGEARBPROC) (GLenum  target, GLint  lod, GLsizei  bufSize, void * img);
typedef void (GLAPIENTRY *PFNGLGETNCONVOLUTIONFILTERPROC) (GLenum  target, GLenum  format, GLenum  type, GLsizei  bufSize, void * image);
typedef void (GLAPIENTRY *PFNGLGETNCONVOLUTIONFILTERARBPROC) (GLenum  target, GLenum  format, GLenum  type, GLsizei  bufSize, void * image);
typedef void (GLAPIENTRY *PFNGLGETNHISTOGRAMPROC) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, GLsizei  bufSize, void * values);
typedef void (GLAPIENTRY *PFNGLGETNHISTOGRAMARBPROC) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, GLsizei  bufSize, void * values);
typedef void (GLAPIENTRY *PFNGLGETNMAPDVPROC) (GLenum  target, GLenum  query, GLsizei  bufSize, GLdouble * v);
typedef void (GLAPIENTRY *PFNGLGETNMAPDVARBPROC) (GLenum  target, GLenum  query, GLsizei  bufSize, GLdouble * v);
typedef void (GLAPIENTRY *PFNGLGETNMAPFVPROC) (GLenum  target, GLenum  query, GLsizei  bufSize, GLfloat * v);
typedef void (GLAPIENTRY *PFNGLGETNMAPFVARBPROC) (GLenum  target, GLenum  query, GLsizei  bufSize, GLfloat * v);
typedef void (GLAPIENTRY *PFNGLGETNMAPIVPROC) (GLenum  target, GLenum  query, GLsizei  bufSize, GLint * v);
typedef void (GLAPIENTRY *PFNGLGETNMAPIVARBPROC) (GLenum  target, GLenum  query, GLsizei  bufSize, GLint * v);
typedef void (GLAPIENTRY *PFNGLGETNMINMAXPROC) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, GLsizei  bufSize, void * values);
typedef void (GLAPIENTRY *PFNGLGETNMINMAXARBPROC) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, GLsizei  bufSize, void * values);
typedef void (GLAPIENTRY *PFNGLGETNPIXELMAPFVPROC) (GLenum  map, GLsizei  bufSize, GLfloat * values);
typedef void (GLAPIENTRY *PFNGLGETNPIXELMAPFVARBPROC) (GLenum  map, GLsizei  bufSize, GLfloat * values);
typedef void (GLAPIENTRY *PFNGLGETNPIXELMAPUIVPROC) (GLenum  map, GLsizei  bufSize, GLuint * values);
typedef void (GLAPIENTRY *PFNGLGETNPIXELMAPUIVARBPROC) (GLenum  map, GLsizei  bufSize, GLuint * values);
typedef void (GLAPIENTRY *PFNGLGETNPIXELMAPUSVPROC) (GLenum  map, GLsizei  bufSize, GLushort * values);
typedef void (GLAPIENTRY *PFNGLGETNPIXELMAPUSVARBPROC) (GLenum  map, GLsizei  bufSize, GLushort * values);
typedef void (GLAPIENTRY *PFNGLGETNPOLYGONSTIPPLEPROC) (GLsizei  bufSize, GLubyte * pattern);
typedef void (GLAPIENTRY *PFNGLGETNPOLYGONSTIPPLEARBPROC) (GLsizei  bufSize, GLubyte * pattern);
typedef void (GLAPIENTRY *PFNGLGETNSEPARABLEFILTERPROC) (GLenum  target, GLenum  format, GLenum  type, GLsizei  rowBufSize, void * row, GLsizei  columnBufSize, void * column, void * span);
typedef void (GLAPIENTRY *PFNGLGETNSEPARABLEFILTERARBPROC) (GLenum  target, GLenum  format, GLenum  type, GLsizei  rowBufSize, void * row, GLsizei  columnBufSize, void * column, void * span);
typedef void (GLAPIENTRY *PFNGLGETNTEXIMAGEPROC) (GLenum  target, GLint  level, GLenum  format, GLenum  type, GLsizei  bufSize, void * pixels);
typedef void (GLAPIENTRY *PFNGLGETNTEXIMAGEARBPROC) (GLenum  target, GLint  level, GLenum  format, GLenum  type, GLsizei  bufSize, void * img);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMDVPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMDVARBPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLdouble * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMFVPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMFVARBPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMFVKHRPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLfloat * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMI64VARBPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLint64 * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMIVPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMIVARBPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMIVKHRPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLint * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMUI64VARBPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLuint64 * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMUIVPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMUIVARBPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLuint * params);
typedef void (GLAPIENTRY *PFNGLGETNUNIFORMUIVKHRPROC) (GLuint  program, GLint  location, GLsizei  bufSize, GLuint * params);
typedef void (GLAPIENTRY *PFNGLHINTPROC) (GLenum  target, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLHISTOGRAMPROC) (GLenum  target, GLsizei  width, GLenum  internalformat, GLboolean  sink);
typedef void (GLAPIENTRY *PFNGLHISTOGRAMEXTPROC) (GLenum  target, GLsizei  width, GLenum  internalformat, GLboolean  sink);
typedef void (GLAPIENTRY *PFNGLIMPORTMEMORYFDEXTPROC) (GLuint  memory, GLuint64  size, GLenum  handleType, GLint  fd);
typedef void (GLAPIENTRY *PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC) (GLuint  memory, GLuint64  size, GLenum  handleType, void * handle);
typedef void (GLAPIENTRY *PFNGLIMPORTMEMORYWIN32NAMEEXTPROC) (GLuint  memory, GLuint64  size, GLenum  handleType, const void * name);
typedef void (GLAPIENTRY *PFNGLIMPORTSEMAPHOREFDEXTPROC) (GLuint  semaphore, GLenum  handleType, GLint  fd);
typedef void (GLAPIENTRY *PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC) (GLuint  semaphore, GLenum  handleType, void * handle);
typedef void (GLAPIENTRY *PFNGLIMPORTSEMAPHOREWIN32NAMEEXTPROC) (GLuint  semaphore, GLenum  handleType, const void * name);
typedef GLsync (GLAPIENTRY *PFNGLIMPORTSYNCEXTPROC) (GLenum  external_sync_type, GLintptr  external_sync, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLINDEXFORMATNVPROC) (GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLINDEXFUNCEXTPROC) (GLenum  func, GLclampf  ref);
typedef void (GLAPIENTRY *PFNGLINDEXMASKPROC) (GLuint  mask);
typedef void (GLAPIENTRY *PFNGLINDEXMATERIALEXTPROC) (GLenum  face, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLINDEXPOINTERPROC) (GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLINDEXPOINTEREXTPROC) (GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
typedef void (GLAPIENTRY *PFNGLINDEXDPROC) (GLdouble  c);
typedef void (GLAPIENTRY *PFNGLINDEXDVPROC) (const GLdouble * c);
typedef void (GLAPIENTRY *PFNGLINDEXFPROC) (GLfloat  c);
typedef void (GLAPIENTRY *PFNGLINDEXFVPROC) (const GLfloat * c);
typedef void (GLAPIENTRY *PFNGLINDEXIPROC) (GLint  c);
typedef void (GLAPIENTRY *PFNGLINDEXIVPROC) (const GLint * c);
typedef void (GLAPIENTRY *PFNGLINDEXSPROC) (GLshort  c);
typedef void (GLAPIENTRY *PFNGLINDEXSVPROC) (const GLshort * c);
typedef void (GLAPIENTRY *PFNGLINDEXUBPROC) (GLubyte  c);
typedef void (GLAPIENTRY *PFNGLINDEXUBVPROC) (const GLubyte * c);
typedef void (GLAPIENTRY *PFNGLINITNAMESPROC) ();
typedef void (GLAPIENTRY *PFNGLINSERTCOMPONENTEXTPROC) (GLuint  res, GLuint  src, GLuint  num);
typedef void (GLAPIENTRY *PFNGLINSERTEVENTMARKEREXTPROC) (GLsizei  length, const GLchar * marker);
typedef void (GLAPIENTRY *PFNGLINTERLEAVEDARRAYSPROC) (GLenum  format, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLINTERPOLATEPATHSNVPROC) (GLuint  resultPath, GLuint  pathA, GLuint  pathB, GLfloat  weight);
typedef void (GLAPIENTRY *PFNGLINVALIDATEBUFFERDATAPROC) (GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLINVALIDATEBUFFERSUBDATAPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length);
typedef void (GLAPIENTRY *PFNGLINVALIDATEFRAMEBUFFERPROC) (GLenum  target, GLsizei  numAttachments, const GLenum * attachments);
typedef void (GLAPIENTRY *PFNGLINVALIDATENAMEDFRAMEBUFFERDATAPROC) (GLuint  framebuffer, GLsizei  numAttachments, const GLenum * attachments);
typedef void (GLAPIENTRY *PFNGLINVALIDATENAMEDFRAMEBUFFERSUBDATAPROC) (GLuint  framebuffer, GLsizei  numAttachments, const GLenum * attachments, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLINVALIDATESUBFRAMEBUFFERPROC) (GLenum  target, GLsizei  numAttachments, const GLenum * attachments, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLINVALIDATETEXIMAGEPROC) (GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLINVALIDATETEXSUBIMAGEPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth);
typedef GLboolean (GLAPIENTRY *PFNGLISBUFFERPROC) (GLuint  buffer);
typedef GLboolean (GLAPIENTRY *PFNGLISBUFFERARBPROC) (GLuint  buffer);
typedef GLboolean (GLAPIENTRY *PFNGLISBUFFERRESIDENTNVPROC) (GLenum  target);
typedef GLboolean (GLAPIENTRY *PFNGLISCOMMANDLISTNVPROC) (GLuint  list);
typedef GLboolean (GLAPIENTRY *PFNGLISENABLEDPROC) (GLenum  cap);
typedef GLboolean (GLAPIENTRY *PFNGLISENABLEDINDEXEDEXTPROC) (GLenum  target, GLuint  index);
typedef GLboolean (GLAPIENTRY *PFNGLISENABLEDIPROC) (GLenum  target, GLuint  index);
typedef GLboolean (GLAPIENTRY *PFNGLISFENCEAPPLEPROC) (GLuint  fence);
typedef GLboolean (GLAPIENTRY *PFNGLISFENCENVPROC) (GLuint  fence);
typedef GLboolean (GLAPIENTRY *PFNGLISFRAMEBUFFERPROC) (GLuint  framebuffer);
typedef GLboolean (GLAPIENTRY *PFNGLISFRAMEBUFFEREXTPROC) (GLuint  framebuffer);
typedef GLboolean (GLAPIENTRY *PFNGLISIMAGEHANDLERESIDENTARBPROC) (GLuint64  handle);
typedef GLboolean (GLAPIENTRY *PFNGLISIMAGEHANDLERESIDENTNVPROC) (GLuint64  handle);
typedef GLboolean (GLAPIENTRY *PFNGLISLISTPROC) (GLuint  list);
typedef GLboolean (GLAPIENTRY *PFNGLISMEMORYOBJECTEXTPROC) (GLuint  memoryObject);
typedef GLboolean (GLAPIENTRY *PFNGLISNAMEAMDPROC) (GLenum  identifier, GLuint  name);
typedef GLboolean (GLAPIENTRY *PFNGLISNAMEDBUFFERRESIDENTNVPROC) (GLuint  buffer);
typedef GLboolean (GLAPIENTRY *PFNGLISNAMEDSTRINGARBPROC) (GLint  namelen, const GLchar * name);
typedef GLboolean (GLAPIENTRY *PFNGLISOCCLUSIONQUERYNVPROC) (GLuint  id);
typedef GLboolean (GLAPIENTRY *PFNGLISPATHNVPROC) (GLuint  path);
typedef GLboolean (GLAPIENTRY *PFNGLISPOINTINFILLPATHNVPROC) (GLuint  path, GLuint  mask, GLfloat  x, GLfloat  y);
typedef GLboolean (GLAPIENTRY *PFNGLISPOINTINSTROKEPATHNVPROC) (GLuint  path, GLfloat  x, GLfloat  y);
typedef GLboolean (GLAPIENTRY *PFNGLISPROGRAMPROC) (GLuint  program);
typedef GLboolean (GLAPIENTRY *PFNGLISPROGRAMARBPROC) (GLuint  program);
typedef GLboolean (GLAPIENTRY *PFNGLISPROGRAMNVPROC) (GLuint  id);
typedef GLboolean (GLAPIENTRY *PFNGLISPROGRAMPIPELINEPROC) (GLuint  pipeline);
typedef GLboolean (GLAPIENTRY *PFNGLISPROGRAMPIPELINEEXTPROC) (GLuint  pipeline);
typedef GLboolean (GLAPIENTRY *PFNGLISQUERYPROC) (GLuint  id);
typedef GLboolean (GLAPIENTRY *PFNGLISQUERYARBPROC) (GLuint  id);
typedef GLboolean (GLAPIENTRY *PFNGLISRENDERBUFFERPROC) (GLuint  renderbuffer);
typedef GLboolean (GLAPIENTRY *PFNGLISRENDERBUFFEREXTPROC) (GLuint  renderbuffer);
typedef GLboolean (GLAPIENTRY *PFNGLISSEMAPHOREEXTPROC) (GLuint  semaphore);
typedef GLboolean (GLAPIENTRY *PFNGLISSAMPLERPROC) (GLuint  sampler);
typedef GLboolean (GLAPIENTRY *PFNGLISSHADERPROC) (GLuint  shader);
typedef GLboolean (GLAPIENTRY *PFNGLISSTATENVPROC) (GLuint  state);
typedef GLboolean (GLAPIENTRY *PFNGLISSYNCPROC) (GLsync  sync);
typedef GLboolean (GLAPIENTRY *PFNGLISTEXTUREPROC) (GLuint  texture);
typedef GLboolean (GLAPIENTRY *PFNGLISTEXTUREEXTPROC) (GLuint  texture);
typedef GLboolean (GLAPIENTRY *PFNGLISTEXTUREHANDLERESIDENTARBPROC) (GLuint64  handle);
typedef GLboolean (GLAPIENTRY *PFNGLISTEXTUREHANDLERESIDENTNVPROC) (GLuint64  handle);
typedef GLboolean (GLAPIENTRY *PFNGLISTRANSFORMFEEDBACKPROC) (GLuint  id);
typedef GLboolean (GLAPIENTRY *PFNGLISTRANSFORMFEEDBACKNVPROC) (GLuint  id);
typedef GLboolean (GLAPIENTRY *PFNGLISVARIANTENABLEDEXTPROC) (GLuint  id, GLenum  cap);
typedef GLboolean (GLAPIENTRY *PFNGLISVERTEXARRAYPROC) (GLuint  array);
typedef GLboolean (GLAPIENTRY *PFNGLISVERTEXARRAYAPPLEPROC) (GLuint  array);
typedef GLboolean (GLAPIENTRY *PFNGLISVERTEXATTRIBENABLEDAPPLEPROC) (GLuint  index, GLenum  pname);
typedef void (GLAPIENTRY *PFNGLLABELOBJECTEXTPROC) (GLenum  type, GLuint  object, GLsizei  length, const GLchar * label);
typedef void (GLAPIENTRY *PFNGLLIGHTMODELFPROC) (GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLLIGHTMODELFVPROC) (GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLLIGHTMODELIPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLLIGHTMODELIVPROC) (GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLLIGHTFPROC) (GLenum  light, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLLIGHTFVPROC) (GLenum  light, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLLIGHTIPROC) (GLenum  light, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLLIGHTIVPROC) (GLenum  light, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLLINESTIPPLEPROC) (GLint  factor, GLushort  pattern);
typedef void (GLAPIENTRY *PFNGLLINEWIDTHPROC) (GLfloat  width);
typedef void (GLAPIENTRY *PFNGLLINKPROGRAMPROC) (GLuint  program);
typedef void (GLAPIENTRY *PFNGLLINKPROGRAMARBPROC) (GLhandleARB  programObj);
typedef void (GLAPIENTRY *PFNGLLISTBASEPROC) (GLuint  base);
typedef void (GLAPIENTRY *PFNGLLISTDRAWCOMMANDSSTATESCLIENTNVPROC) (GLuint  list, GLuint  segment, const void ** indirects, const GLsizei * sizes, const GLuint * states, const GLuint * fbos, GLuint  count);
typedef void (GLAPIENTRY *PFNGLLOADIDENTITYPROC) ();
typedef void (GLAPIENTRY *PFNGLLOADMATRIXDPROC) (const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLLOADMATRIXFPROC) (const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLLOADNAMEPROC) (GLuint  name);
typedef void (GLAPIENTRY *PFNGLLOADPROGRAMNVPROC) (GLenum  target, GLuint  id, GLsizei  len, const GLubyte * program);
typedef void (GLAPIENTRY *PFNGLLOADTRANSPOSEMATRIXDPROC) (const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLLOADTRANSPOSEMATRIXDARBPROC) (const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLLOADTRANSPOSEMATRIXFPROC) (const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLLOADTRANSPOSEMATRIXFARBPROC) (const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLLOCKARRAYSEXTPROC) (GLint  first, GLsizei  count);
typedef void (GLAPIENTRY *PFNGLLOGICOPPROC) (GLenum  opcode);
typedef void (GLAPIENTRY *PFNGLMAKEBUFFERNONRESIDENTNVPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLMAKEBUFFERRESIDENTNVPROC) (GLenum  target, GLenum  access);
typedef void (GLAPIENTRY *PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC) (GLuint64  handle);
typedef void (GLAPIENTRY *PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC) (GLuint64  handle);
typedef void (GLAPIENTRY *PFNGLMAKEIMAGEHANDLERESIDENTARBPROC) (GLuint64  handle, GLenum  access);
typedef void (GLAPIENTRY *PFNGLMAKEIMAGEHANDLERESIDENTNVPROC) (GLuint64  handle, GLenum  access);
typedef void (GLAPIENTRY *PFNGLMAKENAMEDBUFFERNONRESIDENTNVPROC) (GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLMAKENAMEDBUFFERRESIDENTNVPROC) (GLuint  buffer, GLenum  access);
typedef void (GLAPIENTRY *PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC) (GLuint64  handle);
typedef void (GLAPIENTRY *PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC) (GLuint64  handle);
typedef void (GLAPIENTRY *PFNGLMAKETEXTUREHANDLERESIDENTARBPROC) (GLuint64  handle);
typedef void (GLAPIENTRY *PFNGLMAKETEXTUREHANDLERESIDENTNVPROC) (GLuint64  handle);
typedef void (GLAPIENTRY *PFNGLMAP1DPROC) (GLenum  target, GLdouble  u1, GLdouble  u2, GLint  stride, GLint  order, const GLdouble * points);
typedef void (GLAPIENTRY *PFNGLMAP1FPROC) (GLenum  target, GLfloat  u1, GLfloat  u2, GLint  stride, GLint  order, const GLfloat * points);
typedef void (GLAPIENTRY *PFNGLMAP2DPROC) (GLenum  target, GLdouble  u1, GLdouble  u2, GLint  ustride, GLint  uorder, GLdouble  v1, GLdouble  v2, GLint  vstride, GLint  vorder, const GLdouble * points);
typedef void (GLAPIENTRY *PFNGLMAP2FPROC) (GLenum  target, GLfloat  u1, GLfloat  u2, GLint  ustride, GLint  uorder, GLfloat  v1, GLfloat  v2, GLint  vstride, GLint  vorder, const GLfloat * points);
typedef void *(GLAPIENTRY *PFNGLMAPBUFFERPROC) (GLenum  target, GLenum  access);
typedef void *(GLAPIENTRY *PFNGLMAPBUFFERARBPROC) (GLenum  target, GLenum  access);
typedef void *(GLAPIENTRY *PFNGLMAPBUFFERRANGEPROC) (GLenum  target, GLintptr  offset, GLsizeiptr  length, GLbitfield  access);
typedef void (GLAPIENTRY *PFNGLMAPCONTROLPOINTSNVPROC) (GLenum  target, GLuint  index, GLenum  type, GLsizei  ustride, GLsizei  vstride, GLint  uorder, GLint  vorder, GLboolean  packed, const void * points);
typedef void (GLAPIENTRY *PFNGLMAPGRID1DPROC) (GLint  un, GLdouble  u1, GLdouble  u2);
typedef void (GLAPIENTRY *PFNGLMAPGRID1FPROC) (GLint  un, GLfloat  u1, GLfloat  u2);
typedef void (GLAPIENTRY *PFNGLMAPGRID2DPROC) (GLint  un, GLdouble  u1, GLdouble  u2, GLint  vn, GLdouble  v1, GLdouble  v2);
typedef void (GLAPIENTRY *PFNGLMAPGRID2FPROC) (GLint  un, GLfloat  u1, GLfloat  u2, GLint  vn, GLfloat  v1, GLfloat  v2);
typedef void *(GLAPIENTRY *PFNGLMAPNAMEDBUFFERPROC) (GLuint  buffer, GLenum  access);
typedef void *(GLAPIENTRY *PFNGLMAPNAMEDBUFFEREXTPROC) (GLuint  buffer, GLenum  access);
typedef void *(GLAPIENTRY *PFNGLMAPNAMEDBUFFERRANGEPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length, GLbitfield  access);
typedef void *(GLAPIENTRY *PFNGLMAPNAMEDBUFFERRANGEEXTPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length, GLbitfield  access);
typedef void (GLAPIENTRY *PFNGLMAPPARAMETERFVNVPROC) (GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLMAPPARAMETERIVNVPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void *(GLAPIENTRY *PFNGLMAPTEXTURE2DINTELPROC) (GLuint  texture, GLint  level, GLbitfield  access, GLint * stride, GLenum * layout);
typedef void (GLAPIENTRY *PFNGLMAPVERTEXATTRIB1DAPPLEPROC) (GLuint  index, GLuint  size, GLdouble  u1, GLdouble  u2, GLint  stride, GLint  order, const GLdouble * points);
typedef void (GLAPIENTRY *PFNGLMAPVERTEXATTRIB1FAPPLEPROC) (GLuint  index, GLuint  size, GLfloat  u1, GLfloat  u2, GLint  stride, GLint  order, const GLfloat * points);
typedef void (GLAPIENTRY *PFNGLMAPVERTEXATTRIB2DAPPLEPROC) (GLuint  index, GLuint  size, GLdouble  u1, GLdouble  u2, GLint  ustride, GLint  uorder, GLdouble  v1, GLdouble  v2, GLint  vstride, GLint  vorder, const GLdouble * points);
typedef void (GLAPIENTRY *PFNGLMAPVERTEXATTRIB2FAPPLEPROC) (GLuint  index, GLuint  size, GLfloat  u1, GLfloat  u2, GLint  ustride, GLint  uorder, GLfloat  v1, GLfloat  v2, GLint  vstride, GLint  vorder, const GLfloat * points);
typedef void (GLAPIENTRY *PFNGLMATERIALFPROC) (GLenum  face, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLMATERIALFVPROC) (GLenum  face, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLMATERIALIPROC) (GLenum  face, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLMATERIALIVPROC) (GLenum  face, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLMATRIXFRUSTUMEXTPROC) (GLenum  mode, GLdouble  left, GLdouble  right, GLdouble  bottom, GLdouble  top, GLdouble  zNear, GLdouble  zFar);
typedef void (GLAPIENTRY *PFNGLMATRIXINDEXPOINTERARBPROC) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLMATRIXINDEXUBVARBPROC) (GLint  size, const GLubyte * indices);
typedef void (GLAPIENTRY *PFNGLMATRIXINDEXUIVARBPROC) (GLint  size, const GLuint * indices);
typedef void (GLAPIENTRY *PFNGLMATRIXINDEXUSVARBPROC) (GLint  size, const GLushort * indices);
typedef void (GLAPIENTRY *PFNGLMATRIXLOAD3X2FNVPROC) (GLenum  matrixMode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXLOAD3X3FNVPROC) (GLenum  matrixMode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXLOADIDENTITYEXTPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLMATRIXLOADTRANSPOSE3X3FNVPROC) (GLenum  matrixMode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXLOADTRANSPOSEDEXTPROC) (GLenum  mode, const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLMATRIXLOADTRANSPOSEFEXTPROC) (GLenum  mode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXLOADDEXTPROC) (GLenum  mode, const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLMATRIXLOADFEXTPROC) (GLenum  mode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXMODEPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLMATRIXMULT3X2FNVPROC) (GLenum  matrixMode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXMULT3X3FNVPROC) (GLenum  matrixMode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXMULTTRANSPOSE3X3FNVPROC) (GLenum  matrixMode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXMULTTRANSPOSEDEXTPROC) (GLenum  mode, const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLMATRIXMULTTRANSPOSEFEXTPROC) (GLenum  mode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXMULTDEXTPROC) (GLenum  mode, const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLMATRIXMULTFEXTPROC) (GLenum  mode, const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMATRIXORTHOEXTPROC) (GLenum  mode, GLdouble  left, GLdouble  right, GLdouble  bottom, GLdouble  top, GLdouble  zNear, GLdouble  zFar);
typedef void (GLAPIENTRY *PFNGLMATRIXPOPEXTPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLMATRIXPUSHEXTPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLMATRIXROTATEDEXTPROC) (GLenum  mode, GLdouble  angle, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLMATRIXROTATEFEXTPROC) (GLenum  mode, GLfloat  angle, GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLMATRIXSCALEDEXTPROC) (GLenum  mode, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLMATRIXSCALEFEXTPROC) (GLenum  mode, GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLMATRIXTRANSLATEDEXTPROC) (GLenum  mode, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLMATRIXTRANSLATEFEXTPROC) (GLenum  mode, GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLMAXSHADERCOMPILERTHREADSKHRPROC) (GLuint  count);
typedef void (GLAPIENTRY *PFNGLMAXSHADERCOMPILERTHREADSARBPROC) (GLuint  count);
typedef void (GLAPIENTRY *PFNGLMEMORYBARRIERPROC) (GLbitfield  barriers);
typedef void (GLAPIENTRY *PFNGLMEMORYBARRIERBYREGIONPROC) (GLbitfield  barriers);
typedef void (GLAPIENTRY *PFNGLMEMORYBARRIEREXTPROC) (GLbitfield  barriers);
typedef void (GLAPIENTRY *PFNGLMEMORYOBJECTPARAMETERIVEXTPROC) (GLuint  memoryObject, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLMINSAMPLESHADINGPROC) (GLfloat  value);
typedef void (GLAPIENTRY *PFNGLMINSAMPLESHADINGARBPROC) (GLfloat  value);
typedef void (GLAPIENTRY *PFNGLMINMAXPROC) (GLenum  target, GLenum  internalformat, GLboolean  sink);
typedef void (GLAPIENTRY *PFNGLMINMAXEXTPROC) (GLenum  target, GLenum  internalformat, GLboolean  sink);
typedef void (GLAPIENTRY *PFNGLMULTMATRIXDPROC) (const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLMULTMATRIXFPROC) (const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMULTTRANSPOSEMATRIXDPROC) (const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLMULTTRANSPOSEMATRIXDARBPROC) (const GLdouble * m);
typedef void (GLAPIENTRY *PFNGLMULTTRANSPOSEMATRIXFPROC) (const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMULTTRANSPOSEMATRIXFARBPROC) (const GLfloat * m);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWARRAYSPROC) (GLenum  mode, const GLint * first, const GLsizei * count, GLsizei  drawcount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWARRAYSEXTPROC) (GLenum  mode, const GLint * first, const GLsizei * count, GLsizei  primcount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWARRAYSINDIRECTPROC) (GLenum  mode, const void * indirect, GLsizei  drawcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWARRAYSINDIRECTAMDPROC) (GLenum  mode, const void * indirect, GLsizei  primcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSCOUNTNVPROC) (GLenum  mode, const void * indirect, GLsizei  drawCount, GLsizei  maxDrawCount, GLsizei  stride, GLint  vertexBufferCount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSNVPROC) (GLenum  mode, const void * indirect, GLsizei  drawCount, GLsizei  stride, GLint  vertexBufferCount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWARRAYSINDIRECTCOUNTPROC) (GLenum  mode, const void * indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWARRAYSINDIRECTCOUNTARBPROC) (GLenum  mode, const void * indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTARRAYAPPLEPROC) (GLenum  mode, const GLint * first, const GLsizei * count, GLsizei  primcount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSPROC) (GLenum  mode, const GLsizei * count, GLenum  type, const void *const* indices, GLsizei  drawcount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC) (GLenum  mode, const GLsizei * count, GLenum  type, const void *const* indices, GLsizei  drawcount, const GLint * basevertex);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSEXTPROC) (GLenum  mode, const GLsizei * count, GLenum  type, const void *const* indices, GLsizei  primcount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSINDIRECTPROC) (GLenum  mode, GLenum  type, const void * indirect, GLsizei  drawcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSINDIRECTAMDPROC) (GLenum  mode, GLenum  type, const void * indirect, GLsizei  primcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSCOUNTNVPROC) (GLenum  mode, GLenum  type, const void * indirect, GLsizei  drawCount, GLsizei  maxDrawCount, GLsizei  stride, GLint  vertexBufferCount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSNVPROC) (GLenum  mode, GLenum  type, const void * indirect, GLsizei  drawCount, GLsizei  stride, GLint  vertexBufferCount);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTPROC) (GLenum  mode, GLenum  type, const void * indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTARBPROC) (GLenum  mode, GLenum  type, const void * indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWMESHTASKSINDIRECTNVPROC) (GLintptr  indirect, GLsizei  drawcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWMESHTASKSINDIRECTCOUNTNVPROC) (GLintptr  indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLMULTIDRAWRANGEELEMENTARRAYAPPLEPROC) (GLenum  mode, GLuint  start, GLuint  end, const GLint * first, const GLsizei * count, GLsizei  primcount);
typedef void (GLAPIENTRY *PFNGLMULTITEXBUFFEREXTPROC) (GLenum  texunit, GLenum  target, GLenum  internalformat, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1DPROC) (GLenum  target, GLdouble  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1DARBPROC) (GLenum  target, GLdouble  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1DVPROC) (GLenum  target, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1DVARBPROC) (GLenum  target, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1FPROC) (GLenum  target, GLfloat  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1FARBPROC) (GLenum  target, GLfloat  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1FVPROC) (GLenum  target, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1FVARBPROC) (GLenum  target, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1HNVPROC) (GLenum  target, GLhalfNV  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1HVNVPROC) (GLenum  target, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1IPROC) (GLenum  target, GLint  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1IARBPROC) (GLenum  target, GLint  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1IVPROC) (GLenum  target, const GLint * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1IVARBPROC) (GLenum  target, const GLint * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1SPROC) (GLenum  target, GLshort  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1SARBPROC) (GLenum  target, GLshort  s);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1SVPROC) (GLenum  target, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD1SVARBPROC) (GLenum  target, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2DPROC) (GLenum  target, GLdouble  s, GLdouble  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2DARBPROC) (GLenum  target, GLdouble  s, GLdouble  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2DVPROC) (GLenum  target, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2DVARBPROC) (GLenum  target, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2FPROC) (GLenum  target, GLfloat  s, GLfloat  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2FARBPROC) (GLenum  target, GLfloat  s, GLfloat  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2FVPROC) (GLenum  target, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2FVARBPROC) (GLenum  target, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2HNVPROC) (GLenum  target, GLhalfNV  s, GLhalfNV  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2HVNVPROC) (GLenum  target, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2IPROC) (GLenum  target, GLint  s, GLint  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2IARBPROC) (GLenum  target, GLint  s, GLint  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2IVPROC) (GLenum  target, const GLint * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2IVARBPROC) (GLenum  target, const GLint * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2SPROC) (GLenum  target, GLshort  s, GLshort  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2SARBPROC) (GLenum  target, GLshort  s, GLshort  t);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2SVPROC) (GLenum  target, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD2SVARBPROC) (GLenum  target, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3DPROC) (GLenum  target, GLdouble  s, GLdouble  t, GLdouble  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3DARBPROC) (GLenum  target, GLdouble  s, GLdouble  t, GLdouble  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3DVPROC) (GLenum  target, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3DVARBPROC) (GLenum  target, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3FPROC) (GLenum  target, GLfloat  s, GLfloat  t, GLfloat  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3FARBPROC) (GLenum  target, GLfloat  s, GLfloat  t, GLfloat  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3FVPROC) (GLenum  target, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3FVARBPROC) (GLenum  target, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3HNVPROC) (GLenum  target, GLhalfNV  s, GLhalfNV  t, GLhalfNV  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3HVNVPROC) (GLenum  target, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3IPROC) (GLenum  target, GLint  s, GLint  t, GLint  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3IARBPROC) (GLenum  target, GLint  s, GLint  t, GLint  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3IVPROC) (GLenum  target, const GLint * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3IVARBPROC) (GLenum  target, const GLint * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3SPROC) (GLenum  target, GLshort  s, GLshort  t, GLshort  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3SARBPROC) (GLenum  target, GLshort  s, GLshort  t, GLshort  r);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3SVPROC) (GLenum  target, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD3SVARBPROC) (GLenum  target, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4DPROC) (GLenum  target, GLdouble  s, GLdouble  t, GLdouble  r, GLdouble  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4DARBPROC) (GLenum  target, GLdouble  s, GLdouble  t, GLdouble  r, GLdouble  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4DVPROC) (GLenum  target, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4DVARBPROC) (GLenum  target, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4FPROC) (GLenum  target, GLfloat  s, GLfloat  t, GLfloat  r, GLfloat  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4FARBPROC) (GLenum  target, GLfloat  s, GLfloat  t, GLfloat  r, GLfloat  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4FVPROC) (GLenum  target, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4FVARBPROC) (GLenum  target, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4HNVPROC) (GLenum  target, GLhalfNV  s, GLhalfNV  t, GLhalfNV  r, GLhalfNV  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4HVNVPROC) (GLenum  target, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4IPROC) (GLenum  target, GLint  s, GLint  t, GLint  r, GLint  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4IARBPROC) (GLenum  target, GLint  s, GLint  t, GLint  r, GLint  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4IVPROC) (GLenum  target, const GLint * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4IVARBPROC) (GLenum  target, const GLint * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4SPROC) (GLenum  target, GLshort  s, GLshort  t, GLshort  r, GLshort  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4SARBPROC) (GLenum  target, GLshort  s, GLshort  t, GLshort  r, GLshort  q);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4SVPROC) (GLenum  target, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORD4SVARBPROC) (GLenum  target, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDP1UIPROC) (GLenum  texture, GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDP1UIVPROC) (GLenum  texture, GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDP2UIPROC) (GLenum  texture, GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDP2UIVPROC) (GLenum  texture, GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDP3UIPROC) (GLenum  texture, GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDP3UIVPROC) (GLenum  texture, GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDP4UIPROC) (GLenum  texture, GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDP4UIVPROC) (GLenum  texture, GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLMULTITEXCOORDPOINTEREXTPROC) (GLenum  texunit, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLMULTITEXENVFEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLMULTITEXENVFVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXENVIEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLMULTITEXENVIVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXGENDEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, GLdouble  param);
typedef void (GLAPIENTRY *PFNGLMULTITEXGENDVEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, const GLdouble * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXGENFEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLMULTITEXGENFVEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXGENIEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLMULTITEXGENIVEXTPROC) (GLenum  texunit, GLenum  coord, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXIMAGE1DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLMULTITEXIMAGE2DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLMULTITEXIMAGE3DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLMULTITEXPARAMETERIIVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXPARAMETERIUIVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXPARAMETERFEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLMULTITEXPARAMETERFVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXPARAMETERIEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLMULTITEXPARAMETERIVEXTPROC) (GLenum  texunit, GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLMULTITEXRENDERBUFFEREXTPROC) (GLenum  texunit, GLenum  target, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLMULTITEXSUBIMAGE1DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLMULTITEXSUBIMAGE2DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLMULTITEXSUBIMAGE3DEXTPROC) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLMULTICASTBARRIERNVPROC) ();
typedef void (GLAPIENTRY *PFNGLMULTICASTBLITFRAMEBUFFERNVPROC) (GLuint  srcGpu, GLuint  dstGpu, GLint  srcX0, GLint  srcY0, GLint  srcX1, GLint  srcY1, GLint  dstX0, GLint  dstY0, GLint  dstX1, GLint  dstY1, GLbitfield  mask, GLenum  filter);
typedef void (GLAPIENTRY *PFNGLMULTICASTBUFFERSUBDATANVPROC) (GLbitfield  gpuMask, GLuint  buffer, GLintptr  offset, GLsizeiptr  size, const void * data);
typedef void (GLAPIENTRY *PFNGLMULTICASTCOPYBUFFERSUBDATANVPROC) (GLuint  readGpu, GLbitfield  writeGpuMask, GLuint  readBuffer, GLuint  writeBuffer, GLintptr  readOffset, GLintptr  writeOffset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLMULTICASTCOPYIMAGESUBDATANVPROC) (GLuint  srcGpu, GLbitfield  dstGpuMask, GLuint  srcName, GLenum  srcTarget, GLint  srcLevel, GLint  srcX, GLint  srcY, GLint  srcZ, GLuint  dstName, GLenum  dstTarget, GLint  dstLevel, GLint  dstX, GLint  dstY, GLint  dstZ, GLsizei  srcWidth, GLsizei  srcHeight, GLsizei  srcDepth);
typedef void (GLAPIENTRY *PFNGLMULTICASTFRAMEBUFFERSAMPLELOCATIONSFVNVPROC) (GLuint  gpu, GLuint  framebuffer, GLuint  start, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLMULTICASTGETQUERYOBJECTI64VNVPROC) (GLuint  gpu, GLuint  id, GLenum  pname, GLint64 * params);
typedef void (GLAPIENTRY *PFNGLMULTICASTGETQUERYOBJECTIVNVPROC) (GLuint  gpu, GLuint  id, GLenum  pname, GLint * params);
typedef void (GLAPIENTRY *PFNGLMULTICASTGETQUERYOBJECTUI64VNVPROC) (GLuint  gpu, GLuint  id, GLenum  pname, GLuint64 * params);
typedef void (GLAPIENTRY *PFNGLMULTICASTGETQUERYOBJECTUIVNVPROC) (GLuint  gpu, GLuint  id, GLenum  pname, GLuint * params);
typedef void (GLAPIENTRY *PFNGLMULTICASTWAITSYNCNVPROC) (GLuint  signalGpu, GLbitfield  waitGpuMask);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERATTACHMEMORYNVPROC) (GLuint  buffer, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERDATAPROC) (GLuint  buffer, GLsizeiptr  size, const void * data, GLenum  usage);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERDATAEXTPROC) (GLuint  buffer, GLsizeiptr  size, const void * data, GLenum  usage);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERPAGECOMMITMENTARBPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, GLboolean  commit);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERPAGECOMMITMENTEXTPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, GLboolean  commit);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERSTORAGEPROC) (GLuint  buffer, GLsizeiptr  size, const void * data, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERSTORAGEEXTERNALEXTPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, GLeglClientBufferEXT  clientBuffer, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERSTORAGEEXTPROC) (GLuint  buffer, GLsizeiptr  size, const void * data, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERSTORAGEMEMEXTPROC) (GLuint  buffer, GLsizeiptr  size, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERSUBDATAPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, const void * data);
typedef void (GLAPIENTRY *PFNGLNAMEDBUFFERSUBDATAEXTPROC) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, const void * data);
typedef void (GLAPIENTRY *PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC) (GLuint  readBuffer, GLuint  writeBuffer, GLintptr  readOffset, GLintptr  writeOffset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC) (GLuint  framebuffer, GLenum  buf);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC) (GLuint  framebuffer, GLsizei  n, const GLenum * bufs);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERPARAMETERIPROC) (GLuint  framebuffer, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERPARAMETERIEXTPROC) (GLuint  framebuffer, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC) (GLuint  framebuffer, GLenum  src);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC) (GLuint  framebuffer, GLenum  attachment, GLenum  renderbuffertarget, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERRENDERBUFFEREXTPROC) (GLuint  framebuffer, GLenum  attachment, GLenum  renderbuffertarget, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVARBPROC) (GLuint  framebuffer, GLuint  start, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVNVPROC) (GLuint  framebuffer, GLuint  start, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTUREPROC) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERSAMPLEPOSITIONSFVAMDPROC) (GLuint  framebuffer, GLuint  numsamples, GLuint  pixelindex, const GLfloat * values);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTURE1DEXTPROC) (GLuint  framebuffer, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC) (GLuint  framebuffer, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTURE3DEXTPROC) (GLuint  framebuffer, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level, GLint  zoffset);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTUREFACEEXTPROC) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level, GLenum  face);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
typedef void (GLAPIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETER4DEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETER4DVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, const GLdouble * params);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETER4FEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETER4FVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETERI4IEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETERI4IVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, const GLint * params);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETERI4UIEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETERI4UIVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETERS4FVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLsizei  count, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETERSI4IVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLsizei  count, const GLint * params);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMLOCALPARAMETERSI4UIVEXTPROC) (GLuint  program, GLenum  target, GLuint  index, GLsizei  count, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLNAMEDPROGRAMSTRINGEXTPROC) (GLuint  program, GLenum  target, GLenum  format, GLsizei  len, const void * string);
typedef void (GLAPIENTRY *PFNGLNAMEDRENDERBUFFERSTORAGEPROC) (GLuint  renderbuffer, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLNAMEDRENDERBUFFERSTORAGEEXTPROC) (GLuint  renderbuffer, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEPROC) (GLuint  renderbuffer, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEADVANCEDAMDPROC) (GLuint  renderbuffer, GLsizei  samples, GLsizei  storageSamples, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLECOVERAGEEXTPROC) (GLuint  renderbuffer, GLsizei  coverageSamples, GLsizei  colorSamples, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC) (GLuint  renderbuffer, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLNAMEDSTRINGARBPROC) (GLenum  type, GLint  namelen, const GLchar * name, GLint  stringlen, const GLchar * string);
typedef void (GLAPIENTRY *PFNGLNEWLISTPROC) (GLuint  list, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLNORMAL3BPROC) (GLbyte  nx, GLbyte  ny, GLbyte  nz);
typedef void (GLAPIENTRY *PFNGLNORMAL3BVPROC) (const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLNORMAL3DPROC) (GLdouble  nx, GLdouble  ny, GLdouble  nz);
typedef void (GLAPIENTRY *PFNGLNORMAL3DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLNORMAL3FPROC) (GLfloat  nx, GLfloat  ny, GLfloat  nz);
typedef void (GLAPIENTRY *PFNGLNORMAL3FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLNORMAL3HNVPROC) (GLhalfNV  nx, GLhalfNV  ny, GLhalfNV  nz);
typedef void (GLAPIENTRY *PFNGLNORMAL3HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLNORMAL3IPROC) (GLint  nx, GLint  ny, GLint  nz);
typedef void (GLAPIENTRY *PFNGLNORMAL3IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLNORMAL3SPROC) (GLshort  nx, GLshort  ny, GLshort  nz);
typedef void (GLAPIENTRY *PFNGLNORMAL3SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLNORMALFORMATNVPROC) (GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLNORMALP3UIPROC) (GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLNORMALP3UIVPROC) (GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLNORMALPOINTERPROC) (GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLNORMALPOINTEREXTPROC) (GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
typedef void (GLAPIENTRY *PFNGLNORMALPOINTERVINTELPROC) (GLenum  type, const void ** pointer);
typedef void (GLAPIENTRY *PFNGLOBJECTLABELPROC) (GLenum  identifier, GLuint  name, GLsizei  length, const GLchar * label);
typedef void (GLAPIENTRY *PFNGLOBJECTLABELKHRPROC) (GLenum  identifier, GLuint  name, GLsizei  length, const GLchar * label);
typedef void (GLAPIENTRY *PFNGLOBJECTPTRLABELPROC) (const void * ptr, GLsizei  length, const GLchar * label);
typedef void (GLAPIENTRY *PFNGLOBJECTPTRLABELKHRPROC) (const void * ptr, GLsizei  length, const GLchar * label);
typedef GLenum (GLAPIENTRY *PFNGLOBJECTPURGEABLEAPPLEPROC) (GLenum  objectType, GLuint  name, GLenum  option);
typedef GLenum (GLAPIENTRY *PFNGLOBJECTUNPURGEABLEAPPLEPROC) (GLenum  objectType, GLuint  name, GLenum  option);
typedef void (GLAPIENTRY *PFNGLORTHOPROC) (GLdouble  left, GLdouble  right, GLdouble  bottom, GLdouble  top, GLdouble  zNear, GLdouble  zFar);
typedef void (GLAPIENTRY *PFNGLPASSTHROUGHPROC) (GLfloat  token);
typedef void (GLAPIENTRY *PFNGLPATCHPARAMETERFVPROC) (GLenum  pname, const GLfloat * values);
typedef void (GLAPIENTRY *PFNGLPATCHPARAMETERIPROC) (GLenum  pname, GLint  value);
typedef void (GLAPIENTRY *PFNGLPATHCOLORGENNVPROC) (GLenum  color, GLenum  genMode, GLenum  colorFormat, const GLfloat * coeffs);
typedef void (GLAPIENTRY *PFNGLPATHCOMMANDSNVPROC) (GLuint  path, GLsizei  numCommands, const GLubyte * commands, GLsizei  numCoords, GLenum  coordType, const void * coords);
typedef void (GLAPIENTRY *PFNGLPATHCOORDSNVPROC) (GLuint  path, GLsizei  numCoords, GLenum  coordType, const void * coords);
typedef void (GLAPIENTRY *PFNGLPATHCOVERDEPTHFUNCNVPROC) (GLenum  func);
typedef void (GLAPIENTRY *PFNGLPATHDASHARRAYNVPROC) (GLuint  path, GLsizei  dashCount, const GLfloat * dashArray);
typedef void (GLAPIENTRY *PFNGLPATHFOGGENNVPROC) (GLenum  genMode);
typedef GLenum (GLAPIENTRY *PFNGLPATHGLYPHINDEXARRAYNVPROC) (GLuint  firstPathName, GLenum  fontTarget, const void * fontName, GLbitfield  fontStyle, GLuint  firstGlyphIndex, GLsizei  numGlyphs, GLuint  pathParameterTemplate, GLfloat  emScale);
typedef GLenum (GLAPIENTRY *PFNGLPATHGLYPHINDEXRANGENVPROC) (GLenum  fontTarget, const void * fontName, GLbitfield  fontStyle, GLuint  pathParameterTemplate, GLfloat  emScale, GLuint  baseAndCount);
typedef void (GLAPIENTRY *PFNGLPATHGLYPHRANGENVPROC) (GLuint  firstPathName, GLenum  fontTarget, const void * fontName, GLbitfield  fontStyle, GLuint  firstGlyph, GLsizei  numGlyphs, GLenum  handleMissingGlyphs, GLuint  pathParameterTemplate, GLfloat  emScale);
typedef void (GLAPIENTRY *PFNGLPATHGLYPHSNVPROC) (GLuint  firstPathName, GLenum  fontTarget, const void * fontName, GLbitfield  fontStyle, GLsizei  numGlyphs, GLenum  type, const void * charcodes, GLenum  handleMissingGlyphs, GLuint  pathParameterTemplate, GLfloat  emScale);
typedef GLenum (GLAPIENTRY *PFNGLPATHMEMORYGLYPHINDEXARRAYNVPROC) (GLuint  firstPathName, GLenum  fontTarget, GLsizeiptr  fontSize, const void * fontData, GLsizei  faceIndex, GLuint  firstGlyphIndex, GLsizei  numGlyphs, GLuint  pathParameterTemplate, GLfloat  emScale);
typedef void (GLAPIENTRY *PFNGLPATHPARAMETERFNVPROC) (GLuint  path, GLenum  pname, GLfloat  value);
typedef void (GLAPIENTRY *PFNGLPATHPARAMETERFVNVPROC) (GLuint  path, GLenum  pname, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPATHPARAMETERINVPROC) (GLuint  path, GLenum  pname, GLint  value);
typedef void (GLAPIENTRY *PFNGLPATHPARAMETERIVNVPROC) (GLuint  path, GLenum  pname, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPATHSTENCILDEPTHOFFSETNVPROC) (GLfloat  factor, GLfloat  units);
typedef void (GLAPIENTRY *PFNGLPATHSTENCILFUNCNVPROC) (GLenum  func, GLint  ref, GLuint  mask);
typedef void (GLAPIENTRY *PFNGLPATHSTRINGNVPROC) (GLuint  path, GLenum  format, GLsizei  length, const void * pathString);
typedef void (GLAPIENTRY *PFNGLPATHSUBCOMMANDSNVPROC) (GLuint  path, GLsizei  commandStart, GLsizei  commandsToDelete, GLsizei  numCommands, const GLubyte * commands, GLsizei  numCoords, GLenum  coordType, const void * coords);
typedef void (GLAPIENTRY *PFNGLPATHSUBCOORDSNVPROC) (GLuint  path, GLsizei  coordStart, GLsizei  numCoords, GLenum  coordType, const void * coords);
typedef void (GLAPIENTRY *PFNGLPATHTEXGENNVPROC) (GLenum  texCoordSet, GLenum  genMode, GLint  components, const GLfloat * coeffs);
typedef void (GLAPIENTRY *PFNGLPAUSETRANSFORMFEEDBACKPROC) ();
typedef void (GLAPIENTRY *PFNGLPAUSETRANSFORMFEEDBACKNVPROC) ();
typedef void (GLAPIENTRY *PFNGLPIXELDATARANGENVPROC) (GLenum  target, GLsizei  length, const void * pointer);
typedef void (GLAPIENTRY *PFNGLPIXELMAPFVPROC) (GLenum  map, GLsizei  mapsize, const GLfloat * values);
typedef void (GLAPIENTRY *PFNGLPIXELMAPUIVPROC) (GLenum  map, GLsizei  mapsize, const GLuint * values);
typedef void (GLAPIENTRY *PFNGLPIXELMAPUSVPROC) (GLenum  map, GLsizei  mapsize, const GLushort * values);
typedef void (GLAPIENTRY *PFNGLPIXELSTOREFPROC) (GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLPIXELSTOREIPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLPIXELTRANSFERFPROC) (GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLPIXELTRANSFERIPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLPIXELTRANSFORMPARAMETERFEXTPROC) (GLenum  target, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLPIXELTRANSFORMPARAMETERFVEXTPROC) (GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPIXELTRANSFORMPARAMETERIEXTPROC) (GLenum  target, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLPIXELTRANSFORMPARAMETERIVEXTPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLPIXELZOOMPROC) (GLfloat  xfactor, GLfloat  yfactor);
typedef GLboolean (GLAPIENTRY *PFNGLPOINTALONGPATHNVPROC) (GLuint  path, GLsizei  startSegment, GLsizei  numSegments, GLfloat  distance, GLfloat * x, GLfloat * y, GLfloat * tangentX, GLfloat * tangentY);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERFPROC) (GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERFARBPROC) (GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERFEXTPROC) (GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERFVPROC) (GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERFVARBPROC) (GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERFVEXTPROC) (GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERIPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERINVPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERIVPROC) (GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLPOINTPARAMETERIVNVPROC) (GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLPOINTSIZEPROC) (GLfloat  size);
typedef void (GLAPIENTRY *PFNGLPOLYGONMODEPROC) (GLenum  face, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLPOLYGONOFFSETPROC) (GLfloat  factor, GLfloat  units);
typedef void (GLAPIENTRY *PFNGLPOLYGONOFFSETCLAMPPROC) (GLfloat  factor, GLfloat  units, GLfloat  clamp);
typedef void (GLAPIENTRY *PFNGLPOLYGONOFFSETCLAMPEXTPROC) (GLfloat  factor, GLfloat  units, GLfloat  clamp);
typedef void (GLAPIENTRY *PFNGLPOLYGONOFFSETEXTPROC) (GLfloat  factor, GLfloat  bias);
typedef void (GLAPIENTRY *PFNGLPOLYGONSTIPPLEPROC) (const GLubyte * mask);
typedef void (GLAPIENTRY *PFNGLPOPATTRIBPROC) ();
typedef void (GLAPIENTRY *PFNGLPOPCLIENTATTRIBPROC) ();
typedef void (GLAPIENTRY *PFNGLPOPDEBUGGROUPPROC) ();
typedef void (GLAPIENTRY *PFNGLPOPDEBUGGROUPKHRPROC) ();
typedef void (GLAPIENTRY *PFNGLPOPGROUPMARKEREXTPROC) ();
typedef void (GLAPIENTRY *PFNGLPOPMATRIXPROC) ();
typedef void (GLAPIENTRY *PFNGLPOPNAMEPROC) ();
typedef void (GLAPIENTRY *PFNGLPRESENTFRAMEDUALFILLNVPROC) (GLuint  video_slot, GLuint64EXT  minPresentTime, GLuint  beginPresentTimeId, GLuint  presentDurationId, GLenum  type, GLenum  target0, GLuint  fill0, GLenum  target1, GLuint  fill1, GLenum  target2, GLuint  fill2, GLenum  target3, GLuint  fill3);
typedef void (GLAPIENTRY *PFNGLPRESENTFRAMEKEYEDNVPROC) (GLuint  video_slot, GLuint64EXT  minPresentTime, GLuint  beginPresentTimeId, GLuint  presentDurationId, GLenum  type, GLenum  target0, GLuint  fill0, GLuint  key0, GLenum  target1, GLuint  fill1, GLuint  key1);
typedef void (GLAPIENTRY *PFNGLPRIMITIVEBOUNDINGBOXARBPROC) (GLfloat  minX, GLfloat  minY, GLfloat  minZ, GLfloat  minW, GLfloat  maxX, GLfloat  maxY, GLfloat  maxZ, GLfloat  maxW);
typedef void (GLAPIENTRY *PFNGLPRIMITIVERESTARTINDEXPROC) (GLuint  index);
typedef void (GLAPIENTRY *PFNGLPRIMITIVERESTARTINDEXNVPROC) (GLuint  index);
typedef void (GLAPIENTRY *PFNGLPRIMITIVERESTARTNVPROC) ();
typedef void (GLAPIENTRY *PFNGLPRIORITIZETEXTURESPROC) (GLsizei  n, const GLuint * textures, const GLfloat * priorities);
typedef void (GLAPIENTRY *PFNGLPRIORITIZETEXTURESEXTPROC) (GLsizei  n, const GLuint * textures, const GLclampf * priorities);
typedef void (GLAPIENTRY *PFNGLPROGRAMBINARYPROC) (GLuint  program, GLenum  binaryFormat, const void * binary, GLsizei  length);
typedef void (GLAPIENTRY *PFNGLPROGRAMBUFFERPARAMETERSIIVNVPROC) (GLenum  target, GLuint  bindingIndex, GLuint  wordIndex, GLsizei  count, const GLint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMBUFFERPARAMETERSIUIVNVPROC) (GLenum  target, GLuint  bindingIndex, GLuint  wordIndex, GLsizei  count, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMBUFFERPARAMETERSFVNVPROC) (GLenum  target, GLuint  bindingIndex, GLuint  wordIndex, GLsizei  count, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETER4DARBPROC) (GLenum  target, GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETER4DVARBPROC) (GLenum  target, GLuint  index, const GLdouble * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETER4FARBPROC) (GLenum  target, GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETER4FVARBPROC) (GLenum  target, GLuint  index, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETERI4INVPROC) (GLenum  target, GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETERI4IVNVPROC) (GLenum  target, GLuint  index, const GLint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETERI4UINVPROC) (GLenum  target, GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETERI4UIVNVPROC) (GLenum  target, GLuint  index, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETERS4FVEXTPROC) (GLenum  target, GLuint  index, GLsizei  count, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETERSI4IVNVPROC) (GLenum  target, GLuint  index, GLsizei  count, const GLint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMENVPARAMETERSI4UIVNVPROC) (GLenum  target, GLuint  index, GLsizei  count, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETER4DARBPROC) (GLenum  target, GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETER4DVARBPROC) (GLenum  target, GLuint  index, const GLdouble * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETER4FARBPROC) (GLenum  target, GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETER4FVARBPROC) (GLenum  target, GLuint  index, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETERI4INVPROC) (GLenum  target, GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETERI4IVNVPROC) (GLenum  target, GLuint  index, const GLint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETERI4UINVPROC) (GLenum  target, GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETERI4UIVNVPROC) (GLenum  target, GLuint  index, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETERS4FVEXTPROC) (GLenum  target, GLuint  index, GLsizei  count, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETERSI4IVNVPROC) (GLenum  target, GLuint  index, GLsizei  count, const GLint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMLOCALPARAMETERSI4UIVNVPROC) (GLenum  target, GLuint  index, GLsizei  count, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMNAMEDPARAMETER4DNVPROC) (GLuint  id, GLsizei  len, const GLubyte * name, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMNAMEDPARAMETER4DVNVPROC) (GLuint  id, GLsizei  len, const GLubyte * name, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLPROGRAMNAMEDPARAMETER4FNVPROC) (GLuint  id, GLsizei  len, const GLubyte * name, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC) (GLuint  id, GLsizei  len, const GLubyte * name, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETER4DNVPROC) (GLenum  target, GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETER4DVNVPROC) (GLenum  target, GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETER4FNVPROC) (GLenum  target, GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETER4FVNVPROC) (GLenum  target, GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETERIPROC) (GLuint  program, GLenum  pname, GLint  value);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETERIARBPROC) (GLuint  program, GLenum  pname, GLint  value);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETERIEXTPROC) (GLuint  program, GLenum  pname, GLint  value);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETERS4DVNVPROC) (GLenum  target, GLuint  index, GLsizei  count, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLPROGRAMPARAMETERS4FVNVPROC) (GLenum  target, GLuint  index, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLPROGRAMPATHFRAGMENTINPUTGENNVPROC) (GLuint  program, GLint  location, GLenum  genMode, GLint  components, const GLfloat * coeffs);
typedef void (GLAPIENTRY *PFNGLPROGRAMSTRINGARBPROC) (GLenum  target, GLenum  format, GLsizei  len, const void * string);
typedef void (GLAPIENTRY *PFNGLPROGRAMSUBROUTINEPARAMETERSUIVNVPROC) (GLenum  target, GLsizei  count, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1DPROC) (GLuint  program, GLint  location, GLdouble  v0);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1DEXTPROC) (GLuint  program, GLint  location, GLdouble  x);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1DVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1FPROC) (GLuint  program, GLint  location, GLfloat  v0);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1FEXTPROC) (GLuint  program, GLint  location, GLfloat  v0);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1FVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1IPROC) (GLuint  program, GLint  location, GLint  v0);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1I64ARBPROC) (GLuint  program, GLint  location, GLint64  x);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1I64NVPROC) (GLuint  program, GLint  location, GLint64EXT  x);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1I64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint64 * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1I64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1IEXTPROC) (GLuint  program, GLint  location, GLint  v0);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1IVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1IVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1UIPROC) (GLuint  program, GLint  location, GLuint  v0);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1UI64ARBPROC) (GLuint  program, GLint  location, GLuint64  x);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1UI64NVPROC) (GLuint  program, GLint  location, GLuint64EXT  x);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1UI64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1UI64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1UIEXTPROC) (GLuint  program, GLint  location, GLuint  v0);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1UIVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM1UIVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2DPROC) (GLuint  program, GLint  location, GLdouble  v0, GLdouble  v1);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2DEXTPROC) (GLuint  program, GLint  location, GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2DVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2FPROC) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2FEXTPROC) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2FVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2IPROC) (GLuint  program, GLint  location, GLint  v0, GLint  v1);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2I64ARBPROC) (GLuint  program, GLint  location, GLint64  x, GLint64  y);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2I64NVPROC) (GLuint  program, GLint  location, GLint64EXT  x, GLint64EXT  y);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2I64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint64 * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2I64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2IEXTPROC) (GLuint  program, GLint  location, GLint  v0, GLint  v1);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2IVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2IVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2UIPROC) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2UI64ARBPROC) (GLuint  program, GLint  location, GLuint64  x, GLuint64  y);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2UI64NVPROC) (GLuint  program, GLint  location, GLuint64EXT  x, GLuint64EXT  y);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2UI64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2UI64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2UIEXTPROC) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2UIVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM2UIVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3DPROC) (GLuint  program, GLint  location, GLdouble  v0, GLdouble  v1, GLdouble  v2);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3DEXTPROC) (GLuint  program, GLint  location, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3DVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3FPROC) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3FEXTPROC) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3FVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3IPROC) (GLuint  program, GLint  location, GLint  v0, GLint  v1, GLint  v2);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3I64ARBPROC) (GLuint  program, GLint  location, GLint64  x, GLint64  y, GLint64  z);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3I64NVPROC) (GLuint  program, GLint  location, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3I64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint64 * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3I64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3IEXTPROC) (GLuint  program, GLint  location, GLint  v0, GLint  v1, GLint  v2);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3IVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3IVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3UIPROC) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1, GLuint  v2);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3UI64ARBPROC) (GLuint  program, GLint  location, GLuint64  x, GLuint64  y, GLuint64  z);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3UI64NVPROC) (GLuint  program, GLint  location, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3UI64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3UI64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3UIEXTPROC) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1, GLuint  v2);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3UIVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM3UIVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4DPROC) (GLuint  program, GLint  location, GLdouble  v0, GLdouble  v1, GLdouble  v2, GLdouble  v3);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4DEXTPROC) (GLuint  program, GLint  location, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4DVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4FPROC) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2, GLfloat  v3);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4FEXTPROC) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2, GLfloat  v3);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4FVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4IPROC) (GLuint  program, GLint  location, GLint  v0, GLint  v1, GLint  v2, GLint  v3);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4I64ARBPROC) (GLuint  program, GLint  location, GLint64  x, GLint64  y, GLint64  z, GLint64  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4I64NVPROC) (GLuint  program, GLint  location, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z, GLint64EXT  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4I64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint64 * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4I64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4IEXTPROC) (GLuint  program, GLint  location, GLint  v0, GLint  v1, GLint  v2, GLint  v3);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4IVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4IVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4UIPROC) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1, GLuint  v2, GLuint  v3);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4UI64ARBPROC) (GLuint  program, GLint  location, GLuint64  x, GLuint64  y, GLuint64  z, GLuint64  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4UI64NVPROC) (GLuint  program, GLint  location, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z, GLuint64EXT  w);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4UI64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4UI64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4UIEXTPROC) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1, GLuint  v2, GLuint  v3);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4UIVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORM4UIVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMHANDLEUI64ARBPROC) (GLuint  program, GLint  location, GLuint64  value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMHANDLEUI64NVPROC) (GLuint  program, GLint  location, GLuint64  value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMHANDLEUI64VARBPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * values);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMHANDLEUI64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * values);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X3DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X3FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X4DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X4FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X2DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X2FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X4DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X4FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X2DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X2FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X3DVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X3FVEXTPROC) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMUI64NVPROC) (GLuint  program, GLint  location, GLuint64EXT  value);
typedef void (GLAPIENTRY *PFNGLPROGRAMUNIFORMUI64VNVPROC) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLPROGRAMVERTEXLIMITNVPROC) (GLenum  target, GLint  limit);
typedef void (GLAPIENTRY *PFNGLPROVOKINGVERTEXPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLPROVOKINGVERTEXEXTPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLPUSHATTRIBPROC) (GLbitfield  mask);
typedef void (GLAPIENTRY *PFNGLPUSHCLIENTATTRIBPROC) (GLbitfield  mask);
typedef void (GLAPIENTRY *PFNGLPUSHCLIENTATTRIBDEFAULTEXTPROC) (GLbitfield  mask);
typedef void (GLAPIENTRY *PFNGLPUSHDEBUGGROUPPROC) (GLenum  source, GLuint  id, GLsizei  length, const GLchar * message);
typedef void (GLAPIENTRY *PFNGLPUSHDEBUGGROUPKHRPROC) (GLenum  source, GLuint  id, GLsizei  length, const GLchar * message);
typedef void (GLAPIENTRY *PFNGLPUSHGROUPMARKEREXTPROC) (GLsizei  length, const GLchar * marker);
typedef void (GLAPIENTRY *PFNGLPUSHMATRIXPROC) ();
typedef void (GLAPIENTRY *PFNGLPUSHNAMEPROC) (GLuint  name);
typedef void (GLAPIENTRY *PFNGLQUERYCOUNTERPROC) (GLuint  id, GLenum  target);
typedef void (GLAPIENTRY *PFNGLQUERYOBJECTPARAMETERUIAMDPROC) (GLenum  target, GLuint  id, GLenum  pname, GLuint  param);
typedef GLint (GLAPIENTRY *PFNGLQUERYRESOURCENVPROC) (GLenum  queryType, GLint  tagId, GLuint  count, GLint * buffer);
typedef void (GLAPIENTRY *PFNGLQUERYRESOURCETAGNVPROC) (GLint  tagId, const GLchar * tagString);
typedef void (GLAPIENTRY *PFNGLRASTERPOS2DPROC) (GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLRASTERPOS2DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS2FPROC) (GLfloat  x, GLfloat  y);
typedef void (GLAPIENTRY *PFNGLRASTERPOS2FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS2IPROC) (GLint  x, GLint  y);
typedef void (GLAPIENTRY *PFNGLRASTERPOS2IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS2SPROC) (GLshort  x, GLshort  y);
typedef void (GLAPIENTRY *PFNGLRASTERPOS2SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS3DPROC) (GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLRASTERPOS3DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS3FPROC) (GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLRASTERPOS3FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS3IPROC) (GLint  x, GLint  y, GLint  z);
typedef void (GLAPIENTRY *PFNGLRASTERPOS3IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS3SPROC) (GLshort  x, GLshort  y, GLshort  z);
typedef void (GLAPIENTRY *PFNGLRASTERPOS3SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS4DPROC) (GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLRASTERPOS4DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS4FPROC) (GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLRASTERPOS4FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS4IPROC) (GLint  x, GLint  y, GLint  z, GLint  w);
typedef void (GLAPIENTRY *PFNGLRASTERPOS4IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLRASTERPOS4SPROC) (GLshort  x, GLshort  y, GLshort  z, GLshort  w);
typedef void (GLAPIENTRY *PFNGLRASTERPOS4SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLRASTERSAMPLESEXTPROC) (GLuint  samples, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLREADBUFFERPROC) (GLenum  src);
typedef void (GLAPIENTRY *PFNGLREADPIXELSPROC) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, void * pixels);
typedef void (GLAPIENTRY *PFNGLREADNPIXELSPROC) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, GLsizei  bufSize, void * data);
typedef void (GLAPIENTRY *PFNGLREADNPIXELSARBPROC) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, GLsizei  bufSize, void * data);
typedef void (GLAPIENTRY *PFNGLREADNPIXELSKHRPROC) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, GLsizei  bufSize, void * data);
typedef GLboolean (GLAPIENTRY *PFNGLRELEASEKEYEDMUTEXWIN32EXTPROC) (GLuint  memory, GLuint64  key);
typedef void (GLAPIENTRY *PFNGLRECTDPROC) (GLdouble  x1, GLdouble  y1, GLdouble  x2, GLdouble  y2);
typedef void (GLAPIENTRY *PFNGLRECTDVPROC) (const GLdouble * v1, const GLdouble * v2);
typedef void (GLAPIENTRY *PFNGLRECTFPROC) (GLfloat  x1, GLfloat  y1, GLfloat  x2, GLfloat  y2);
typedef void (GLAPIENTRY *PFNGLRECTFVPROC) (const GLfloat * v1, const GLfloat * v2);
typedef void (GLAPIENTRY *PFNGLRECTIPROC) (GLint  x1, GLint  y1, GLint  x2, GLint  y2);
typedef void (GLAPIENTRY *PFNGLRECTIVPROC) (const GLint * v1, const GLint * v2);
typedef void (GLAPIENTRY *PFNGLRECTSPROC) (GLshort  x1, GLshort  y1, GLshort  x2, GLshort  y2);
typedef void (GLAPIENTRY *PFNGLRECTSVPROC) (const GLshort * v1, const GLshort * v2);
typedef void (GLAPIENTRY *PFNGLRELEASESHADERCOMPILERPROC) ();
typedef void (GLAPIENTRY *PFNGLRENDERGPUMASKNVPROC) (GLbitfield  mask);
typedef GLint (GLAPIENTRY *PFNGLRENDERMODEPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLRENDERBUFFERSTORAGEPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLRENDERBUFFERSTORAGEEXTPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLRENDERBUFFERSTORAGEMULTISAMPLEADVANCEDAMDPROC) (GLenum  target, GLsizei  samples, GLsizei  storageSamples, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLRENDERBUFFERSTORAGEMULTISAMPLECOVERAGENVPROC) (GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLREQUESTRESIDENTPROGRAMSNVPROC) (GLsizei  n, const GLuint * programs);
typedef void (GLAPIENTRY *PFNGLRESETHISTOGRAMPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLRESETHISTOGRAMEXTPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLRESETMEMORYOBJECTPARAMETERNVPROC) (GLuint  memory, GLenum  pname);
typedef void (GLAPIENTRY *PFNGLRESETMINMAXPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLRESETMINMAXEXTPROC) (GLenum  target);
typedef void (GLAPIENTRY *PFNGLRESOLVEDEPTHVALUESNVPROC) ();
typedef void (GLAPIENTRY *PFNGLRESUMETRANSFORMFEEDBACKPROC) ();
typedef void (GLAPIENTRY *PFNGLRESUMETRANSFORMFEEDBACKNVPROC) ();
typedef void (GLAPIENTRY *PFNGLROTATEDPROC) (GLdouble  angle, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLROTATEFPROC) (GLfloat  angle, GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLSAMPLECOVERAGEPROC) (GLfloat  value, GLboolean  invert);
typedef void (GLAPIENTRY *PFNGLSAMPLECOVERAGEARBPROC) (GLfloat  value, GLboolean  invert);
typedef void (GLAPIENTRY *PFNGLSAMPLEMASKEXTPROC) (GLclampf  value, GLboolean  invert);
typedef void (GLAPIENTRY *PFNGLSAMPLEMASKINDEXEDNVPROC) (GLuint  index, GLbitfield  mask);
typedef void (GLAPIENTRY *PFNGLSAMPLEMASKIPROC) (GLuint  maskNumber, GLbitfield  mask);
typedef void (GLAPIENTRY *PFNGLSAMPLEPATTERNEXTPROC) (GLenum  pattern);
typedef void (GLAPIENTRY *PFNGLSAMPLERPARAMETERIIVPROC) (GLuint  sampler, GLenum  pname, const GLint * param);
typedef void (GLAPIENTRY *PFNGLSAMPLERPARAMETERIUIVPROC) (GLuint  sampler, GLenum  pname, const GLuint * param);
typedef void (GLAPIENTRY *PFNGLSAMPLERPARAMETERFPROC) (GLuint  sampler, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLSAMPLERPARAMETERFVPROC) (GLuint  sampler, GLenum  pname, const GLfloat * param);
typedef void (GLAPIENTRY *PFNGLSAMPLERPARAMETERIPROC) (GLuint  sampler, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLSAMPLERPARAMETERIVPROC) (GLuint  sampler, GLenum  pname, const GLint * param);
typedef void (GLAPIENTRY *PFNGLSCALEDPROC) (GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLSCALEFPROC) (GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLSCISSORPROC) (GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLSCISSORARRAYVPROC) (GLuint  first, GLsizei  count, const GLint * v);
typedef void (GLAPIENTRY *PFNGLSCISSOREXCLUSIVEARRAYVNVPROC) (GLuint  first, GLsizei  count, const GLint * v);
typedef void (GLAPIENTRY *PFNGLSCISSOREXCLUSIVENVPROC) (GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLSCISSORINDEXEDPROC) (GLuint  index, GLint  left, GLint  bottom, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLSCISSORINDEXEDVPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3BPROC) (GLbyte  red, GLbyte  green, GLbyte  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3BEXTPROC) (GLbyte  red, GLbyte  green, GLbyte  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3BVPROC) (const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3BVEXTPROC) (const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3DPROC) (GLdouble  red, GLdouble  green, GLdouble  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3DEXTPROC) (GLdouble  red, GLdouble  green, GLdouble  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3DVEXTPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3FPROC) (GLfloat  red, GLfloat  green, GLfloat  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3FEXTPROC) (GLfloat  red, GLfloat  green, GLfloat  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3FVEXTPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3HNVPROC) (GLhalfNV  red, GLhalfNV  green, GLhalfNV  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3IPROC) (GLint  red, GLint  green, GLint  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3IEXTPROC) (GLint  red, GLint  green, GLint  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3IVEXTPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3SPROC) (GLshort  red, GLshort  green, GLshort  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3SEXTPROC) (GLshort  red, GLshort  green, GLshort  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3SVEXTPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3UBPROC) (GLubyte  red, GLubyte  green, GLubyte  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3UBEXTPROC) (GLubyte  red, GLubyte  green, GLubyte  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3UBVPROC) (const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3UBVEXTPROC) (const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3UIPROC) (GLuint  red, GLuint  green, GLuint  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3UIEXTPROC) (GLuint  red, GLuint  green, GLuint  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3UIVPROC) (const GLuint * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3UIVEXTPROC) (const GLuint * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3USPROC) (GLushort  red, GLushort  green, GLushort  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3USEXTPROC) (GLushort  red, GLushort  green, GLushort  blue);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3USVPROC) (const GLushort * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLOR3USVEXTPROC) (const GLushort * v);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLORFORMATNVPROC) (GLint  size, GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLORP3UIPROC) (GLenum  type, GLuint  color);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLORP3UIVPROC) (GLenum  type, const GLuint * color);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLORPOINTERPROC) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLSECONDARYCOLORPOINTEREXTPROC) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLSELECTBUFFERPROC) (GLsizei  size, GLuint * buffer);
typedef void (GLAPIENTRY *PFNGLSELECTPERFMONITORCOUNTERSAMDPROC) (GLuint  monitor, GLboolean  enable, GLuint  group, GLint  numCounters, GLuint * counterList);
typedef void (GLAPIENTRY *PFNGLSEMAPHOREPARAMETERUI64VEXTPROC) (GLuint  semaphore, GLenum  pname, const GLuint64 * params);
typedef void (GLAPIENTRY *PFNGLSEPARABLEFILTER2DPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * row, const void * column);
typedef void (GLAPIENTRY *PFNGLSEPARABLEFILTER2DEXTPROC) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * row, const void * column);
typedef void (GLAPIENTRY *PFNGLSETFENCEAPPLEPROC) (GLuint  fence);
typedef void (GLAPIENTRY *PFNGLSETFENCENVPROC) (GLuint  fence, GLenum  condition);
typedef void (GLAPIENTRY *PFNGLSETINVARIANTEXTPROC) (GLuint  id, GLenum  type, const void * addr);
typedef void (GLAPIENTRY *PFNGLSETLOCALCONSTANTEXTPROC) (GLuint  id, GLenum  type, const void * addr);
typedef void (GLAPIENTRY *PFNGLSETMULTISAMPLEFVAMDPROC) (GLenum  pname, GLuint  index, const GLfloat * val);
typedef void (GLAPIENTRY *PFNGLSHADEMODELPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLSHADERBINARYPROC) (GLsizei  count, const GLuint * shaders, GLenum  binaryformat, const void * binary, GLsizei  length);
typedef void (GLAPIENTRY *PFNGLSHADEROP1EXTPROC) (GLenum  op, GLuint  res, GLuint  arg1);
typedef void (GLAPIENTRY *PFNGLSHADEROP2EXTPROC) (GLenum  op, GLuint  res, GLuint  arg1, GLuint  arg2);
typedef void (GLAPIENTRY *PFNGLSHADEROP3EXTPROC) (GLenum  op, GLuint  res, GLuint  arg1, GLuint  arg2, GLuint  arg3);
typedef void (GLAPIENTRY *PFNGLSHADERSOURCEPROC) (GLuint  shader, GLsizei  count, const GLchar *const* string, const GLint * length);
typedef void (GLAPIENTRY *PFNGLSHADERSOURCEARBPROC) (GLhandleARB  shaderObj, GLsizei  count, const GLcharARB ** string, const GLint * length);
typedef void (GLAPIENTRY *PFNGLSHADERSTORAGEBLOCKBINDINGPROC) (GLuint  program, GLuint  storageBlockIndex, GLuint  storageBlockBinding);
typedef void (GLAPIENTRY *PFNGLSHADINGRATEIMAGEBARRIERNVPROC) (GLboolean  synchronize);
typedef void (GLAPIENTRY *PFNGLSHADINGRATEIMAGEPALETTENVPROC) (GLuint  viewport, GLuint  first, GLsizei  count, const GLenum * rates);
typedef void (GLAPIENTRY *PFNGLSHADINGRATESAMPLEORDERNVPROC) (GLenum  order);
typedef void (GLAPIENTRY *PFNGLSHADINGRATESAMPLEORDERCUSTOMNVPROC) (GLenum  rate, GLuint  samples, const GLint * locations);
typedef void (GLAPIENTRY *PFNGLSIGNALSEMAPHOREEXTPROC) (GLuint  semaphore, GLuint  numBufferBarriers, const GLuint * buffers, GLuint  numTextureBarriers, const GLuint * textures, const GLenum * dstLayouts);
typedef void (GLAPIENTRY *PFNGLSPECIALIZESHADERPROC) (GLuint  shader, const GLchar * pEntryPoint, GLuint  numSpecializationConstants, const GLuint * pConstantIndex, const GLuint * pConstantValue);
typedef void (GLAPIENTRY *PFNGLSPECIALIZESHADERARBPROC) (GLuint  shader, const GLchar * pEntryPoint, GLuint  numSpecializationConstants, const GLuint * pConstantIndex, const GLuint * pConstantValue);
typedef void (GLAPIENTRY *PFNGLSTATECAPTURENVPROC) (GLuint  state, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLSTENCILCLEARTAGEXTPROC) (GLsizei  stencilTagBits, GLuint  stencilClearTag);
typedef void (GLAPIENTRY *PFNGLSTENCILFILLPATHINSTANCEDNVPROC) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLenum  fillMode, GLuint  mask, GLenum  transformType, const GLfloat * transformValues);
typedef void (GLAPIENTRY *PFNGLSTENCILFILLPATHNVPROC) (GLuint  path, GLenum  fillMode, GLuint  mask);
typedef void (GLAPIENTRY *PFNGLSTENCILFUNCPROC) (GLenum  func, GLint  ref, GLuint  mask);
typedef void (GLAPIENTRY *PFNGLSTENCILFUNCSEPARATEPROC) (GLenum  face, GLenum  func, GLint  ref, GLuint  mask);
typedef void (GLAPIENTRY *PFNGLSTENCILMASKPROC) (GLuint  mask);
typedef void (GLAPIENTRY *PFNGLSTENCILMASKSEPARATEPROC) (GLenum  face, GLuint  mask);
typedef void (GLAPIENTRY *PFNGLSTENCILOPPROC) (GLenum  fail, GLenum  zfail, GLenum  zpass);
typedef void (GLAPIENTRY *PFNGLSTENCILOPSEPARATEPROC) (GLenum  face, GLenum  sfail, GLenum  dpfail, GLenum  dppass);
typedef void (GLAPIENTRY *PFNGLSTENCILOPVALUEAMDPROC) (GLenum  face, GLuint  value);
typedef void (GLAPIENTRY *PFNGLSTENCILSTROKEPATHINSTANCEDNVPROC) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLint  reference, GLuint  mask, GLenum  transformType, const GLfloat * transformValues);
typedef void (GLAPIENTRY *PFNGLSTENCILSTROKEPATHNVPROC) (GLuint  path, GLint  reference, GLuint  mask);
typedef void (GLAPIENTRY *PFNGLSTENCILTHENCOVERFILLPATHINSTANCEDNVPROC) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLenum  fillMode, GLuint  mask, GLenum  coverMode, GLenum  transformType, const GLfloat * transformValues);
typedef void (GLAPIENTRY *PFNGLSTENCILTHENCOVERFILLPATHNVPROC) (GLuint  path, GLenum  fillMode, GLuint  mask, GLenum  coverMode);
typedef void (GLAPIENTRY *PFNGLSTENCILTHENCOVERSTROKEPATHINSTANCEDNVPROC) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLint  reference, GLuint  mask, GLenum  coverMode, GLenum  transformType, const GLfloat * transformValues);
typedef void (GLAPIENTRY *PFNGLSTENCILTHENCOVERSTROKEPATHNVPROC) (GLuint  path, GLint  reference, GLuint  mask, GLenum  coverMode);
typedef void (GLAPIENTRY *PFNGLSUBPIXELPRECISIONBIASNVPROC) (GLuint  xbits, GLuint  ybits);
typedef void (GLAPIENTRY *PFNGLSWIZZLEEXTPROC) (GLuint  res, GLuint  in, GLenum  outX, GLenum  outY, GLenum  outZ, GLenum  outW);
typedef void (GLAPIENTRY *PFNGLSYNCTEXTUREINTELPROC) (GLuint  texture);
typedef void (GLAPIENTRY *PFNGLTANGENT3BEXTPROC) (GLbyte  tx, GLbyte  ty, GLbyte  tz);
typedef void (GLAPIENTRY *PFNGLTANGENT3BVEXTPROC) (const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLTANGENT3DEXTPROC) (GLdouble  tx, GLdouble  ty, GLdouble  tz);
typedef void (GLAPIENTRY *PFNGLTANGENT3DVEXTPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLTANGENT3FEXTPROC) (GLfloat  tx, GLfloat  ty, GLfloat  tz);
typedef void (GLAPIENTRY *PFNGLTANGENT3FVEXTPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLTANGENT3IEXTPROC) (GLint  tx, GLint  ty, GLint  tz);
typedef void (GLAPIENTRY *PFNGLTANGENT3IVEXTPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLTANGENT3SEXTPROC) (GLshort  tx, GLshort  ty, GLshort  tz);
typedef void (GLAPIENTRY *PFNGLTANGENT3SVEXTPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLTANGENTPOINTEREXTPROC) (GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLTESSELLATIONFACTORAMDPROC) (GLfloat  factor);
typedef void (GLAPIENTRY *PFNGLTESSELLATIONMODEAMDPROC) (GLenum  mode);
typedef GLboolean (GLAPIENTRY *PFNGLTESTFENCEAPPLEPROC) (GLuint  fence);
typedef GLboolean (GLAPIENTRY *PFNGLTESTFENCENVPROC) (GLuint  fence);
typedef GLboolean (GLAPIENTRY *PFNGLTESTOBJECTAPPLEPROC) (GLenum  object, GLuint  name);
typedef void (GLAPIENTRY *PFNGLTEXATTACHMEMORYNVPROC) (GLenum  target, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXBUFFERPROC) (GLenum  target, GLenum  internalformat, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLTEXBUFFERARBPROC) (GLenum  target, GLenum  internalformat, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLTEXBUFFEREXTPROC) (GLenum  target, GLenum  internalformat, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLTEXBUFFERRANGEPROC) (GLenum  target, GLenum  internalformat, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1DPROC) (GLdouble  s);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1FPROC) (GLfloat  s);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1HNVPROC) (GLhalfNV  s);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1IPROC) (GLint  s);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1SPROC) (GLshort  s);
typedef void (GLAPIENTRY *PFNGLTEXCOORD1SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2DPROC) (GLdouble  s, GLdouble  t);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2FPROC) (GLfloat  s, GLfloat  t);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2HNVPROC) (GLhalfNV  s, GLhalfNV  t);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2IPROC) (GLint  s, GLint  t);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2SPROC) (GLshort  s, GLshort  t);
typedef void (GLAPIENTRY *PFNGLTEXCOORD2SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3DPROC) (GLdouble  s, GLdouble  t, GLdouble  r);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3FPROC) (GLfloat  s, GLfloat  t, GLfloat  r);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3HNVPROC) (GLhalfNV  s, GLhalfNV  t, GLhalfNV  r);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3IPROC) (GLint  s, GLint  t, GLint  r);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3SPROC) (GLshort  s, GLshort  t, GLshort  r);
typedef void (GLAPIENTRY *PFNGLTEXCOORD3SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4DPROC) (GLdouble  s, GLdouble  t, GLdouble  r, GLdouble  q);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4FPROC) (GLfloat  s, GLfloat  t, GLfloat  r, GLfloat  q);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4HNVPROC) (GLhalfNV  s, GLhalfNV  t, GLhalfNV  r, GLhalfNV  q);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4IPROC) (GLint  s, GLint  t, GLint  r, GLint  q);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4SPROC) (GLshort  s, GLshort  t, GLshort  r, GLshort  q);
typedef void (GLAPIENTRY *PFNGLTEXCOORD4SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLTEXCOORDFORMATNVPROC) (GLint  size, GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLTEXCOORDP1UIPROC) (GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLTEXCOORDP1UIVPROC) (GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLTEXCOORDP2UIPROC) (GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLTEXCOORDP2UIVPROC) (GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLTEXCOORDP3UIPROC) (GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLTEXCOORDP3UIVPROC) (GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLTEXCOORDP4UIPROC) (GLenum  type, GLuint  coords);
typedef void (GLAPIENTRY *PFNGLTEXCOORDP4UIVPROC) (GLenum  type, const GLuint * coords);
typedef void (GLAPIENTRY *PFNGLTEXCOORDPOINTERPROC) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLTEXCOORDPOINTEREXTPROC) (GLint  size, GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
typedef void (GLAPIENTRY *PFNGLTEXCOORDPOINTERVINTELPROC) (GLint  size, GLenum  type, const void ** pointer);
typedef void (GLAPIENTRY *PFNGLTEXENVFPROC) (GLenum  target, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLTEXENVFVPROC) (GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLTEXENVIPROC) (GLenum  target, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLTEXENVIVPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLTEXGENDPROC) (GLenum  coord, GLenum  pname, GLdouble  param);
typedef void (GLAPIENTRY *PFNGLTEXGENDVPROC) (GLenum  coord, GLenum  pname, const GLdouble * params);
typedef void (GLAPIENTRY *PFNGLTEXGENFPROC) (GLenum  coord, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLTEXGENFVPROC) (GLenum  coord, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLTEXGENIPROC) (GLenum  coord, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLTEXGENIVPROC) (GLenum  coord, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLTEXIMAGE1DPROC) (GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXIMAGE2DPROC) (GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXIMAGE2DMULTISAMPLEPROC) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLTEXIMAGE2DMULTISAMPLECOVERAGENVPROC) (GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations);
typedef void (GLAPIENTRY *PFNGLTEXIMAGE3DPROC) (GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXIMAGE3DEXTPROC) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXIMAGE3DMULTISAMPLEPROC) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLTEXIMAGE3DMULTISAMPLECOVERAGENVPROC) (GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations);
typedef void (GLAPIENTRY *PFNGLTEXPAGECOMMITMENTARBPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  commit);
typedef void (GLAPIENTRY *PFNGLTEXPARAMETERIIVPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLTEXPARAMETERIIVEXTPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLTEXPARAMETERIUIVPROC) (GLenum  target, GLenum  pname, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLTEXPARAMETERIUIVEXTPROC) (GLenum  target, GLenum  pname, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLTEXPARAMETERFPROC) (GLenum  target, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLTEXPARAMETERFVPROC) (GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLTEXPARAMETERIPROC) (GLenum  target, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLTEXPARAMETERIVPROC) (GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLTEXRENDERBUFFERNVPROC) (GLenum  target, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGE1DPROC) (GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGE2DPROC) (GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGE2DMULTISAMPLEPROC) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGE3DPROC) (GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGE3DMULTISAMPLEPROC) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGEMEM1DEXTPROC) (GLenum  target, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGEMEM2DEXTPROC) (GLenum  target, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGEMEM2DMULTISAMPLEEXTPROC) (GLenum  target, GLsizei  samples, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGEMEM3DEXTPROC) (GLenum  target, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGEMEM3DMULTISAMPLEEXTPROC) (GLenum  target, GLsizei  samples, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXSTORAGESPARSEAMDPROC) (GLenum  target, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLsizei  layers, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLTEXSUBIMAGE1DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXSUBIMAGE1DEXTPROC) (GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXSUBIMAGE2DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXSUBIMAGE2DEXTPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXSUBIMAGE3DPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXSUBIMAGE3DEXTPROC) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTUREATTACHMEMORYNVPROC) (GLuint  texture, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXTUREBARRIERPROC) ();
typedef void (GLAPIENTRY *PFNGLTEXTUREBARRIERNVPROC) ();
typedef void (GLAPIENTRY *PFNGLTEXTUREBUFFERPROC) (GLuint  texture, GLenum  internalformat, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLTEXTUREBUFFEREXTPROC) (GLuint  texture, GLenum  target, GLenum  internalformat, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLTEXTUREBUFFERRANGEPROC) (GLuint  texture, GLenum  internalformat, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLTEXTUREBUFFERRANGEEXTPROC) (GLuint  texture, GLenum  target, GLenum  internalformat, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLTEXTUREIMAGE1DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTUREIMAGE2DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTUREIMAGE2DMULTISAMPLECOVERAGENVPROC) (GLuint  texture, GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations);
typedef void (GLAPIENTRY *PFNGLTEXTUREIMAGE2DMULTISAMPLENVPROC) (GLuint  texture, GLenum  target, GLsizei  samples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations);
typedef void (GLAPIENTRY *PFNGLTEXTUREIMAGE3DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTUREIMAGE3DMULTISAMPLECOVERAGENVPROC) (GLuint  texture, GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations);
typedef void (GLAPIENTRY *PFNGLTEXTUREIMAGE3DMULTISAMPLENVPROC) (GLuint  texture, GLenum  target, GLsizei  samples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations);
typedef void (GLAPIENTRY *PFNGLTEXTURELIGHTEXTPROC) (GLenum  pname);
typedef void (GLAPIENTRY *PFNGLTEXTUREMATERIALEXTPROC) (GLenum  face, GLenum  mode);
typedef void (GLAPIENTRY *PFNGLTEXTURENORMALEXTPROC) (GLenum  mode);
typedef void (GLAPIENTRY *PFNGLTEXTUREPAGECOMMITMENTEXTPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  commit);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERIIVPROC) (GLuint  texture, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERIIVEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERIUIVPROC) (GLuint  texture, GLenum  pname, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERIUIVEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, const GLuint * params);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERFPROC) (GLuint  texture, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERFEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, GLfloat  param);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERFVPROC) (GLuint  texture, GLenum  pname, const GLfloat * param);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERFVEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERIPROC) (GLuint  texture, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERIEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERIVPROC) (GLuint  texture, GLenum  pname, const GLint * param);
typedef void (GLAPIENTRY *PFNGLTEXTUREPARAMETERIVEXTPROC) (GLuint  texture, GLenum  target, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLTEXTURERANGEAPPLEPROC) (GLenum  target, GLsizei  length, const void * pointer);
typedef void (GLAPIENTRY *PFNGLTEXTURERENDERBUFFEREXTPROC) (GLuint  texture, GLenum  target, GLuint  renderbuffer);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE1DPROC) (GLuint  texture, GLsizei  levels, GLenum  internalformat, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE1DEXTPROC) (GLuint  texture, GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE2DPROC) (GLuint  texture, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE2DEXTPROC) (GLuint  texture, GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC) (GLuint  texture, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC) (GLuint  texture, GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE3DPROC) (GLuint  texture, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE3DEXTPROC) (GLuint  texture, GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC) (GLuint  texture, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC) (GLuint  texture, GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedsamplelocations);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGEMEM1DEXTPROC) (GLuint  texture, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGEMEM2DEXTPROC) (GLuint  texture, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGEMEM2DMULTISAMPLEEXTPROC) (GLuint  texture, GLsizei  samples, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGEMEM3DEXTPROC) (GLuint  texture, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGEMEM3DMULTISAMPLEEXTPROC) (GLuint  texture, GLsizei  samples, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations, GLuint  memory, GLuint64  offset);
typedef void (GLAPIENTRY *PFNGLTEXTURESTORAGESPARSEAMDPROC) (GLuint  texture, GLenum  target, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLsizei  layers, GLbitfield  flags);
typedef void (GLAPIENTRY *PFNGLTEXTURESUBIMAGE1DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTURESUBIMAGE1DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTURESUBIMAGE2DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTURESUBIMAGE2DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTURESUBIMAGE3DPROC) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTURESUBIMAGE3DEXTPROC) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
typedef void (GLAPIENTRY *PFNGLTEXTUREVIEWPROC) (GLuint  texture, GLenum  target, GLuint  origtexture, GLenum  internalformat, GLuint  minlevel, GLuint  numlevels, GLuint  minlayer, GLuint  numlayers);
typedef void (GLAPIENTRY *PFNGLTRACKMATRIXNVPROC) (GLenum  target, GLuint  address, GLenum  matrix, GLenum  transform);
typedef void (GLAPIENTRY *PFNGLTRANSFORMFEEDBACKATTRIBSNVPROC) (GLsizei  count, const GLint * attribs, GLenum  bufferMode);
typedef void (GLAPIENTRY *PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC) (GLuint  xfb, GLuint  index, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC) (GLuint  xfb, GLuint  index, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
typedef void (GLAPIENTRY *PFNGLTRANSFORMFEEDBACKSTREAMATTRIBSNVPROC) (GLsizei  count, const GLint * attribs, GLsizei  nbuffers, const GLint * bufstreams, GLenum  bufferMode);
typedef void (GLAPIENTRY *PFNGLTRANSFORMFEEDBACKVARYINGSPROC) (GLuint  program, GLsizei  count, const GLchar *const* varyings, GLenum  bufferMode);
typedef void (GLAPIENTRY *PFNGLTRANSFORMFEEDBACKVARYINGSEXTPROC) (GLuint  program, GLsizei  count, const GLchar *const* varyings, GLenum  bufferMode);
typedef void (GLAPIENTRY *PFNGLTRANSFORMFEEDBACKVARYINGSNVPROC) (GLuint  program, GLsizei  count, const GLint * locations, GLenum  bufferMode);
typedef void (GLAPIENTRY *PFNGLTRANSFORMPATHNVPROC) (GLuint  resultPath, GLuint  srcPath, GLenum  transformType, const GLfloat * transformValues);
typedef void (GLAPIENTRY *PFNGLTRANSLATEDPROC) (GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLTRANSLATEFPROC) (GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLUNIFORM1DPROC) (GLint  location, GLdouble  x);
typedef void (GLAPIENTRY *PFNGLUNIFORM1DVPROC) (GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1FPROC) (GLint  location, GLfloat  v0);
typedef void (GLAPIENTRY *PFNGLUNIFORM1FARBPROC) (GLint  location, GLfloat  v0);
typedef void (GLAPIENTRY *PFNGLUNIFORM1FVPROC) (GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1FVARBPROC) (GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1IPROC) (GLint  location, GLint  v0);
typedef void (GLAPIENTRY *PFNGLUNIFORM1I64ARBPROC) (GLint  location, GLint64  x);
typedef void (GLAPIENTRY *PFNGLUNIFORM1I64NVPROC) (GLint  location, GLint64EXT  x);
typedef void (GLAPIENTRY *PFNGLUNIFORM1I64VARBPROC) (GLint  location, GLsizei  count, const GLint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1I64VNVPROC) (GLint  location, GLsizei  count, const GLint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1IARBPROC) (GLint  location, GLint  v0);
typedef void (GLAPIENTRY *PFNGLUNIFORM1IVPROC) (GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1IVARBPROC) (GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1UIPROC) (GLint  location, GLuint  v0);
typedef void (GLAPIENTRY *PFNGLUNIFORM1UI64ARBPROC) (GLint  location, GLuint64  x);
typedef void (GLAPIENTRY *PFNGLUNIFORM1UI64NVPROC) (GLint  location, GLuint64EXT  x);
typedef void (GLAPIENTRY *PFNGLUNIFORM1UI64VARBPROC) (GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1UI64VNVPROC) (GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1UIEXTPROC) (GLint  location, GLuint  v0);
typedef void (GLAPIENTRY *PFNGLUNIFORM1UIVPROC) (GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM1UIVEXTPROC) (GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2DPROC) (GLint  location, GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLUNIFORM2DVPROC) (GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2FPROC) (GLint  location, GLfloat  v0, GLfloat  v1);
typedef void (GLAPIENTRY *PFNGLUNIFORM2FARBPROC) (GLint  location, GLfloat  v0, GLfloat  v1);
typedef void (GLAPIENTRY *PFNGLUNIFORM2FVPROC) (GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2FVARBPROC) (GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2IPROC) (GLint  location, GLint  v0, GLint  v1);
typedef void (GLAPIENTRY *PFNGLUNIFORM2I64ARBPROC) (GLint  location, GLint64  x, GLint64  y);
typedef void (GLAPIENTRY *PFNGLUNIFORM2I64NVPROC) (GLint  location, GLint64EXT  x, GLint64EXT  y);
typedef void (GLAPIENTRY *PFNGLUNIFORM2I64VARBPROC) (GLint  location, GLsizei  count, const GLint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2I64VNVPROC) (GLint  location, GLsizei  count, const GLint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2IARBPROC) (GLint  location, GLint  v0, GLint  v1);
typedef void (GLAPIENTRY *PFNGLUNIFORM2IVPROC) (GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2IVARBPROC) (GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2UIPROC) (GLint  location, GLuint  v0, GLuint  v1);
typedef void (GLAPIENTRY *PFNGLUNIFORM2UI64ARBPROC) (GLint  location, GLuint64  x, GLuint64  y);
typedef void (GLAPIENTRY *PFNGLUNIFORM2UI64NVPROC) (GLint  location, GLuint64EXT  x, GLuint64EXT  y);
typedef void (GLAPIENTRY *PFNGLUNIFORM2UI64VARBPROC) (GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2UI64VNVPROC) (GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2UIEXTPROC) (GLint  location, GLuint  v0, GLuint  v1);
typedef void (GLAPIENTRY *PFNGLUNIFORM2UIVPROC) (GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM2UIVEXTPROC) (GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3DPROC) (GLint  location, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLUNIFORM3DVPROC) (GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3FPROC) (GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2);
typedef void (GLAPIENTRY *PFNGLUNIFORM3FARBPROC) (GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2);
typedef void (GLAPIENTRY *PFNGLUNIFORM3FVPROC) (GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3FVARBPROC) (GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3IPROC) (GLint  location, GLint  v0, GLint  v1, GLint  v2);
typedef void (GLAPIENTRY *PFNGLUNIFORM3I64ARBPROC) (GLint  location, GLint64  x, GLint64  y, GLint64  z);
typedef void (GLAPIENTRY *PFNGLUNIFORM3I64NVPROC) (GLint  location, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z);
typedef void (GLAPIENTRY *PFNGLUNIFORM3I64VARBPROC) (GLint  location, GLsizei  count, const GLint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3I64VNVPROC) (GLint  location, GLsizei  count, const GLint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3IARBPROC) (GLint  location, GLint  v0, GLint  v1, GLint  v2);
typedef void (GLAPIENTRY *PFNGLUNIFORM3IVPROC) (GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3IVARBPROC) (GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3UIPROC) (GLint  location, GLuint  v0, GLuint  v1, GLuint  v2);
typedef void (GLAPIENTRY *PFNGLUNIFORM3UI64ARBPROC) (GLint  location, GLuint64  x, GLuint64  y, GLuint64  z);
typedef void (GLAPIENTRY *PFNGLUNIFORM3UI64NVPROC) (GLint  location, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z);
typedef void (GLAPIENTRY *PFNGLUNIFORM3UI64VARBPROC) (GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3UI64VNVPROC) (GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3UIEXTPROC) (GLint  location, GLuint  v0, GLuint  v1, GLuint  v2);
typedef void (GLAPIENTRY *PFNGLUNIFORM3UIVPROC) (GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM3UIVEXTPROC) (GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4DPROC) (GLint  location, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLUNIFORM4DVPROC) (GLint  location, GLsizei  count, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4FPROC) (GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2, GLfloat  v3);
typedef void (GLAPIENTRY *PFNGLUNIFORM4FARBPROC) (GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2, GLfloat  v3);
typedef void (GLAPIENTRY *PFNGLUNIFORM4FVPROC) (GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4FVARBPROC) (GLint  location, GLsizei  count, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4IPROC) (GLint  location, GLint  v0, GLint  v1, GLint  v2, GLint  v3);
typedef void (GLAPIENTRY *PFNGLUNIFORM4I64ARBPROC) (GLint  location, GLint64  x, GLint64  y, GLint64  z, GLint64  w);
typedef void (GLAPIENTRY *PFNGLUNIFORM4I64NVPROC) (GLint  location, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z, GLint64EXT  w);
typedef void (GLAPIENTRY *PFNGLUNIFORM4I64VARBPROC) (GLint  location, GLsizei  count, const GLint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4I64VNVPROC) (GLint  location, GLsizei  count, const GLint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4IARBPROC) (GLint  location, GLint  v0, GLint  v1, GLint  v2, GLint  v3);
typedef void (GLAPIENTRY *PFNGLUNIFORM4IVPROC) (GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4IVARBPROC) (GLint  location, GLsizei  count, const GLint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4UIPROC) (GLint  location, GLuint  v0, GLuint  v1, GLuint  v2, GLuint  v3);
typedef void (GLAPIENTRY *PFNGLUNIFORM4UI64ARBPROC) (GLint  location, GLuint64  x, GLuint64  y, GLuint64  z, GLuint64  w);
typedef void (GLAPIENTRY *PFNGLUNIFORM4UI64NVPROC) (GLint  location, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z, GLuint64EXT  w);
typedef void (GLAPIENTRY *PFNGLUNIFORM4UI64VARBPROC) (GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4UI64VNVPROC) (GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4UIEXTPROC) (GLint  location, GLuint  v0, GLuint  v1, GLuint  v2, GLuint  v3);
typedef void (GLAPIENTRY *PFNGLUNIFORM4UIVPROC) (GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORM4UIVEXTPROC) (GLint  location, GLsizei  count, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMBLOCKBINDINGPROC) (GLuint  program, GLuint  uniformBlockIndex, GLuint  uniformBlockBinding);
typedef void (GLAPIENTRY *PFNGLUNIFORMBUFFEREXTPROC) (GLuint  program, GLint  location, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLUNIFORMHANDLEUI64ARBPROC) (GLint  location, GLuint64  value);
typedef void (GLAPIENTRY *PFNGLUNIFORMHANDLEUI64NVPROC) (GLint  location, GLuint64  value);
typedef void (GLAPIENTRY *PFNGLUNIFORMHANDLEUI64VARBPROC) (GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMHANDLEUI64VNVPROC) (GLint  location, GLsizei  count, const GLuint64 * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX2DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX2FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX2FVARBPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX2X3DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX2X3FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX2X4DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX2X4FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX3DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX3FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX3FVARBPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX3X2DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX3X2FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX3X4DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX3X4FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX4DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX4FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX4FVARBPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX4X2DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX4X2FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX4X3DVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMMATRIX4X3FVPROC) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
typedef void (GLAPIENTRY *PFNGLUNIFORMSUBROUTINESUIVPROC) (GLenum  shadertype, GLsizei  count, const GLuint * indices);
typedef void (GLAPIENTRY *PFNGLUNIFORMUI64NVPROC) (GLint  location, GLuint64EXT  value);
typedef void (GLAPIENTRY *PFNGLUNIFORMUI64VNVPROC) (GLint  location, GLsizei  count, const GLuint64EXT * value);
typedef void (GLAPIENTRY *PFNGLUNLOCKARRAYSEXTPROC) ();
typedef GLboolean (GLAPIENTRY *PFNGLUNMAPBUFFERPROC) (GLenum  target);
typedef GLboolean (GLAPIENTRY *PFNGLUNMAPBUFFERARBPROC) (GLenum  target);
typedef GLboolean (GLAPIENTRY *PFNGLUNMAPNAMEDBUFFERPROC) (GLuint  buffer);
typedef GLboolean (GLAPIENTRY *PFNGLUNMAPNAMEDBUFFEREXTPROC) (GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLUNMAPTEXTURE2DINTELPROC) (GLuint  texture, GLint  level);
typedef void (GLAPIENTRY *PFNGLUSEPROGRAMPROC) (GLuint  program);
typedef void (GLAPIENTRY *PFNGLUSEPROGRAMOBJECTARBPROC) (GLhandleARB  programObj);
typedef void (GLAPIENTRY *PFNGLUSEPROGRAMSTAGESPROC) (GLuint  pipeline, GLbitfield  stages, GLuint  program);
typedef void (GLAPIENTRY *PFNGLUSEPROGRAMSTAGESEXTPROC) (GLuint  pipeline, GLbitfield  stages, GLuint  program);
typedef void (GLAPIENTRY *PFNGLUSESHADERPROGRAMEXTPROC) (GLenum  type, GLuint  program);
typedef void (GLAPIENTRY *PFNGLVDPAUFININVPROC) ();
typedef void (GLAPIENTRY *PFNGLVDPAUGETSURFACEIVNVPROC) (GLvdpauSurfaceNV  surface, GLenum  pname, GLsizei  count, GLsizei * length, GLint * values);
typedef void (GLAPIENTRY *PFNGLVDPAUINITNVPROC) (const void * vdpDevice, const void * getProcAddress);
typedef GLboolean (GLAPIENTRY *PFNGLVDPAUISSURFACENVPROC) (GLvdpauSurfaceNV  surface);
typedef void (GLAPIENTRY *PFNGLVDPAUMAPSURFACESNVPROC) (GLsizei  numSurfaces, const GLvdpauSurfaceNV * surfaces);
typedef GLvdpauSurfaceNV (GLAPIENTRY *PFNGLVDPAUREGISTEROUTPUTSURFACENVPROC) (const void * vdpSurface, GLenum  target, GLsizei  numTextureNames, const GLuint * textureNames);
typedef GLvdpauSurfaceNV (GLAPIENTRY *PFNGLVDPAUREGISTERVIDEOSURFACENVPROC) (const void * vdpSurface, GLenum  target, GLsizei  numTextureNames, const GLuint * textureNames);
typedef GLvdpauSurfaceNV (GLAPIENTRY *PFNGLVDPAUREGISTERVIDEOSURFACEWITHPICTURESTRUCTURENVPROC) (const void * vdpSurface, GLenum  target, GLsizei  numTextureNames, const GLuint * textureNames, GLboolean  isFrameStructure);
typedef void (GLAPIENTRY *PFNGLVDPAUSURFACEACCESSNVPROC) (GLvdpauSurfaceNV  surface, GLenum  access);
typedef void (GLAPIENTRY *PFNGLVDPAUUNMAPSURFACESNVPROC) (GLsizei  numSurface, const GLvdpauSurfaceNV * surfaces);
typedef void (GLAPIENTRY *PFNGLVDPAUUNREGISTERSURFACENVPROC) (GLvdpauSurfaceNV  surface);
typedef void (GLAPIENTRY *PFNGLVALIDATEPROGRAMPROC) (GLuint  program);
typedef void (GLAPIENTRY *PFNGLVALIDATEPROGRAMARBPROC) (GLhandleARB  programObj);
typedef void (GLAPIENTRY *PFNGLVALIDATEPROGRAMPIPELINEPROC) (GLuint  pipeline);
typedef void (GLAPIENTRY *PFNGLVALIDATEPROGRAMPIPELINEEXTPROC) (GLuint  pipeline);
typedef void (GLAPIENTRY *PFNGLVARIANTPOINTEREXTPROC) (GLuint  id, GLenum  type, GLuint  stride, const void * addr);
typedef void (GLAPIENTRY *PFNGLVARIANTBVEXTPROC) (GLuint  id, const GLbyte * addr);
typedef void (GLAPIENTRY *PFNGLVARIANTDVEXTPROC) (GLuint  id, const GLdouble * addr);
typedef void (GLAPIENTRY *PFNGLVARIANTFVEXTPROC) (GLuint  id, const GLfloat * addr);
typedef void (GLAPIENTRY *PFNGLVARIANTIVEXTPROC) (GLuint  id, const GLint * addr);
typedef void (GLAPIENTRY *PFNGLVARIANTSVEXTPROC) (GLuint  id, const GLshort * addr);
typedef void (GLAPIENTRY *PFNGLVARIANTUBVEXTPROC) (GLuint  id, const GLubyte * addr);
typedef void (GLAPIENTRY *PFNGLVARIANTUIVEXTPROC) (GLuint  id, const GLuint * addr);
typedef void (GLAPIENTRY *PFNGLVARIANTUSVEXTPROC) (GLuint  id, const GLushort * addr);
typedef void (GLAPIENTRY *PFNGLVERTEX2DPROC) (GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLVERTEX2DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEX2FPROC) (GLfloat  x, GLfloat  y);
typedef void (GLAPIENTRY *PFNGLVERTEX2FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEX2HNVPROC) (GLhalfNV  x, GLhalfNV  y);
typedef void (GLAPIENTRY *PFNGLVERTEX2HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEX2IPROC) (GLint  x, GLint  y);
typedef void (GLAPIENTRY *PFNGLVERTEX2IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEX2SPROC) (GLshort  x, GLshort  y);
typedef void (GLAPIENTRY *PFNGLVERTEX2SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEX3DPROC) (GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLVERTEX3DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEX3FPROC) (GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLVERTEX3FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEX3HNVPROC) (GLhalfNV  x, GLhalfNV  y, GLhalfNV  z);
typedef void (GLAPIENTRY *PFNGLVERTEX3HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEX3IPROC) (GLint  x, GLint  y, GLint  z);
typedef void (GLAPIENTRY *PFNGLVERTEX3IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEX3SPROC) (GLshort  x, GLshort  y, GLshort  z);
typedef void (GLAPIENTRY *PFNGLVERTEX3SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEX4DPROC) (GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLVERTEX4DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEX4FPROC) (GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLVERTEX4FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEX4HNVPROC) (GLhalfNV  x, GLhalfNV  y, GLhalfNV  z, GLhalfNV  w);
typedef void (GLAPIENTRY *PFNGLVERTEX4HVNVPROC) (const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEX4IPROC) (GLint  x, GLint  y, GLint  z, GLint  w);
typedef void (GLAPIENTRY *PFNGLVERTEX4IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEX4SPROC) (GLshort  x, GLshort  y, GLshort  z, GLshort  w);
typedef void (GLAPIENTRY *PFNGLVERTEX4SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYATTRIBBINDINGPROC) (GLuint  vaobj, GLuint  attribindex, GLuint  bindingindex);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYATTRIBFORMATPROC) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLboolean  normalized, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYATTRIBIFORMATPROC) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYATTRIBLFORMATPROC) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC) (GLuint  vaobj, GLuint  bindingindex, GLuint  buffer, GLintptr  offset, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYBINDINGDIVISORPROC) (GLuint  vaobj, GLuint  bindingindex, GLuint  divisor);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYCOLOROFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYEDGEFLAGOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYELEMENTBUFFERPROC) (GLuint  vaobj, GLuint  buffer);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYFOGCOORDOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYINDEXOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYMULTITEXCOORDOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLenum  texunit, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYNORMALOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYPARAMETERIAPPLEPROC) (GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYRANGEAPPLEPROC) (GLsizei  length, void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYRANGENVPROC) (GLsizei  length, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYSECONDARYCOLOROFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYTEXCOORDOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC) (GLuint  vaobj, GLuint  attribindex, GLuint  bindingindex);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXATTRIBDIVISOREXTPROC) (GLuint  vaobj, GLuint  index, GLuint  divisor);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLboolean  normalized, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXATTRIBIOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLuint  index, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXATTRIBLOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLuint  index, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXATTRIBOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLuint  index, GLint  size, GLenum  type, GLboolean  normalized, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC) (GLuint  vaobj, GLuint  bindingindex, GLuint  divisor);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXBUFFERPROC) (GLuint  vaobj, GLuint  bindingindex, GLuint  buffer, GLintptr  offset, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXBUFFERSPROC) (GLuint  vaobj, GLuint  first, GLsizei  count, const GLuint * buffers, const GLintptr * offsets, const GLsizei * strides);
typedef void (GLAPIENTRY *PFNGLVERTEXARRAYVERTEXOFFSETEXTPROC) (GLuint  vaobj, GLuint  buffer, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1DPROC) (GLuint  index, GLdouble  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1DARBPROC) (GLuint  index, GLdouble  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1DNVPROC) (GLuint  index, GLdouble  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1DVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1DVARBPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1DVNVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1FPROC) (GLuint  index, GLfloat  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1FARBPROC) (GLuint  index, GLfloat  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1FNVPROC) (GLuint  index, GLfloat  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1FVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1FVARBPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1FVNVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1HNVPROC) (GLuint  index, GLhalfNV  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1HVNVPROC) (GLuint  index, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1SPROC) (GLuint  index, GLshort  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1SARBPROC) (GLuint  index, GLshort  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1SNVPROC) (GLuint  index, GLshort  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1SVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1SVARBPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB1SVNVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2DPROC) (GLuint  index, GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2DARBPROC) (GLuint  index, GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2DNVPROC) (GLuint  index, GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2DVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2DVARBPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2DVNVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2FPROC) (GLuint  index, GLfloat  x, GLfloat  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2FARBPROC) (GLuint  index, GLfloat  x, GLfloat  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2FNVPROC) (GLuint  index, GLfloat  x, GLfloat  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2FVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2FVARBPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2FVNVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2HNVPROC) (GLuint  index, GLhalfNV  x, GLhalfNV  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2HVNVPROC) (GLuint  index, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2SPROC) (GLuint  index, GLshort  x, GLshort  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2SARBPROC) (GLuint  index, GLshort  x, GLshort  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2SNVPROC) (GLuint  index, GLshort  x, GLshort  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2SVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2SVARBPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB2SVNVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3DPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3DARBPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3DNVPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3DVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3DVARBPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3DVNVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3FPROC) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3FARBPROC) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3FNVPROC) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3FVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3FVARBPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3FVNVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3HNVPROC) (GLuint  index, GLhalfNV  x, GLhalfNV  y, GLhalfNV  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3HVNVPROC) (GLuint  index, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3SPROC) (GLuint  index, GLshort  x, GLshort  y, GLshort  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3SARBPROC) (GLuint  index, GLshort  x, GLshort  y, GLshort  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3SNVPROC) (GLuint  index, GLshort  x, GLshort  y, GLshort  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3SVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3SVARBPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB3SVNVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NBVPROC) (GLuint  index, const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NBVARBPROC) (GLuint  index, const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NIVPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NIVARBPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NSVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NSVARBPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NUBPROC) (GLuint  index, GLubyte  x, GLubyte  y, GLubyte  z, GLubyte  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NUBARBPROC) (GLuint  index, GLubyte  x, GLubyte  y, GLubyte  z, GLubyte  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NUBVPROC) (GLuint  index, const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NUBVARBPROC) (GLuint  index, const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NUIVPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NUIVARBPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NUSVPROC) (GLuint  index, const GLushort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4NUSVARBPROC) (GLuint  index, const GLushort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4BVPROC) (GLuint  index, const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4BVARBPROC) (GLuint  index, const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4DPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4DARBPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4DNVPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4DVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4DVARBPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4DVNVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4FPROC) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4FARBPROC) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4FNVPROC) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4FVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4FVARBPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4FVNVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4HNVPROC) (GLuint  index, GLhalfNV  x, GLhalfNV  y, GLhalfNV  z, GLhalfNV  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4HVNVPROC) (GLuint  index, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4IVPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4IVARBPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4SPROC) (GLuint  index, GLshort  x, GLshort  y, GLshort  z, GLshort  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4SARBPROC) (GLuint  index, GLshort  x, GLshort  y, GLshort  z, GLshort  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4SNVPROC) (GLuint  index, GLshort  x, GLshort  y, GLshort  z, GLshort  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4SVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4SVARBPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4SVNVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4UBNVPROC) (GLuint  index, GLubyte  x, GLubyte  y, GLubyte  z, GLubyte  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4UBVPROC) (GLuint  index, const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4UBVARBPROC) (GLuint  index, const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4UBVNVPROC) (GLuint  index, const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4UIVPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4UIVARBPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4USVPROC) (GLuint  index, const GLushort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIB4USVARBPROC) (GLuint  index, const GLushort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBBINDINGPROC) (GLuint  attribindex, GLuint  bindingindex);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBDIVISORPROC) (GLuint  index, GLuint  divisor);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBDIVISORARBPROC) (GLuint  index, GLuint  divisor);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBFORMATPROC) (GLuint  attribindex, GLint  size, GLenum  type, GLboolean  normalized, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBFORMATNVPROC) (GLuint  index, GLint  size, GLenum  type, GLboolean  normalized, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI1IPROC) (GLuint  index, GLint  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI1IEXTPROC) (GLuint  index, GLint  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI1IVPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI1IVEXTPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI1UIPROC) (GLuint  index, GLuint  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI1UIEXTPROC) (GLuint  index, GLuint  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI1UIVPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI1UIVEXTPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI2IPROC) (GLuint  index, GLint  x, GLint  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI2IEXTPROC) (GLuint  index, GLint  x, GLint  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI2IVPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI2IVEXTPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI2UIPROC) (GLuint  index, GLuint  x, GLuint  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI2UIEXTPROC) (GLuint  index, GLuint  x, GLuint  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI2UIVPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI2UIVEXTPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI3IPROC) (GLuint  index, GLint  x, GLint  y, GLint  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI3IEXTPROC) (GLuint  index, GLint  x, GLint  y, GLint  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI3IVPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI3IVEXTPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI3UIPROC) (GLuint  index, GLuint  x, GLuint  y, GLuint  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI3UIEXTPROC) (GLuint  index, GLuint  x, GLuint  y, GLuint  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI3UIVPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI3UIVEXTPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4BVPROC) (GLuint  index, const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4BVEXTPROC) (GLuint  index, const GLbyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4IPROC) (GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4IEXTPROC) (GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4IVPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4IVEXTPROC) (GLuint  index, const GLint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4SVPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4SVEXTPROC) (GLuint  index, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4UBVPROC) (GLuint  index, const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4UBVEXTPROC) (GLuint  index, const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4UIPROC) (GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4UIEXTPROC) (GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4UIVPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4UIVEXTPROC) (GLuint  index, const GLuint * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4USVPROC) (GLuint  index, const GLushort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBI4USVEXTPROC) (GLuint  index, const GLushort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBIFORMATPROC) (GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBIFORMATNVPROC) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBIPOINTERPROC) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBIPOINTEREXTPROC) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1DPROC) (GLuint  index, GLdouble  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1DEXTPROC) (GLuint  index, GLdouble  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1DVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1DVEXTPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1I64NVPROC) (GLuint  index, GLint64EXT  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1I64VNVPROC) (GLuint  index, const GLint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1UI64ARBPROC) (GLuint  index, GLuint64EXT  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1UI64NVPROC) (GLuint  index, GLuint64EXT  x);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1UI64VARBPROC) (GLuint  index, const GLuint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL1UI64VNVPROC) (GLuint  index, const GLuint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL2DPROC) (GLuint  index, GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL2DEXTPROC) (GLuint  index, GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL2DVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL2DVEXTPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL2I64NVPROC) (GLuint  index, GLint64EXT  x, GLint64EXT  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL2I64VNVPROC) (GLuint  index, const GLint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL2UI64NVPROC) (GLuint  index, GLuint64EXT  x, GLuint64EXT  y);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL2UI64VNVPROC) (GLuint  index, const GLuint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL3DPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL3DEXTPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL3DVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL3DVEXTPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL3I64NVPROC) (GLuint  index, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL3I64VNVPROC) (GLuint  index, const GLint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL3UI64NVPROC) (GLuint  index, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL3UI64VNVPROC) (GLuint  index, const GLuint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL4DPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL4DEXTPROC) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL4DVPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL4DVEXTPROC) (GLuint  index, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL4I64NVPROC) (GLuint  index, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z, GLint64EXT  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL4I64VNVPROC) (GLuint  index, const GLint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL4UI64NVPROC) (GLuint  index, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z, GLuint64EXT  w);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBL4UI64VNVPROC) (GLuint  index, const GLuint64EXT * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBLFORMATPROC) (GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBLFORMATNVPROC) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBLPOINTERPROC) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBLPOINTEREXTPROC) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBP1UIPROC) (GLuint  index, GLenum  type, GLboolean  normalized, GLuint  value);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBP1UIVPROC) (GLuint  index, GLenum  type, GLboolean  normalized, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBP2UIPROC) (GLuint  index, GLenum  type, GLboolean  normalized, GLuint  value);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBP2UIVPROC) (GLuint  index, GLenum  type, GLboolean  normalized, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBP3UIPROC) (GLuint  index, GLenum  type, GLboolean  normalized, GLuint  value);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBP3UIVPROC) (GLuint  index, GLenum  type, GLboolean  normalized, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBP4UIPROC) (GLuint  index, GLenum  type, GLboolean  normalized, GLuint  value);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBP4UIVPROC) (GLuint  index, GLenum  type, GLboolean  normalized, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBPARAMETERIAMDPROC) (GLuint  index, GLenum  pname, GLint  param);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBPOINTERPROC) (GLuint  index, GLint  size, GLenum  type, GLboolean  normalized, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBPOINTERARBPROC) (GLuint  index, GLint  size, GLenum  type, GLboolean  normalized, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBPOINTERNVPROC) (GLuint  index, GLint  fsize, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS1DVNVPROC) (GLuint  index, GLsizei  count, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS1FVNVPROC) (GLuint  index, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS1HVNVPROC) (GLuint  index, GLsizei  n, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS1SVNVPROC) (GLuint  index, GLsizei  count, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS2DVNVPROC) (GLuint  index, GLsizei  count, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS2FVNVPROC) (GLuint  index, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS2HVNVPROC) (GLuint  index, GLsizei  n, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS2SVNVPROC) (GLuint  index, GLsizei  count, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS3DVNVPROC) (GLuint  index, GLsizei  count, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS3FVNVPROC) (GLuint  index, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS3HVNVPROC) (GLuint  index, GLsizei  n, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS3SVNVPROC) (GLuint  index, GLsizei  count, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS4DVNVPROC) (GLuint  index, GLsizei  count, const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS4FVNVPROC) (GLuint  index, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS4HVNVPROC) (GLuint  index, GLsizei  n, const GLhalfNV * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS4SVNVPROC) (GLuint  index, GLsizei  count, const GLshort * v);
typedef void (GLAPIENTRY *PFNGLVERTEXATTRIBS4UBVNVPROC) (GLuint  index, GLsizei  count, const GLubyte * v);
typedef void (GLAPIENTRY *PFNGLVERTEXBINDINGDIVISORPROC) (GLuint  bindingindex, GLuint  divisor);
typedef void (GLAPIENTRY *PFNGLVERTEXBLENDARBPROC) (GLint  count);
typedef void (GLAPIENTRY *PFNGLVERTEXFORMATNVPROC) (GLint  size, GLenum  type, GLsizei  stride);
typedef void (GLAPIENTRY *PFNGLVERTEXP2UIPROC) (GLenum  type, GLuint  value);
typedef void (GLAPIENTRY *PFNGLVERTEXP2UIVPROC) (GLenum  type, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLVERTEXP3UIPROC) (GLenum  type, GLuint  value);
typedef void (GLAPIENTRY *PFNGLVERTEXP3UIVPROC) (GLenum  type, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLVERTEXP4UIPROC) (GLenum  type, GLuint  value);
typedef void (GLAPIENTRY *PFNGLVERTEXP4UIVPROC) (GLenum  type, const GLuint * value);
typedef void (GLAPIENTRY *PFNGLVERTEXPOINTERPROC) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXPOINTEREXTPROC) (GLint  size, GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXPOINTERVINTELPROC) (GLint  size, GLenum  type, const void ** pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXWEIGHTPOINTEREXTPROC) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLVERTEXWEIGHTFEXTPROC) (GLfloat  weight);
typedef void (GLAPIENTRY *PFNGLVERTEXWEIGHTFVEXTPROC) (const GLfloat * weight);
typedef void (GLAPIENTRY *PFNGLVERTEXWEIGHTHNVPROC) (GLhalfNV  weight);
typedef void (GLAPIENTRY *PFNGLVERTEXWEIGHTHVNVPROC) (const GLhalfNV * weight);
typedef GLenum (GLAPIENTRY *PFNGLVIDEOCAPTURENVPROC) (GLuint  video_capture_slot, GLuint * sequence_num, GLuint64EXT * capture_time);
typedef void (GLAPIENTRY *PFNGLVIDEOCAPTURESTREAMPARAMETERDVNVPROC) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, const GLdouble * params);
typedef void (GLAPIENTRY *PFNGLVIDEOCAPTURESTREAMPARAMETERFVNVPROC) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, const GLfloat * params);
typedef void (GLAPIENTRY *PFNGLVIDEOCAPTURESTREAMPARAMETERIVNVPROC) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, const GLint * params);
typedef void (GLAPIENTRY *PFNGLVIEWPORTPROC) (GLint  x, GLint  y, GLsizei  width, GLsizei  height);
typedef void (GLAPIENTRY *PFNGLVIEWPORTARRAYVPROC) (GLuint  first, GLsizei  count, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVIEWPORTINDEXEDFPROC) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  w, GLfloat  h);
typedef void (GLAPIENTRY *PFNGLVIEWPORTINDEXEDFVPROC) (GLuint  index, const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLVIEWPORTPOSITIONWSCALENVPROC) (GLuint  index, GLfloat  xcoeff, GLfloat  ycoeff);
typedef void (GLAPIENTRY *PFNGLVIEWPORTSWIZZLENVPROC) (GLuint  index, GLenum  swizzlex, GLenum  swizzley, GLenum  swizzlez, GLenum  swizzlew);
typedef void (GLAPIENTRY *PFNGLWAITSEMAPHOREEXTPROC) (GLuint  semaphore, GLuint  numBufferBarriers, const GLuint * buffers, GLuint  numTextureBarriers, const GLuint * textures, const GLenum * srcLayouts);
typedef void (GLAPIENTRY *PFNGLWAITSYNCPROC) (GLsync  sync, GLbitfield  flags, GLuint64  timeout);
typedef void (GLAPIENTRY *PFNGLWEIGHTPATHSNVPROC) (GLuint  resultPath, GLsizei  numPaths, const GLuint * paths, const GLfloat * weights);
typedef void (GLAPIENTRY *PFNGLWEIGHTPOINTERARBPROC) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
typedef void (GLAPIENTRY *PFNGLWEIGHTBVARBPROC) (GLint  size, const GLbyte * weights);
typedef void (GLAPIENTRY *PFNGLWEIGHTDVARBPROC) (GLint  size, const GLdouble * weights);
typedef void (GLAPIENTRY *PFNGLWEIGHTFVARBPROC) (GLint  size, const GLfloat * weights);
typedef void (GLAPIENTRY *PFNGLWEIGHTIVARBPROC) (GLint  size, const GLint * weights);
typedef void (GLAPIENTRY *PFNGLWEIGHTSVARBPROC) (GLint  size, const GLshort * weights);
typedef void (GLAPIENTRY *PFNGLWEIGHTUBVARBPROC) (GLint  size, const GLubyte * weights);
typedef void (GLAPIENTRY *PFNGLWEIGHTUIVARBPROC) (GLint  size, const GLuint * weights);
typedef void (GLAPIENTRY *PFNGLWEIGHTUSVARBPROC) (GLint  size, const GLushort * weights);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2DPROC) (GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2DARBPROC) (GLdouble  x, GLdouble  y);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2DVARBPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2FPROC) (GLfloat  x, GLfloat  y);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2FARBPROC) (GLfloat  x, GLfloat  y);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2FVARBPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2IPROC) (GLint  x, GLint  y);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2IARBPROC) (GLint  x, GLint  y);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2IVARBPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2SPROC) (GLshort  x, GLshort  y);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2SARBPROC) (GLshort  x, GLshort  y);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS2SVARBPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3DPROC) (GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3DARBPROC) (GLdouble  x, GLdouble  y, GLdouble  z);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3DVPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3DVARBPROC) (const GLdouble * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3FPROC) (GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3FARBPROC) (GLfloat  x, GLfloat  y, GLfloat  z);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3FVPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3FVARBPROC) (const GLfloat * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3IPROC) (GLint  x, GLint  y, GLint  z);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3IARBPROC) (GLint  x, GLint  y, GLint  z);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3IVPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3IVARBPROC) (const GLint * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3SPROC) (GLshort  x, GLshort  y, GLshort  z);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3SARBPROC) (GLshort  x, GLshort  y, GLshort  z);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3SVPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLWINDOWPOS3SVARBPROC) (const GLshort * v);
typedef void (GLAPIENTRY *PFNGLWINDOWRECTANGLESEXTPROC) (GLenum  mode, GLsizei  count, const GLint * box);
typedef void (GLAPIENTRY *PFNGLWRITEMASKEXTPROC) (GLuint  res, GLuint  in, GLenum  outX, GLenum  outY, GLenum  outZ, GLenum  outW);
typedef void (GLAPIENTRY *PFNGLDRAWVKIMAGENVPROC) (GLuint64  vkImage, GLuint  sampler, GLfloat  x0, GLfloat  y0, GLfloat  x1, GLfloat  y1, GLfloat  z, GLfloat  s0, GLfloat  t0, GLfloat  s1, GLfloat  t1);
typedef GLVULKANPROCNV (GLAPIENTRY *PFNGLGETVKPROCADDRNVPROC) (const GLchar * name);
typedef void (GLAPIENTRY *PFNGLWAITVKSEMAPHORENVPROC) (GLuint64  vkSemaphore);
typedef void (GLAPIENTRY *PFNGLSIGNALVKSEMAPHORENVPROC) (GLuint64  vkSemaphore);
typedef void (GLAPIENTRY *PFNGLSIGNALVKFENCENVPROC) (GLuint64  vkFence);


extern bool GLAPILOADER_GL_VERSION_1_0;
extern bool GLAPILOADER_GL_VERSION_1_1;
extern bool GLAPILOADER_GL_VERSION_1_2;
extern bool GLAPILOADER_GL_VERSION_1_3;
extern bool GLAPILOADER_GL_VERSION_1_4;
extern bool GLAPILOADER_GL_VERSION_1_5;
extern bool GLAPILOADER_GL_VERSION_2_0;
extern bool GLAPILOADER_GL_VERSION_2_1;
extern bool GLAPILOADER_GL_VERSION_3_0;
extern bool GLAPILOADER_GL_VERSION_3_1;
extern bool GLAPILOADER_GL_VERSION_3_2;
extern bool GLAPILOADER_GL_VERSION_3_3;
extern bool GLAPILOADER_GL_VERSION_4_0;
extern bool GLAPILOADER_GL_VERSION_4_1;
extern bool GLAPILOADER_GL_VERSION_4_2;
extern bool GLAPILOADER_GL_VERSION_4_3;
extern bool GLAPILOADER_GL_VERSION_4_4;
extern bool GLAPILOADER_GL_VERSION_4_5;
extern bool GLAPILOADER_GL_VERSION_4_6;


extern bool GLAPILOADER_GL_AMD_blend_minmax_factor;
extern bool GLAPILOADER_GL_AMD_conservative_depth;
extern bool GLAPILOADER_GL_AMD_debug_output;
extern bool GLAPILOADER_GL_AMD_depth_clamp_separate;
extern bool GLAPILOADER_GL_AMD_draw_buffers_blend;
extern bool GLAPILOADER_GL_AMD_framebuffer_multisample_advanced;
extern bool GLAPILOADER_GL_AMD_framebuffer_sample_positions;
extern bool GLAPILOADER_GL_AMD_gcn_shader;
extern bool GLAPILOADER_GL_AMD_gpu_shader_half_float;
extern bool GLAPILOADER_GL_AMD_gpu_shader_int16;
extern bool GLAPILOADER_GL_AMD_gpu_shader_int64;
extern bool GLAPILOADER_GL_AMD_interleaved_elements;
extern bool GLAPILOADER_GL_AMD_multi_draw_indirect;
extern bool GLAPILOADER_GL_AMD_name_gen_delete;
extern bool GLAPILOADER_GL_AMD_occlusion_query_event;
extern bool GLAPILOADER_GL_AMD_performance_monitor;
extern bool GLAPILOADER_GL_AMD_pinned_memory;
extern bool GLAPILOADER_GL_AMD_query_buffer_object;
extern bool GLAPILOADER_GL_AMD_sample_positions;
extern bool GLAPILOADER_GL_AMD_seamless_cubemap_per_texture;
extern bool GLAPILOADER_GL_AMD_shader_atomic_counter_ops;
extern bool GLAPILOADER_GL_AMD_shader_ballot;
extern bool GLAPILOADER_GL_AMD_shader_gpu_shader_half_float_fetch;
extern bool GLAPILOADER_GL_AMD_shader_image_load_store_lod;
extern bool GLAPILOADER_GL_AMD_shader_stencil_export;
extern bool GLAPILOADER_GL_AMD_shader_trinary_minmax;
extern bool GLAPILOADER_GL_AMD_shader_explicit_vertex_parameter;
extern bool GLAPILOADER_GL_AMD_sparse_texture;
extern bool GLAPILOADER_GL_AMD_stencil_operation_extended;
extern bool GLAPILOADER_GL_AMD_texture_gather_bias_lod;
extern bool GLAPILOADER_GL_AMD_texture_texture4;
extern bool GLAPILOADER_GL_AMD_transform_feedback3_lines_triangles;
extern bool GLAPILOADER_GL_AMD_transform_feedback4;
extern bool GLAPILOADER_GL_AMD_vertex_shader_layer;
extern bool GLAPILOADER_GL_AMD_vertex_shader_tessellator;
extern bool GLAPILOADER_GL_AMD_vertex_shader_viewport_index;
extern bool GLAPILOADER_GL_APPLE_aux_depth_stencil;
extern bool GLAPILOADER_GL_APPLE_client_storage;
extern bool GLAPILOADER_GL_APPLE_element_array;
extern bool GLAPILOADER_GL_APPLE_fence;
extern bool GLAPILOADER_GL_APPLE_float_pixels;
extern bool GLAPILOADER_GL_APPLE_flush_buffer_range;
extern bool GLAPILOADER_GL_APPLE_object_purgeable;
extern bool GLAPILOADER_GL_APPLE_rgb_422;
extern bool GLAPILOADER_GL_APPLE_row_bytes;
extern bool GLAPILOADER_GL_APPLE_specular_vector;
extern bool GLAPILOADER_GL_APPLE_texture_range;
extern bool GLAPILOADER_GL_APPLE_transform_hint;
extern bool GLAPILOADER_GL_APPLE_vertex_array_object;
extern bool GLAPILOADER_GL_APPLE_vertex_array_range;
extern bool GLAPILOADER_GL_APPLE_vertex_program_evaluators;
extern bool GLAPILOADER_GL_APPLE_ycbcr_422;
extern bool GLAPILOADER_GL_ARB_ES2_compatibility;
extern bool GLAPILOADER_GL_ARB_ES3_1_compatibility;
extern bool GLAPILOADER_GL_ARB_ES3_2_compatibility;
extern bool GLAPILOADER_GL_ARB_ES3_compatibility;
extern bool GLAPILOADER_GL_ARB_arrays_of_arrays;
extern bool GLAPILOADER_GL_ARB_base_instance;
extern bool GLAPILOADER_GL_ARB_bindless_texture;
extern bool GLAPILOADER_GL_ARB_blend_func_extended;
extern bool GLAPILOADER_GL_ARB_buffer_storage;
extern bool GLAPILOADER_GL_ARB_cl_event;
extern bool GLAPILOADER_GL_ARB_clear_buffer_object;
extern bool GLAPILOADER_GL_ARB_clear_texture;
extern bool GLAPILOADER_GL_ARB_clip_control;
extern bool GLAPILOADER_GL_ARB_color_buffer_float;
extern bool GLAPILOADER_GL_ARB_compatibility;
extern bool GLAPILOADER_GL_ARB_compressed_texture_pixel_storage;
extern bool GLAPILOADER_GL_ARB_compute_shader;
extern bool GLAPILOADER_GL_ARB_compute_variable_group_size;
extern bool GLAPILOADER_GL_ARB_conditional_render_inverted;
extern bool GLAPILOADER_GL_ARB_conservative_depth;
extern bool GLAPILOADER_GL_ARB_copy_buffer;
extern bool GLAPILOADER_GL_ARB_copy_image;
extern bool GLAPILOADER_GL_ARB_cull_distance;
extern bool GLAPILOADER_GL_ARB_debug_output;
extern bool GLAPILOADER_GL_ARB_depth_buffer_float;
extern bool GLAPILOADER_GL_ARB_depth_clamp;
extern bool GLAPILOADER_GL_ARB_depth_texture;
extern bool GLAPILOADER_GL_ARB_derivative_control;
extern bool GLAPILOADER_GL_ARB_direct_state_access;
extern bool GLAPILOADER_GL_ARB_draw_buffers;
extern bool GLAPILOADER_GL_ARB_draw_buffers_blend;
extern bool GLAPILOADER_GL_ARB_draw_elements_base_vertex;
extern bool GLAPILOADER_GL_ARB_draw_indirect;
extern bool GLAPILOADER_GL_ARB_draw_instanced;
extern bool GLAPILOADER_GL_ARB_enhanced_layouts;
extern bool GLAPILOADER_GL_ARB_explicit_attrib_location;
extern bool GLAPILOADER_GL_ARB_explicit_uniform_location;
extern bool GLAPILOADER_GL_ARB_fragment_coord_conventions;
extern bool GLAPILOADER_GL_ARB_fragment_layer_viewport;
extern bool GLAPILOADER_GL_ARB_fragment_program;
extern bool GLAPILOADER_GL_ARB_fragment_program_shadow;
extern bool GLAPILOADER_GL_ARB_fragment_shader;
extern bool GLAPILOADER_GL_ARB_fragment_shader_interlock;
extern bool GLAPILOADER_GL_ARB_framebuffer_no_attachments;
extern bool GLAPILOADER_GL_ARB_framebuffer_object;
extern bool GLAPILOADER_GL_ARB_framebuffer_sRGB;
extern bool GLAPILOADER_GL_ARB_geometry_shader4;
extern bool GLAPILOADER_GL_ARB_get_program_binary;
extern bool GLAPILOADER_GL_ARB_get_texture_sub_image;
extern bool GLAPILOADER_GL_ARB_gl_spirv;
extern bool GLAPILOADER_GL_ARB_gpu_shader5;
extern bool GLAPILOADER_GL_ARB_gpu_shader_fp64;
extern bool GLAPILOADER_GL_ARB_gpu_shader_int64;
extern bool GLAPILOADER_GL_ARB_half_float_pixel;
extern bool GLAPILOADER_GL_ARB_half_float_vertex;
extern bool GLAPILOADER_GL_ARB_imaging;
extern bool GLAPILOADER_GL_ARB_indirect_parameters;
extern bool GLAPILOADER_GL_ARB_instanced_arrays;
extern bool GLAPILOADER_GL_ARB_internalformat_query;
extern bool GLAPILOADER_GL_ARB_internalformat_query2;
extern bool GLAPILOADER_GL_ARB_invalidate_subdata;
extern bool GLAPILOADER_GL_ARB_map_buffer_alignment;
extern bool GLAPILOADER_GL_ARB_map_buffer_range;
extern bool GLAPILOADER_GL_ARB_matrix_palette;
extern bool GLAPILOADER_GL_ARB_multi_bind;
extern bool GLAPILOADER_GL_ARB_multi_draw_indirect;
extern bool GLAPILOADER_GL_ARB_multisample;
extern bool GLAPILOADER_GL_ARB_multitexture;
extern bool GLAPILOADER_GL_ARB_occlusion_query;
extern bool GLAPILOADER_GL_ARB_occlusion_query2;
extern bool GLAPILOADER_GL_ARB_parallel_shader_compile;
extern bool GLAPILOADER_GL_ARB_pipeline_statistics_query;
extern bool GLAPILOADER_GL_ARB_pixel_buffer_object;
extern bool GLAPILOADER_GL_ARB_point_parameters;
extern bool GLAPILOADER_GL_ARB_point_sprite;
extern bool GLAPILOADER_GL_ARB_polygon_offset_clamp;
extern bool GLAPILOADER_GL_ARB_post_depth_coverage;
extern bool GLAPILOADER_GL_ARB_program_interface_query;
extern bool GLAPILOADER_GL_ARB_provoking_vertex;
extern bool GLAPILOADER_GL_ARB_query_buffer_object;
extern bool GLAPILOADER_GL_ARB_robust_buffer_access_behavior;
extern bool GLAPILOADER_GL_ARB_robustness;
extern bool GLAPILOADER_GL_ARB_robustness_isolation;
extern bool GLAPILOADER_GL_ARB_sample_locations;
extern bool GLAPILOADER_GL_ARB_sample_shading;
extern bool GLAPILOADER_GL_ARB_sampler_objects;
extern bool GLAPILOADER_GL_ARB_seamless_cube_map;
extern bool GLAPILOADER_GL_ARB_seamless_cubemap_per_texture;
extern bool GLAPILOADER_GL_ARB_separate_shader_objects;
extern bool GLAPILOADER_GL_ARB_shader_atomic_counter_ops;
extern bool GLAPILOADER_GL_ARB_shader_atomic_counters;
extern bool GLAPILOADER_GL_ARB_shader_ballot;
extern bool GLAPILOADER_GL_ARB_shader_bit_encoding;
extern bool GLAPILOADER_GL_ARB_shader_clock;
extern bool GLAPILOADER_GL_ARB_shader_draw_parameters;
extern bool GLAPILOADER_GL_ARB_shader_group_vote;
extern bool GLAPILOADER_GL_ARB_shader_image_load_store;
extern bool GLAPILOADER_GL_ARB_shader_image_size;
extern bool GLAPILOADER_GL_ARB_shader_objects;
extern bool GLAPILOADER_GL_ARB_shader_precision;
extern bool GLAPILOADER_GL_ARB_shader_stencil_export;
extern bool GLAPILOADER_GL_ARB_shader_storage_buffer_object;
extern bool GLAPILOADER_GL_ARB_shader_subroutine;
extern bool GLAPILOADER_GL_ARB_shader_texture_image_samples;
extern bool GLAPILOADER_GL_ARB_shader_texture_lod;
extern bool GLAPILOADER_GL_ARB_shader_viewport_layer_array;
extern bool GLAPILOADER_GL_ARB_shading_language_100;
extern bool GLAPILOADER_GL_ARB_shading_language_420pack;
extern bool GLAPILOADER_GL_ARB_shading_language_include;
extern bool GLAPILOADER_GL_ARB_shading_language_packing;
extern bool GLAPILOADER_GL_ARB_shadow;
extern bool GLAPILOADER_GL_ARB_shadow_ambient;
extern bool GLAPILOADER_GL_ARB_sparse_buffer;
extern bool GLAPILOADER_GL_ARB_sparse_texture;
extern bool GLAPILOADER_GL_ARB_sparse_texture2;
extern bool GLAPILOADER_GL_ARB_sparse_texture_clamp;
extern bool GLAPILOADER_GL_ARB_spirv_extensions;
extern bool GLAPILOADER_GL_ARB_stencil_texturing;
extern bool GLAPILOADER_GL_ARB_sync;
extern bool GLAPILOADER_GL_ARB_tessellation_shader;
extern bool GLAPILOADER_GL_ARB_texture_barrier;
extern bool GLAPILOADER_GL_ARB_texture_border_clamp;
extern bool GLAPILOADER_GL_ARB_texture_buffer_object;
extern bool GLAPILOADER_GL_ARB_texture_buffer_object_rgb32;
extern bool GLAPILOADER_GL_ARB_texture_buffer_range;
extern bool GLAPILOADER_GL_ARB_texture_compression;
extern bool GLAPILOADER_GL_ARB_texture_compression_bptc;
extern bool GLAPILOADER_GL_ARB_texture_compression_rgtc;
extern bool GLAPILOADER_GL_ARB_texture_cube_map;
extern bool GLAPILOADER_GL_ARB_texture_cube_map_array;
extern bool GLAPILOADER_GL_ARB_texture_env_add;
extern bool GLAPILOADER_GL_ARB_texture_env_combine;
extern bool GLAPILOADER_GL_ARB_texture_env_crossbar;
extern bool GLAPILOADER_GL_ARB_texture_env_dot3;
extern bool GLAPILOADER_GL_ARB_texture_filter_anisotropic;
extern bool GLAPILOADER_GL_ARB_texture_filter_minmax;
extern bool GLAPILOADER_GL_ARB_texture_float;
extern bool GLAPILOADER_GL_ARB_texture_gather;
extern bool GLAPILOADER_GL_ARB_texture_mirror_clamp_to_edge;
extern bool GLAPILOADER_GL_ARB_texture_mirrored_repeat;
extern bool GLAPILOADER_GL_ARB_texture_multisample;
extern bool GLAPILOADER_GL_ARB_texture_non_power_of_two;
extern bool GLAPILOADER_GL_ARB_texture_query_levels;
extern bool GLAPILOADER_GL_ARB_texture_query_lod;
extern bool GLAPILOADER_GL_ARB_texture_rectangle;
extern bool GLAPILOADER_GL_ARB_texture_rg;
extern bool GLAPILOADER_GL_ARB_texture_rgb10_a2ui;
extern bool GLAPILOADER_GL_ARB_texture_stencil8;
extern bool GLAPILOADER_GL_ARB_texture_storage;
extern bool GLAPILOADER_GL_ARB_texture_storage_multisample;
extern bool GLAPILOADER_GL_ARB_texture_swizzle;
extern bool GLAPILOADER_GL_ARB_texture_view;
extern bool GLAPILOADER_GL_ARB_timer_query;
extern bool GLAPILOADER_GL_ARB_transform_feedback2;
extern bool GLAPILOADER_GL_ARB_transform_feedback3;
extern bool GLAPILOADER_GL_ARB_transform_feedback_instanced;
extern bool GLAPILOADER_GL_ARB_transform_feedback_overflow_query;
extern bool GLAPILOADER_GL_ARB_transpose_matrix;
extern bool GLAPILOADER_GL_ARB_uniform_buffer_object;
extern bool GLAPILOADER_GL_ARB_vertex_array_bgra;
extern bool GLAPILOADER_GL_ARB_vertex_array_object;
extern bool GLAPILOADER_GL_ARB_vertex_attrib_64bit;
extern bool GLAPILOADER_GL_ARB_vertex_attrib_binding;
extern bool GLAPILOADER_GL_ARB_vertex_blend;
extern bool GLAPILOADER_GL_ARB_vertex_buffer_object;
extern bool GLAPILOADER_GL_ARB_vertex_program;
extern bool GLAPILOADER_GL_ARB_vertex_shader;
extern bool GLAPILOADER_GL_ARB_vertex_type_10f_11f_11f_rev;
extern bool GLAPILOADER_GL_ARB_vertex_type_2_10_10_10_rev;
extern bool GLAPILOADER_GL_ARB_viewport_array;
extern bool GLAPILOADER_GL_ARB_window_pos;
extern bool GLAPILOADER_GL_EXT_422_pixels;
extern bool GLAPILOADER_GL_EXT_EGL_image_storage;
extern bool GLAPILOADER_GL_EXT_EGL_sync;
extern bool GLAPILOADER_GL_EXT_abgr;
extern bool GLAPILOADER_GL_EXT_bgra;
extern bool GLAPILOADER_GL_EXT_bindable_uniform;
extern bool GLAPILOADER_GL_EXT_blend_color;
extern bool GLAPILOADER_GL_EXT_blend_equation_separate;
extern bool GLAPILOADER_GL_EXT_blend_func_separate;
extern bool GLAPILOADER_GL_EXT_blend_logic_op;
extern bool GLAPILOADER_GL_EXT_blend_minmax;
extern bool GLAPILOADER_GL_EXT_blend_subtract;
extern bool GLAPILOADER_GL_EXT_clip_volume_hint;
extern bool GLAPILOADER_GL_EXT_cmyka;
extern bool GLAPILOADER_GL_EXT_color_subtable;
extern bool GLAPILOADER_GL_EXT_compiled_vertex_array;
extern bool GLAPILOADER_GL_EXT_convolution;
extern bool GLAPILOADER_GL_EXT_coordinate_frame;
extern bool GLAPILOADER_GL_EXT_copy_texture;
extern bool GLAPILOADER_GL_EXT_cull_vertex;
extern bool GLAPILOADER_GL_EXT_debug_label;
extern bool GLAPILOADER_GL_EXT_debug_marker;
extern bool GLAPILOADER_GL_EXT_depth_bounds_test;
extern bool GLAPILOADER_GL_EXT_direct_state_access;
extern bool GLAPILOADER_GL_EXT_draw_buffers2;
extern bool GLAPILOADER_GL_EXT_draw_instanced;
extern bool GLAPILOADER_GL_EXT_draw_range_elements;
extern bool GLAPILOADER_GL_EXT_external_buffer;
extern bool GLAPILOADER_GL_EXT_fog_coord;
extern bool GLAPILOADER_GL_EXT_framebuffer_blit;
extern bool GLAPILOADER_GL_EXT_framebuffer_multisample;
extern bool GLAPILOADER_GL_EXT_framebuffer_multisample_blit_scaled;
extern bool GLAPILOADER_GL_EXT_framebuffer_object;
extern bool GLAPILOADER_GL_EXT_framebuffer_sRGB;
extern bool GLAPILOADER_GL_EXT_geometry_shader4;
extern bool GLAPILOADER_GL_EXT_gpu_program_parameters;
extern bool GLAPILOADER_GL_EXT_gpu_shader4;
extern bool GLAPILOADER_GL_EXT_histogram;
extern bool GLAPILOADER_GL_EXT_index_array_formats;
extern bool GLAPILOADER_GL_EXT_index_func;
extern bool GLAPILOADER_GL_EXT_index_material;
extern bool GLAPILOADER_GL_EXT_index_texture;
extern bool GLAPILOADER_GL_EXT_light_texture;
extern bool GLAPILOADER_GL_EXT_memory_object;
extern bool GLAPILOADER_GL_EXT_memory_object_fd;
extern bool GLAPILOADER_GL_EXT_memory_object_win32;
extern bool GLAPILOADER_GL_EXT_misc_attribute;
extern bool GLAPILOADER_GL_EXT_multi_draw_arrays;
extern bool GLAPILOADER_GL_EXT_multisample;
extern bool GLAPILOADER_GL_EXT_multiview_tessellation_geometry_shader;
extern bool GLAPILOADER_GL_EXT_multiview_texture_multisample;
extern bool GLAPILOADER_GL_EXT_multiview_timer_query;
extern bool GLAPILOADER_GL_EXT_packed_depth_stencil;
extern bool GLAPILOADER_GL_EXT_packed_float;
extern bool GLAPILOADER_GL_EXT_packed_pixels;
extern bool GLAPILOADER_GL_EXT_paletted_texture;
extern bool GLAPILOADER_GL_EXT_pixel_buffer_object;
extern bool GLAPILOADER_GL_EXT_pixel_transform;
extern bool GLAPILOADER_GL_EXT_pixel_transform_color_table;
extern bool GLAPILOADER_GL_EXT_point_parameters;
extern bool GLAPILOADER_GL_EXT_polygon_offset;
extern bool GLAPILOADER_GL_EXT_polygon_offset_clamp;
extern bool GLAPILOADER_GL_EXT_post_depth_coverage;
extern bool GLAPILOADER_GL_EXT_provoking_vertex;
extern bool GLAPILOADER_GL_EXT_raster_multisample;
extern bool GLAPILOADER_GL_EXT_rescale_normal;
extern bool GLAPILOADER_GL_EXT_semaphore;
extern bool GLAPILOADER_GL_EXT_semaphore_fd;
extern bool GLAPILOADER_GL_EXT_semaphore_win32;
extern bool GLAPILOADER_GL_EXT_secondary_color;
extern bool GLAPILOADER_GL_EXT_separate_shader_objects;
extern bool GLAPILOADER_GL_EXT_separate_specular_color;
extern bool GLAPILOADER_GL_EXT_shader_framebuffer_fetch;
extern bool GLAPILOADER_GL_EXT_shader_framebuffer_fetch_non_coherent;
extern bool GLAPILOADER_GL_EXT_shader_image_load_formatted;
extern bool GLAPILOADER_GL_EXT_shader_image_load_store;
extern bool GLAPILOADER_GL_EXT_shader_integer_mix;
extern bool GLAPILOADER_GL_EXT_shadow_funcs;
extern bool GLAPILOADER_GL_EXT_shared_texture_palette;
extern bool GLAPILOADER_GL_EXT_sparse_texture2;
extern bool GLAPILOADER_GL_EXT_stencil_clear_tag;
extern bool GLAPILOADER_GL_EXT_stencil_two_side;
extern bool GLAPILOADER_GL_EXT_stencil_wrap;
extern bool GLAPILOADER_GL_EXT_subtexture;
extern bool GLAPILOADER_GL_EXT_texture;
extern bool GLAPILOADER_GL_EXT_texture3D;
extern bool GLAPILOADER_GL_EXT_texture_array;
extern bool GLAPILOADER_GL_EXT_texture_buffer_object;
extern bool GLAPILOADER_GL_EXT_texture_compression_latc;
extern bool GLAPILOADER_GL_EXT_texture_compression_rgtc;
extern bool GLAPILOADER_GL_EXT_texture_compression_s3tc;
extern bool GLAPILOADER_GL_EXT_texture_cube_map;
extern bool GLAPILOADER_GL_EXT_texture_env_add;
extern bool GLAPILOADER_GL_EXT_texture_env_combine;
extern bool GLAPILOADER_GL_EXT_texture_env_dot3;
extern bool GLAPILOADER_GL_EXT_texture_filter_anisotropic;
extern bool GLAPILOADER_GL_EXT_texture_filter_minmax;
extern bool GLAPILOADER_GL_EXT_texture_integer;
extern bool GLAPILOADER_GL_EXT_texture_lod_bias;
extern bool GLAPILOADER_GL_EXT_texture_mirror_clamp;
extern bool GLAPILOADER_GL_EXT_texture_object;
extern bool GLAPILOADER_GL_EXT_texture_perturb_normal;
extern bool GLAPILOADER_GL_EXT_texture_sRGB;
extern bool GLAPILOADER_GL_EXT_texture_sRGB_R8;
extern bool GLAPILOADER_GL_EXT_texture_sRGB_decode;
extern bool GLAPILOADER_GL_EXT_texture_shared_exponent;
extern bool GLAPILOADER_GL_EXT_texture_snorm;
extern bool GLAPILOADER_GL_EXT_texture_swizzle;
extern bool GLAPILOADER_GL_EXT_timer_query;
extern bool GLAPILOADER_GL_EXT_transform_feedback;
extern bool GLAPILOADER_GL_EXT_vertex_array;
extern bool GLAPILOADER_GL_EXT_vertex_array_bgra;
extern bool GLAPILOADER_GL_EXT_vertex_attrib_64bit;
extern bool GLAPILOADER_GL_EXT_vertex_shader;
extern bool GLAPILOADER_GL_EXT_vertex_weighting;
extern bool GLAPILOADER_GL_EXT_win32_keyed_mutex;
extern bool GLAPILOADER_GL_EXT_window_rectangles;
extern bool GLAPILOADER_GL_EXT_x11_sync_object;
extern bool GLAPILOADER_GL_INTEL_conservative_rasterization;
extern bool GLAPILOADER_GL_INTEL_fragment_shader_ordering;
extern bool GLAPILOADER_GL_INTEL_framebuffer_CMAA;
extern bool GLAPILOADER_GL_INTEL_map_texture;
extern bool GLAPILOADER_GL_INTEL_blackhole_render;
extern bool GLAPILOADER_GL_INTEL_parallel_arrays;
extern bool GLAPILOADER_GL_INTEL_performance_query;
extern bool GLAPILOADER_GL_KHR_blend_equation_advanced;
extern bool GLAPILOADER_GL_KHR_blend_equation_advanced_coherent;
extern bool GLAPILOADER_GL_KHR_context_flush_control;
extern bool GLAPILOADER_GL_KHR_debug;
extern bool GLAPILOADER_GL_KHR_no_error;
extern bool GLAPILOADER_GL_KHR_robust_buffer_access_behavior;
extern bool GLAPILOADER_GL_KHR_robustness;
extern bool GLAPILOADER_GL_KHR_shader_subgroup;
extern bool GLAPILOADER_GL_KHR_texture_compression_astc_hdr;
extern bool GLAPILOADER_GL_KHR_texture_compression_astc_ldr;
extern bool GLAPILOADER_GL_KHR_texture_compression_astc_sliced_3d;
extern bool GLAPILOADER_GL_KHR_parallel_shader_compile;
extern bool GLAPILOADER_GL_NV_alpha_to_coverage_dither_control;
extern bool GLAPILOADER_GL_NV_bindless_multi_draw_indirect;
extern bool GLAPILOADER_GL_NV_bindless_multi_draw_indirect_count;
extern bool GLAPILOADER_GL_NV_bindless_texture;
extern bool GLAPILOADER_GL_NV_blend_equation_advanced;
extern bool GLAPILOADER_GL_NV_blend_equation_advanced_coherent;
extern bool GLAPILOADER_GL_NV_blend_minmax_factor;
extern bool GLAPILOADER_GL_NV_blend_square;
extern bool GLAPILOADER_GL_NV_clip_space_w_scaling;
extern bool GLAPILOADER_GL_NV_command_list;
extern bool GLAPILOADER_GL_NV_compute_program5;
extern bool GLAPILOADER_GL_NV_compute_shader_derivatives;
extern bool GLAPILOADER_GL_NV_conditional_render;
extern bool GLAPILOADER_GL_NV_conservative_raster;
extern bool GLAPILOADER_GL_NV_conservative_raster_dilate;
extern bool GLAPILOADER_GL_NV_conservative_raster_pre_snap;
extern bool GLAPILOADER_GL_NV_conservative_raster_pre_snap_triangles;
extern bool GLAPILOADER_GL_NV_conservative_raster_underestimation;
extern bool GLAPILOADER_GL_NV_copy_depth_to_color;
extern bool GLAPILOADER_GL_NV_copy_image;
extern bool GLAPILOADER_GL_NV_deep_texture3D;
extern bool GLAPILOADER_GL_NV_depth_buffer_float;
extern bool GLAPILOADER_GL_NV_depth_clamp;
extern bool GLAPILOADER_GL_NV_draw_texture;
extern bool GLAPILOADER_GL_NV_draw_vulkan_image;
extern bool GLAPILOADER_GL_NV_evaluators;
extern bool GLAPILOADER_GL_NV_explicit_multisample;
extern bool GLAPILOADER_GL_NV_fence;
extern bool GLAPILOADER_GL_NV_fill_rectangle;
extern bool GLAPILOADER_GL_NV_float_buffer;
extern bool GLAPILOADER_GL_NV_fog_distance;
extern bool GLAPILOADER_GL_NV_fragment_coverage_to_color;
extern bool GLAPILOADER_GL_NV_fragment_program;
extern bool GLAPILOADER_GL_NV_fragment_program2;
extern bool GLAPILOADER_GL_NV_fragment_program4;
extern bool GLAPILOADER_GL_NV_fragment_program_option;
extern bool GLAPILOADER_GL_NV_fragment_shader_barycentric;
extern bool GLAPILOADER_GL_NV_fragment_shader_interlock;
extern bool GLAPILOADER_GL_NV_framebuffer_mixed_samples;
extern bool GLAPILOADER_GL_NV_framebuffer_multisample_coverage;
extern bool GLAPILOADER_GL_NV_geometry_program4;
extern bool GLAPILOADER_GL_NV_geometry_shader4;
extern bool GLAPILOADER_GL_NV_geometry_shader_passthrough;
extern bool GLAPILOADER_GL_NV_gpu_program4;
extern bool GLAPILOADER_GL_NV_gpu_program5;
extern bool GLAPILOADER_GL_NV_gpu_program5_mem_extended;
extern bool GLAPILOADER_GL_NV_gpu_shader5;
extern bool GLAPILOADER_GL_NV_half_float;
extern bool GLAPILOADER_GL_NV_internalformat_sample_query;
extern bool GLAPILOADER_GL_NV_light_max_exponent;
extern bool GLAPILOADER_GL_NV_gpu_multicast;
extern bool GLAPILOADER_GL_NV_memory_attachment;
extern bool GLAPILOADER_GL_NV_mesh_shader;
extern bool GLAPILOADER_GL_NV_multisample_coverage;
extern bool GLAPILOADER_GL_NV_multisample_filter_hint;
extern bool GLAPILOADER_GL_NV_occlusion_query;
extern bool GLAPILOADER_GL_NV_packed_depth_stencil;
extern bool GLAPILOADER_GL_NV_parameter_buffer_object;
extern bool GLAPILOADER_GL_NV_parameter_buffer_object2;
extern bool GLAPILOADER_GL_NV_path_rendering;
extern bool GLAPILOADER_GL_NV_path_rendering_shared_edge;
extern bool GLAPILOADER_GL_NV_pixel_data_range;
extern bool GLAPILOADER_GL_NV_point_sprite;
extern bool GLAPILOADER_GL_NV_present_video;
extern bool GLAPILOADER_GL_NV_primitive_restart;
extern bool GLAPILOADER_GL_NV_query_resource;
extern bool GLAPILOADER_GL_NV_query_resource_tag;
extern bool GLAPILOADER_GL_NV_register_combiners;
extern bool GLAPILOADER_GL_NV_register_combiners2;
extern bool GLAPILOADER_GL_NV_representative_fragment_test;
extern bool GLAPILOADER_GL_NV_robustness_video_memory_purge;
extern bool GLAPILOADER_GL_NV_sample_locations;
extern bool GLAPILOADER_GL_NV_sample_mask_override_coverage;
extern bool GLAPILOADER_GL_NV_scissor_exclusive;
extern bool GLAPILOADER_GL_NV_shader_atomic_counters;
extern bool GLAPILOADER_GL_NV_shader_atomic_float;
extern bool GLAPILOADER_GL_NV_shader_atomic_float64;
extern bool GLAPILOADER_GL_NV_shader_atomic_fp16_vector;
extern bool GLAPILOADER_GL_NV_shader_atomic_int64;
extern bool GLAPILOADER_GL_NV_shader_buffer_load;
extern bool GLAPILOADER_GL_NV_shader_buffer_store;
extern bool GLAPILOADER_GL_NV_shader_storage_buffer_object;
extern bool GLAPILOADER_GL_NV_shader_subgroup_partitioned;
extern bool GLAPILOADER_GL_NV_shader_texture_footprint;
extern bool GLAPILOADER_GL_NV_shader_thread_group;
extern bool GLAPILOADER_GL_NV_shader_thread_shuffle;
extern bool GLAPILOADER_GL_NV_shading_rate_image;
extern bool GLAPILOADER_GL_NV_stereo_view_rendering;
extern bool GLAPILOADER_GL_NV_tessellation_program5;
extern bool GLAPILOADER_GL_NV_texgen_emboss;
extern bool GLAPILOADER_GL_NV_texgen_reflection;
extern bool GLAPILOADER_GL_NV_texture_barrier;
extern bool GLAPILOADER_GL_NV_texture_compression_vtc;
extern bool GLAPILOADER_GL_NV_texture_env_combine4;
extern bool GLAPILOADER_GL_NV_texture_expand_normal;
extern bool GLAPILOADER_GL_NV_texture_multisample;
extern bool GLAPILOADER_GL_NV_texture_rectangle;
extern bool GLAPILOADER_GL_NV_texture_rectangle_compressed;
extern bool GLAPILOADER_GL_NV_texture_shader;
extern bool GLAPILOADER_GL_NV_texture_shader2;
extern bool GLAPILOADER_GL_NV_texture_shader3;
extern bool GLAPILOADER_GL_NV_transform_feedback;
extern bool GLAPILOADER_GL_NV_transform_feedback2;
extern bool GLAPILOADER_GL_NV_uniform_buffer_unified_memory;
extern bool GLAPILOADER_GL_NV_vdpau_interop;
extern bool GLAPILOADER_GL_NV_vdpau_interop2;
extern bool GLAPILOADER_GL_NV_vertex_array_range;
extern bool GLAPILOADER_GL_NV_vertex_array_range2;
extern bool GLAPILOADER_GL_NV_vertex_attrib_integer_64bit;
extern bool GLAPILOADER_GL_NV_vertex_buffer_unified_memory;
extern bool GLAPILOADER_GL_NV_vertex_program;
extern bool GLAPILOADER_GL_NV_vertex_program1_1;
extern bool GLAPILOADER_GL_NV_vertex_program2;
extern bool GLAPILOADER_GL_NV_vertex_program2_option;
extern bool GLAPILOADER_GL_NV_vertex_program3;
extern bool GLAPILOADER_GL_NV_vertex_program4;
extern bool GLAPILOADER_GL_NV_video_capture;
extern bool GLAPILOADER_GL_NV_viewport_array2;
extern bool GLAPILOADER_GL_NV_viewport_swizzle;
extern bool GLAPILOADER_GL_EXT_texture_shadow_lod;


extern void (GLAPIENTRY *glAccum) (GLenum  op, GLfloat  value);
extern void (GLAPIENTRY *glActiveProgramEXT) (GLuint  program);
extern void (GLAPIENTRY *glActiveShaderProgram) (GLuint  pipeline, GLuint  program);
extern void (GLAPIENTRY *glActiveShaderProgramEXT) (GLuint  pipeline, GLuint  program);
extern void (GLAPIENTRY *glActiveStencilFaceEXT) (GLenum  face);
extern void (GLAPIENTRY *glActiveTexture) (GLenum  texture);
extern void (GLAPIENTRY *glActiveTextureARB) (GLenum  texture);
extern void (GLAPIENTRY *glActiveVaryingNV) (GLuint  program, const GLchar * name);
extern void (GLAPIENTRY *glAlphaFunc) (GLenum  func, GLfloat  ref);
extern void (GLAPIENTRY *glAlphaToCoverageDitherControlNV) (GLenum  mode);
extern void (GLAPIENTRY *glApplyFramebufferAttachmentCMAAINTEL) ();
extern void (GLAPIENTRY *glApplyTextureEXT) (GLenum  mode);
extern GLboolean (GLAPIENTRY *glAcquireKeyedMutexWin32EXT) (GLuint  memory, GLuint64  key, GLuint  timeout);
extern GLboolean (GLAPIENTRY *glAreProgramsResidentNV) (GLsizei  n, const GLuint * programs, GLboolean * residences);
extern GLboolean (GLAPIENTRY *glAreTexturesResident) (GLsizei  n, const GLuint * textures, GLboolean * residences);
extern GLboolean (GLAPIENTRY *glAreTexturesResidentEXT) (GLsizei  n, const GLuint * textures, GLboolean * residences);
extern void (GLAPIENTRY *glArrayElement) (GLint  i);
extern void (GLAPIENTRY *glArrayElementEXT) (GLint  i);
extern void (GLAPIENTRY *glAttachObjectARB) (GLhandleARB  containerObj, GLhandleARB  obj);
extern void (GLAPIENTRY *glAttachShader) (GLuint  program, GLuint  shader);
extern void (GLAPIENTRY *glBegin) (GLenum  mode);
extern void (GLAPIENTRY *glBeginConditionalRender) (GLuint  id, GLenum  mode);
extern void (GLAPIENTRY *glBeginConditionalRenderNV) (GLuint  id, GLenum  mode);
extern void (GLAPIENTRY *glBeginOcclusionQueryNV) (GLuint  id);
extern void (GLAPIENTRY *glBeginPerfMonitorAMD) (GLuint  monitor);
extern void (GLAPIENTRY *glBeginPerfQueryINTEL) (GLuint  queryHandle);
extern void (GLAPIENTRY *glBeginQuery) (GLenum  target, GLuint  id);
extern void (GLAPIENTRY *glBeginQueryARB) (GLenum  target, GLuint  id);
extern void (GLAPIENTRY *glBeginQueryIndexed) (GLenum  target, GLuint  index, GLuint  id);
extern void (GLAPIENTRY *glBeginTransformFeedback) (GLenum  primitiveMode);
extern void (GLAPIENTRY *glBeginTransformFeedbackEXT) (GLenum  primitiveMode);
extern void (GLAPIENTRY *glBeginTransformFeedbackNV) (GLenum  primitiveMode);
extern void (GLAPIENTRY *glBeginVertexShaderEXT) ();
extern void (GLAPIENTRY *glBeginVideoCaptureNV) (GLuint  video_capture_slot);
extern void (GLAPIENTRY *glBindAttribLocation) (GLuint  program, GLuint  index, const GLchar * name);
extern void (GLAPIENTRY *glBindAttribLocationARB) (GLhandleARB  programObj, GLuint  index, const GLcharARB * name);
extern void (GLAPIENTRY *glBindBuffer) (GLenum  target, GLuint  buffer);
extern void (GLAPIENTRY *glBindBufferARB) (GLenum  target, GLuint  buffer);
extern void (GLAPIENTRY *glBindBufferBase) (GLenum  target, GLuint  index, GLuint  buffer);
extern void (GLAPIENTRY *glBindBufferBaseEXT) (GLenum  target, GLuint  index, GLuint  buffer);
extern void (GLAPIENTRY *glBindBufferBaseNV) (GLenum  target, GLuint  index, GLuint  buffer);
extern void (GLAPIENTRY *glBindBufferOffsetEXT) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset);
extern void (GLAPIENTRY *glBindBufferOffsetNV) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset);
extern void (GLAPIENTRY *glBindBufferRange) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
extern void (GLAPIENTRY *glBindBufferRangeEXT) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
extern void (GLAPIENTRY *glBindBufferRangeNV) (GLenum  target, GLuint  index, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
extern void (GLAPIENTRY *glBindBuffersBase) (GLenum  target, GLuint  first, GLsizei  count, const GLuint * buffers);
extern void (GLAPIENTRY *glBindBuffersRange) (GLenum  target, GLuint  first, GLsizei  count, const GLuint * buffers, const GLintptr * offsets, const GLsizeiptr * sizes);
extern void (GLAPIENTRY *glBindFragDataLocation) (GLuint  program, GLuint  color, const GLchar * name);
extern void (GLAPIENTRY *glBindFragDataLocationEXT) (GLuint  program, GLuint  color, const GLchar * name);
extern void (GLAPIENTRY *glBindFragDataLocationIndexed) (GLuint  program, GLuint  colorNumber, GLuint  index, const GLchar * name);
extern void (GLAPIENTRY *glBindFramebuffer) (GLenum  target, GLuint  framebuffer);
extern void (GLAPIENTRY *glBindFramebufferEXT) (GLenum  target, GLuint  framebuffer);
extern void (GLAPIENTRY *glBindImageTexture) (GLuint  unit, GLuint  texture, GLint  level, GLboolean  layered, GLint  layer, GLenum  access, GLenum  format);
extern void (GLAPIENTRY *glBindImageTextureEXT) (GLuint  index, GLuint  texture, GLint  level, GLboolean  layered, GLint  layer, GLenum  access, GLint  format);
extern void (GLAPIENTRY *glBindImageTextures) (GLuint  first, GLsizei  count, const GLuint * textures);
extern GLuint (GLAPIENTRY *glBindLightParameterEXT) (GLenum  light, GLenum  value);
extern GLuint (GLAPIENTRY *glBindMaterialParameterEXT) (GLenum  face, GLenum  value);
extern void (GLAPIENTRY *glBindMultiTextureEXT) (GLenum  texunit, GLenum  target, GLuint  texture);
extern GLuint (GLAPIENTRY *glBindParameterEXT) (GLenum  value);
extern void (GLAPIENTRY *glBindProgramARB) (GLenum  target, GLuint  program);
extern void (GLAPIENTRY *glBindProgramNV) (GLenum  target, GLuint  id);
extern void (GLAPIENTRY *glBindProgramPipeline) (GLuint  pipeline);
extern void (GLAPIENTRY *glBindProgramPipelineEXT) (GLuint  pipeline);
extern void (GLAPIENTRY *glBindRenderbuffer) (GLenum  target, GLuint  renderbuffer);
extern void (GLAPIENTRY *glBindRenderbufferEXT) (GLenum  target, GLuint  renderbuffer);
extern void (GLAPIENTRY *glBindSampler) (GLuint  unit, GLuint  sampler);
extern void (GLAPIENTRY *glBindSamplers) (GLuint  first, GLsizei  count, const GLuint * samplers);
extern void (GLAPIENTRY *glBindShadingRateImageNV) (GLuint  texture);
extern GLuint (GLAPIENTRY *glBindTexGenParameterEXT) (GLenum  unit, GLenum  coord, GLenum  value);
extern void (GLAPIENTRY *glBindTexture) (GLenum  target, GLuint  texture);
extern void (GLAPIENTRY *glBindTextureEXT) (GLenum  target, GLuint  texture);
extern void (GLAPIENTRY *glBindTextureUnit) (GLuint  unit, GLuint  texture);
extern GLuint (GLAPIENTRY *glBindTextureUnitParameterEXT) (GLenum  unit, GLenum  value);
extern void (GLAPIENTRY *glBindTextures) (GLuint  first, GLsizei  count, const GLuint * textures);
extern void (GLAPIENTRY *glBindTransformFeedback) (GLenum  target, GLuint  id);
extern void (GLAPIENTRY *glBindTransformFeedbackNV) (GLenum  target, GLuint  id);
extern void (GLAPIENTRY *glBindVertexArray) (GLuint  array);
extern void (GLAPIENTRY *glBindVertexArrayAPPLE) (GLuint  array);
extern void (GLAPIENTRY *glBindVertexBuffer) (GLuint  bindingindex, GLuint  buffer, GLintptr  offset, GLsizei  stride);
extern void (GLAPIENTRY *glBindVertexBuffers) (GLuint  first, GLsizei  count, const GLuint * buffers, const GLintptr * offsets, const GLsizei * strides);
extern void (GLAPIENTRY *glBindVertexShaderEXT) (GLuint  id);
extern void (GLAPIENTRY *glBindVideoCaptureStreamBufferNV) (GLuint  video_capture_slot, GLuint  stream, GLenum  frame_region, GLintptrARB  offset);
extern void (GLAPIENTRY *glBindVideoCaptureStreamTextureNV) (GLuint  video_capture_slot, GLuint  stream, GLenum  frame_region, GLenum  target, GLuint  texture);
extern void (GLAPIENTRY *glBinormal3bEXT) (GLbyte  bx, GLbyte  by, GLbyte  bz);
extern void (GLAPIENTRY *glBinormal3bvEXT) (const GLbyte * v);
extern void (GLAPIENTRY *glBinormal3dEXT) (GLdouble  bx, GLdouble  by, GLdouble  bz);
extern void (GLAPIENTRY *glBinormal3dvEXT) (const GLdouble * v);
extern void (GLAPIENTRY *glBinormal3fEXT) (GLfloat  bx, GLfloat  by, GLfloat  bz);
extern void (GLAPIENTRY *glBinormal3fvEXT) (const GLfloat * v);
extern void (GLAPIENTRY *glBinormal3iEXT) (GLint  bx, GLint  by, GLint  bz);
extern void (GLAPIENTRY *glBinormal3ivEXT) (const GLint * v);
extern void (GLAPIENTRY *glBinormal3sEXT) (GLshort  bx, GLshort  by, GLshort  bz);
extern void (GLAPIENTRY *glBinormal3svEXT) (const GLshort * v);
extern void (GLAPIENTRY *glBinormalPointerEXT) (GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glBitmap) (GLsizei  width, GLsizei  height, GLfloat  xorig, GLfloat  yorig, GLfloat  xmove, GLfloat  ymove, const GLubyte * bitmap);
extern void (GLAPIENTRY *glBlendBarrierKHR) ();
extern void (GLAPIENTRY *glBlendBarrierNV) ();
extern void (GLAPIENTRY *glBlendColor) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
extern void (GLAPIENTRY *glBlendColorEXT) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
extern void (GLAPIENTRY *glBlendEquation) (GLenum  mode);
extern void (GLAPIENTRY *glBlendEquationEXT) (GLenum  mode);
extern void (GLAPIENTRY *glBlendEquationIndexedAMD) (GLuint  buf, GLenum  mode);
extern void (GLAPIENTRY *glBlendEquationSeparate) (GLenum  modeRGB, GLenum  modeAlpha);
extern void (GLAPIENTRY *glBlendEquationSeparateEXT) (GLenum  modeRGB, GLenum  modeAlpha);
extern void (GLAPIENTRY *glBlendEquationSeparateIndexedAMD) (GLuint  buf, GLenum  modeRGB, GLenum  modeAlpha);
extern void (GLAPIENTRY *glBlendEquationSeparatei) (GLuint  buf, GLenum  modeRGB, GLenum  modeAlpha);
extern void (GLAPIENTRY *glBlendEquationSeparateiARB) (GLuint  buf, GLenum  modeRGB, GLenum  modeAlpha);
extern void (GLAPIENTRY *glBlendEquationi) (GLuint  buf, GLenum  mode);
extern void (GLAPIENTRY *glBlendEquationiARB) (GLuint  buf, GLenum  mode);
extern void (GLAPIENTRY *glBlendFunc) (GLenum  sfactor, GLenum  dfactor);
extern void (GLAPIENTRY *glBlendFuncIndexedAMD) (GLuint  buf, GLenum  src, GLenum  dst);
extern void (GLAPIENTRY *glBlendFuncSeparate) (GLenum  sfactorRGB, GLenum  dfactorRGB, GLenum  sfactorAlpha, GLenum  dfactorAlpha);
extern void (GLAPIENTRY *glBlendFuncSeparateEXT) (GLenum  sfactorRGB, GLenum  dfactorRGB, GLenum  sfactorAlpha, GLenum  dfactorAlpha);
extern void (GLAPIENTRY *glBlendFuncSeparateIndexedAMD) (GLuint  buf, GLenum  srcRGB, GLenum  dstRGB, GLenum  srcAlpha, GLenum  dstAlpha);
extern void (GLAPIENTRY *glBlendFuncSeparatei) (GLuint  buf, GLenum  srcRGB, GLenum  dstRGB, GLenum  srcAlpha, GLenum  dstAlpha);
extern void (GLAPIENTRY *glBlendFuncSeparateiARB) (GLuint  buf, GLenum  srcRGB, GLenum  dstRGB, GLenum  srcAlpha, GLenum  dstAlpha);
extern void (GLAPIENTRY *glBlendFunci) (GLuint  buf, GLenum  src, GLenum  dst);
extern void (GLAPIENTRY *glBlendFunciARB) (GLuint  buf, GLenum  src, GLenum  dst);
extern void (GLAPIENTRY *glBlendParameteriNV) (GLenum  pname, GLint  value);
extern void (GLAPIENTRY *glBlitFramebuffer) (GLint  srcX0, GLint  srcY0, GLint  srcX1, GLint  srcY1, GLint  dstX0, GLint  dstY0, GLint  dstX1, GLint  dstY1, GLbitfield  mask, GLenum  filter);
extern void (GLAPIENTRY *glBlitFramebufferEXT) (GLint  srcX0, GLint  srcY0, GLint  srcX1, GLint  srcY1, GLint  dstX0, GLint  dstY0, GLint  dstX1, GLint  dstY1, GLbitfield  mask, GLenum  filter);
extern void (GLAPIENTRY *glBlitNamedFramebuffer) (GLuint  readFramebuffer, GLuint  drawFramebuffer, GLint  srcX0, GLint  srcY0, GLint  srcX1, GLint  srcY1, GLint  dstX0, GLint  dstY0, GLint  dstX1, GLint  dstY1, GLbitfield  mask, GLenum  filter);
extern void (GLAPIENTRY *glBufferAddressRangeNV) (GLenum  pname, GLuint  index, GLuint64EXT  address, GLsizeiptr  length);
extern void (GLAPIENTRY *glBufferAttachMemoryNV) (GLenum  target, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glBufferData) (GLenum  target, GLsizeiptr  size, const void * data, GLenum  usage);
extern void (GLAPIENTRY *glBufferDataARB) (GLenum  target, GLsizeiptrARB  size, const void * data, GLenum  usage);
extern void (GLAPIENTRY *glBufferPageCommitmentARB) (GLenum  target, GLintptr  offset, GLsizeiptr  size, GLboolean  commit);
extern void (GLAPIENTRY *glBufferParameteriAPPLE) (GLenum  target, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glBufferStorage) (GLenum  target, GLsizeiptr  size, const void * data, GLbitfield  flags);
extern void (GLAPIENTRY *glBufferStorageExternalEXT) (GLenum  target, GLintptr  offset, GLsizeiptr  size, GLeglClientBufferEXT  clientBuffer, GLbitfield  flags);
extern void (GLAPIENTRY *glBufferStorageMemEXT) (GLenum  target, GLsizeiptr  size, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glBufferSubData) (GLenum  target, GLintptr  offset, GLsizeiptr  size, const void * data);
extern void (GLAPIENTRY *glBufferSubDataARB) (GLenum  target, GLintptrARB  offset, GLsizeiptrARB  size, const void * data);
extern void (GLAPIENTRY *glCallCommandListNV) (GLuint  list);
extern void (GLAPIENTRY *glCallList) (GLuint  list);
extern void (GLAPIENTRY *glCallLists) (GLsizei  n, GLenum  type, const void * lists);
extern GLenum (GLAPIENTRY *glCheckFramebufferStatus) (GLenum  target);
extern GLenum (GLAPIENTRY *glCheckFramebufferStatusEXT) (GLenum  target);
extern GLenum (GLAPIENTRY *glCheckNamedFramebufferStatus) (GLuint  framebuffer, GLenum  target);
extern GLenum (GLAPIENTRY *glCheckNamedFramebufferStatusEXT) (GLuint  framebuffer, GLenum  target);
extern void (GLAPIENTRY *glClampColor) (GLenum  target, GLenum  clamp);
extern void (GLAPIENTRY *glClampColorARB) (GLenum  target, GLenum  clamp);
extern void (GLAPIENTRY *glClear) (GLbitfield  mask);
extern void (GLAPIENTRY *glClearAccum) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
extern void (GLAPIENTRY *glClearBufferData) (GLenum  target, GLenum  internalformat, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glClearBufferSubData) (GLenum  target, GLenum  internalformat, GLintptr  offset, GLsizeiptr  size, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glClearBufferfi) (GLenum  buffer, GLint  drawbuffer, GLfloat  depth, GLint  stencil);
extern void (GLAPIENTRY *glClearBufferfv) (GLenum  buffer, GLint  drawbuffer, const GLfloat * value);
extern void (GLAPIENTRY *glClearBufferiv) (GLenum  buffer, GLint  drawbuffer, const GLint * value);
extern void (GLAPIENTRY *glClearBufferuiv) (GLenum  buffer, GLint  drawbuffer, const GLuint * value);
extern void (GLAPIENTRY *glClearColor) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
extern void (GLAPIENTRY *glClearColorIiEXT) (GLint  red, GLint  green, GLint  blue, GLint  alpha);
extern void (GLAPIENTRY *glClearColorIuiEXT) (GLuint  red, GLuint  green, GLuint  blue, GLuint  alpha);
extern void (GLAPIENTRY *glClearDepth) (GLdouble  depth);
extern void (GLAPIENTRY *glClearDepthdNV) (GLdouble  depth);
extern void (GLAPIENTRY *glClearDepthf) (GLfloat  d);
extern void (GLAPIENTRY *glClearIndex) (GLfloat  c);
extern void (GLAPIENTRY *glClearNamedBufferData) (GLuint  buffer, GLenum  internalformat, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glClearNamedBufferDataEXT) (GLuint  buffer, GLenum  internalformat, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glClearNamedBufferSubData) (GLuint  buffer, GLenum  internalformat, GLintptr  offset, GLsizeiptr  size, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glClearNamedBufferSubDataEXT) (GLuint  buffer, GLenum  internalformat, GLsizeiptr  offset, GLsizeiptr  size, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glClearNamedFramebufferfi) (GLuint  framebuffer, GLenum  buffer, GLint  drawbuffer, GLfloat  depth, GLint  stencil);
extern void (GLAPIENTRY *glClearNamedFramebufferfv) (GLuint  framebuffer, GLenum  buffer, GLint  drawbuffer, const GLfloat * value);
extern void (GLAPIENTRY *glClearNamedFramebufferiv) (GLuint  framebuffer, GLenum  buffer, GLint  drawbuffer, const GLint * value);
extern void (GLAPIENTRY *glClearNamedFramebufferuiv) (GLuint  framebuffer, GLenum  buffer, GLint  drawbuffer, const GLuint * value);
extern void (GLAPIENTRY *glClearStencil) (GLint  s);
extern void (GLAPIENTRY *glClearTexImage) (GLuint  texture, GLint  level, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glClearTexSubImage) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glClientActiveTexture) (GLenum  texture);
extern void (GLAPIENTRY *glClientActiveTextureARB) (GLenum  texture);
extern void (GLAPIENTRY *glClientAttribDefaultEXT) (GLbitfield  mask);
extern GLenum (GLAPIENTRY *glClientWaitSync) (GLsync  sync, GLbitfield  flags, GLuint64  timeout);
extern void (GLAPIENTRY *glClipControl) (GLenum  origin, GLenum  depth);
extern void (GLAPIENTRY *glClipPlane) (GLenum  plane, const GLdouble * equation);
extern void (GLAPIENTRY *glColor3b) (GLbyte  red, GLbyte  green, GLbyte  blue);
extern void (GLAPIENTRY *glColor3bv) (const GLbyte * v);
extern void (GLAPIENTRY *glColor3d) (GLdouble  red, GLdouble  green, GLdouble  blue);
extern void (GLAPIENTRY *glColor3dv) (const GLdouble * v);
extern void (GLAPIENTRY *glColor3f) (GLfloat  red, GLfloat  green, GLfloat  blue);
extern void (GLAPIENTRY *glColor3fv) (const GLfloat * v);
extern void (GLAPIENTRY *glColor3hNV) (GLhalfNV  red, GLhalfNV  green, GLhalfNV  blue);
extern void (GLAPIENTRY *glColor3hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glColor3i) (GLint  red, GLint  green, GLint  blue);
extern void (GLAPIENTRY *glColor3iv) (const GLint * v);
extern void (GLAPIENTRY *glColor3s) (GLshort  red, GLshort  green, GLshort  blue);
extern void (GLAPIENTRY *glColor3sv) (const GLshort * v);
extern void (GLAPIENTRY *glColor3ub) (GLubyte  red, GLubyte  green, GLubyte  blue);
extern void (GLAPIENTRY *glColor3ubv) (const GLubyte * v);
extern void (GLAPIENTRY *glColor3ui) (GLuint  red, GLuint  green, GLuint  blue);
extern void (GLAPIENTRY *glColor3uiv) (const GLuint * v);
extern void (GLAPIENTRY *glColor3us) (GLushort  red, GLushort  green, GLushort  blue);
extern void (GLAPIENTRY *glColor3usv) (const GLushort * v);
extern void (GLAPIENTRY *glColor4b) (GLbyte  red, GLbyte  green, GLbyte  blue, GLbyte  alpha);
extern void (GLAPIENTRY *glColor4bv) (const GLbyte * v);
extern void (GLAPIENTRY *glColor4d) (GLdouble  red, GLdouble  green, GLdouble  blue, GLdouble  alpha);
extern void (GLAPIENTRY *glColor4dv) (const GLdouble * v);
extern void (GLAPIENTRY *glColor4f) (GLfloat  red, GLfloat  green, GLfloat  blue, GLfloat  alpha);
extern void (GLAPIENTRY *glColor4fv) (const GLfloat * v);
extern void (GLAPIENTRY *glColor4hNV) (GLhalfNV  red, GLhalfNV  green, GLhalfNV  blue, GLhalfNV  alpha);
extern void (GLAPIENTRY *glColor4hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glColor4i) (GLint  red, GLint  green, GLint  blue, GLint  alpha);
extern void (GLAPIENTRY *glColor4iv) (const GLint * v);
extern void (GLAPIENTRY *glColor4s) (GLshort  red, GLshort  green, GLshort  blue, GLshort  alpha);
extern void (GLAPIENTRY *glColor4sv) (const GLshort * v);
extern void (GLAPIENTRY *glColor4ub) (GLubyte  red, GLubyte  green, GLubyte  blue, GLubyte  alpha);
extern void (GLAPIENTRY *glColor4ubv) (const GLubyte * v);
extern void (GLAPIENTRY *glColor4ui) (GLuint  red, GLuint  green, GLuint  blue, GLuint  alpha);
extern void (GLAPIENTRY *glColor4uiv) (const GLuint * v);
extern void (GLAPIENTRY *glColor4us) (GLushort  red, GLushort  green, GLushort  blue, GLushort  alpha);
extern void (GLAPIENTRY *glColor4usv) (const GLushort * v);
extern void (GLAPIENTRY *glColorFormatNV) (GLint  size, GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glColorMask) (GLboolean  red, GLboolean  green, GLboolean  blue, GLboolean  alpha);
extern void (GLAPIENTRY *glColorMaskIndexedEXT) (GLuint  index, GLboolean  r, GLboolean  g, GLboolean  b, GLboolean  a);
extern void (GLAPIENTRY *glColorMaski) (GLuint  index, GLboolean  r, GLboolean  g, GLboolean  b, GLboolean  a);
extern void (GLAPIENTRY *glColorMaterial) (GLenum  face, GLenum  mode);
extern void (GLAPIENTRY *glColorP3ui) (GLenum  type, GLuint  color);
extern void (GLAPIENTRY *glColorP3uiv) (GLenum  type, const GLuint * color);
extern void (GLAPIENTRY *glColorP4ui) (GLenum  type, GLuint  color);
extern void (GLAPIENTRY *glColorP4uiv) (GLenum  type, const GLuint * color);
extern void (GLAPIENTRY *glColorPointer) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glColorPointerEXT) (GLint  size, GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
extern void (GLAPIENTRY *glColorPointervINTEL) (GLint  size, GLenum  type, const void ** pointer);
extern void (GLAPIENTRY *glColorSubTable) (GLenum  target, GLsizei  start, GLsizei  count, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glColorSubTableEXT) (GLenum  target, GLsizei  start, GLsizei  count, GLenum  format, GLenum  type, const void * data);
extern void (GLAPIENTRY *glColorTable) (GLenum  target, GLenum  internalformat, GLsizei  width, GLenum  format, GLenum  type, const void * table);
extern void (GLAPIENTRY *glColorTableEXT) (GLenum  target, GLenum  internalFormat, GLsizei  width, GLenum  format, GLenum  type, const void * table);
extern void (GLAPIENTRY *glColorTableParameterfv) (GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glColorTableParameteriv) (GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glCombinerInputNV) (GLenum  stage, GLenum  portion, GLenum  variable, GLenum  input, GLenum  mapping, GLenum  componentUsage);
extern void (GLAPIENTRY *glCombinerOutputNV) (GLenum  stage, GLenum  portion, GLenum  abOutput, GLenum  cdOutput, GLenum  sumOutput, GLenum  scale, GLenum  bias, GLboolean  abDotProduct, GLboolean  cdDotProduct, GLboolean  muxSum);
extern void (GLAPIENTRY *glCombinerParameterfNV) (GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glCombinerParameterfvNV) (GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glCombinerParameteriNV) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glCombinerParameterivNV) (GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glCombinerStageParameterfvNV) (GLenum  stage, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glCommandListSegmentsNV) (GLuint  list, GLuint  segments);
extern void (GLAPIENTRY *glCompileCommandListNV) (GLuint  list);
extern void (GLAPIENTRY *glCompileShader) (GLuint  shader);
extern void (GLAPIENTRY *glCompileShaderARB) (GLhandleARB  shaderObj);
extern void (GLAPIENTRY *glCompileShaderIncludeARB) (GLuint  shader, GLsizei  count, const GLchar *const* path, const GLint * length);
extern void (GLAPIENTRY *glCompressedMultiTexImage1DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLint  border, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedMultiTexImage2DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedMultiTexImage3DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedMultiTexSubImage1DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedMultiTexSubImage2DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedMultiTexSubImage3DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedTexImage1D) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLint  border, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexImage1DARB) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLint  border, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexImage2D) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexImage2DARB) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexImage3D) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexImage3DARB) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexSubImage1D) (GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexSubImage1DARB) (GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexSubImage2D) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexSubImage2DARB) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexSubImage3D) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTexSubImage3DARB) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTextureImage1DEXT) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLint  border, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedTextureImage2DEXT) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedTextureImage3DEXT) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedTextureSubImage1D) (GLuint  texture, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTextureSubImage1DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedTextureSubImage2D) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTextureSubImage2DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glCompressedTextureSubImage3D) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * data);
extern void (GLAPIENTRY *glCompressedTextureSubImage3DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLsizei  imageSize, const void * bits);
extern void (GLAPIENTRY *glConservativeRasterParameterfNV) (GLenum  pname, GLfloat  value);
extern void (GLAPIENTRY *glConservativeRasterParameteriNV) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glConvolutionFilter1D) (GLenum  target, GLenum  internalformat, GLsizei  width, GLenum  format, GLenum  type, const void * image);
extern void (GLAPIENTRY *glConvolutionFilter1DEXT) (GLenum  target, GLenum  internalformat, GLsizei  width, GLenum  format, GLenum  type, const void * image);
extern void (GLAPIENTRY *glConvolutionFilter2D) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * image);
extern void (GLAPIENTRY *glConvolutionFilter2DEXT) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * image);
extern void (GLAPIENTRY *glConvolutionParameterf) (GLenum  target, GLenum  pname, GLfloat  params);
extern void (GLAPIENTRY *glConvolutionParameterfEXT) (GLenum  target, GLenum  pname, GLfloat  params);
extern void (GLAPIENTRY *glConvolutionParameterfv) (GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glConvolutionParameterfvEXT) (GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glConvolutionParameteri) (GLenum  target, GLenum  pname, GLint  params);
extern void (GLAPIENTRY *glConvolutionParameteriEXT) (GLenum  target, GLenum  pname, GLint  params);
extern void (GLAPIENTRY *glConvolutionParameteriv) (GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glConvolutionParameterivEXT) (GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glCopyBufferSubData) (GLenum  readTarget, GLenum  writeTarget, GLintptr  readOffset, GLintptr  writeOffset, GLsizeiptr  size);
extern void (GLAPIENTRY *glCopyColorSubTable) (GLenum  target, GLsizei  start, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyColorSubTableEXT) (GLenum  target, GLsizei  start, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyColorTable) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyConvolutionFilter1D) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyConvolutionFilter1DEXT) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyConvolutionFilter2D) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyConvolutionFilter2DEXT) (GLenum  target, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyImageSubData) (GLuint  srcName, GLenum  srcTarget, GLint  srcLevel, GLint  srcX, GLint  srcY, GLint  srcZ, GLuint  dstName, GLenum  dstTarget, GLint  dstLevel, GLint  dstX, GLint  dstY, GLint  dstZ, GLsizei  srcWidth, GLsizei  srcHeight, GLsizei  srcDepth);
extern void (GLAPIENTRY *glCopyImageSubDataNV) (GLuint  srcName, GLenum  srcTarget, GLint  srcLevel, GLint  srcX, GLint  srcY, GLint  srcZ, GLuint  dstName, GLenum  dstTarget, GLint  dstLevel, GLint  dstX, GLint  dstY, GLint  dstZ, GLsizei  width, GLsizei  height, GLsizei  depth);
extern void (GLAPIENTRY *glCopyMultiTexImage1DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLint  border);
extern void (GLAPIENTRY *glCopyMultiTexImage2DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLint  border);
extern void (GLAPIENTRY *glCopyMultiTexSubImage1DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyMultiTexSubImage2DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyMultiTexSubImage3DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyNamedBufferSubData) (GLuint  readBuffer, GLuint  writeBuffer, GLintptr  readOffset, GLintptr  writeOffset, GLsizeiptr  size);
extern void (GLAPIENTRY *glCopyPathNV) (GLuint  resultPath, GLuint  srcPath);
extern void (GLAPIENTRY *glCopyPixels) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  type);
extern void (GLAPIENTRY *glCopyTexImage1D) (GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLint  border);
extern void (GLAPIENTRY *glCopyTexImage1DEXT) (GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLint  border);
extern void (GLAPIENTRY *glCopyTexImage2D) (GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLint  border);
extern void (GLAPIENTRY *glCopyTexImage2DEXT) (GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLint  border);
extern void (GLAPIENTRY *glCopyTexSubImage1D) (GLenum  target, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyTexSubImage1DEXT) (GLenum  target, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyTexSubImage2D) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyTexSubImage2DEXT) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyTexSubImage3D) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyTexSubImage3DEXT) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyTextureImage1DEXT) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLint  border);
extern void (GLAPIENTRY *glCopyTextureImage2DEXT) (GLuint  texture, GLenum  target, GLint  level, GLenum  internalformat, GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLint  border);
extern void (GLAPIENTRY *glCopyTextureSubImage1D) (GLuint  texture, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyTextureSubImage1DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  x, GLint  y, GLsizei  width);
extern void (GLAPIENTRY *glCopyTextureSubImage2D) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyTextureSubImage2DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyTextureSubImage3D) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCopyTextureSubImage3DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glCoverFillPathInstancedNV) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLenum  coverMode, GLenum  transformType, const GLfloat * transformValues);
extern void (GLAPIENTRY *glCoverFillPathNV) (GLuint  path, GLenum  coverMode);
extern void (GLAPIENTRY *glCoverStrokePathInstancedNV) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLenum  coverMode, GLenum  transformType, const GLfloat * transformValues);
extern void (GLAPIENTRY *glCoverStrokePathNV) (GLuint  path, GLenum  coverMode);
extern void (GLAPIENTRY *glCoverageModulationNV) (GLenum  components);
extern void (GLAPIENTRY *glCoverageModulationTableNV) (GLsizei  n, const GLfloat * v);
extern void (GLAPIENTRY *glCreateBuffers) (GLsizei  n, GLuint * buffers);
extern void (GLAPIENTRY *glCreateCommandListsNV) (GLsizei  n, GLuint * lists);
extern void (GLAPIENTRY *glCreateFramebuffers) (GLsizei  n, GLuint * framebuffers);
extern void (GLAPIENTRY *glCreateMemoryObjectsEXT) (GLsizei  n, GLuint * memoryObjects);
extern void (GLAPIENTRY *glCreatePerfQueryINTEL) (GLuint  queryId, GLuint * queryHandle);
extern GLuint (GLAPIENTRY *glCreateProgram) ();
extern GLhandleARB (GLAPIENTRY *glCreateProgramObjectARB) ();
extern void (GLAPIENTRY *glCreateProgramPipelines) (GLsizei  n, GLuint * pipelines);
extern void (GLAPIENTRY *glCreateQueries) (GLenum  target, GLsizei  n, GLuint * ids);
extern void (GLAPIENTRY *glCreateRenderbuffers) (GLsizei  n, GLuint * renderbuffers);
extern void (GLAPIENTRY *glCreateSamplers) (GLsizei  n, GLuint * samplers);
extern GLuint (GLAPIENTRY *glCreateShader) (GLenum  type);
extern GLhandleARB (GLAPIENTRY *glCreateShaderObjectARB) (GLenum  shaderType);
extern GLuint (GLAPIENTRY *glCreateShaderProgramEXT) (GLenum  type, const GLchar * string);
extern GLuint (GLAPIENTRY *glCreateShaderProgramv) (GLenum  type, GLsizei  count, const GLchar *const* strings);
extern GLuint (GLAPIENTRY *glCreateShaderProgramvEXT) (GLenum  type, GLsizei  count, const GLchar ** strings);
extern void (GLAPIENTRY *glCreateStatesNV) (GLsizei  n, GLuint * states);
extern GLsync (GLAPIENTRY *glCreateSyncFromCLeventARB) (struct _cl_context * context, struct _cl_event * event, GLbitfield  flags);
extern void (GLAPIENTRY *glCreateTextures) (GLenum  target, GLsizei  n, GLuint * textures);
extern void (GLAPIENTRY *glCreateTransformFeedbacks) (GLsizei  n, GLuint * ids);
extern void (GLAPIENTRY *glCreateVertexArrays) (GLsizei  n, GLuint * arrays);
extern void (GLAPIENTRY *glCullFace) (GLenum  mode);
extern void (GLAPIENTRY *glCullParameterdvEXT) (GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glCullParameterfvEXT) (GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glCurrentPaletteMatrixARB) (GLint  index);
extern void (GLAPIENTRY *glDebugMessageCallback) (GLDEBUGPROC  callback, const void * userParam);
extern void (GLAPIENTRY *glDebugMessageCallbackAMD) (GLDEBUGPROCAMD  callback, void * userParam);
extern void (GLAPIENTRY *glDebugMessageCallbackARB) (GLDEBUGPROCARB  callback, const void * userParam);
extern void (GLAPIENTRY *glDebugMessageCallbackKHR) (GLDEBUGPROCKHR  callback, const void * userParam);
extern void (GLAPIENTRY *glDebugMessageControl) (GLenum  source, GLenum  type, GLenum  severity, GLsizei  count, const GLuint * ids, GLboolean  enabled);
extern void (GLAPIENTRY *glDebugMessageControlARB) (GLenum  source, GLenum  type, GLenum  severity, GLsizei  count, const GLuint * ids, GLboolean  enabled);
extern void (GLAPIENTRY *glDebugMessageControlKHR) (GLenum  source, GLenum  type, GLenum  severity, GLsizei  count, const GLuint * ids, GLboolean  enabled);
extern void (GLAPIENTRY *glDebugMessageEnableAMD) (GLenum  category, GLenum  severity, GLsizei  count, const GLuint * ids, GLboolean  enabled);
extern void (GLAPIENTRY *glDebugMessageInsert) (GLenum  source, GLenum  type, GLuint  id, GLenum  severity, GLsizei  length, const GLchar * buf);
extern void (GLAPIENTRY *glDebugMessageInsertAMD) (GLenum  category, GLenum  severity, GLuint  id, GLsizei  length, const GLchar * buf);
extern void (GLAPIENTRY *glDebugMessageInsertARB) (GLenum  source, GLenum  type, GLuint  id, GLenum  severity, GLsizei  length, const GLchar * buf);
extern void (GLAPIENTRY *glDebugMessageInsertKHR) (GLenum  source, GLenum  type, GLuint  id, GLenum  severity, GLsizei  length, const GLchar * buf);
extern void (GLAPIENTRY *glDeleteBuffers) (GLsizei  n, const GLuint * buffers);
extern void (GLAPIENTRY *glDeleteBuffersARB) (GLsizei  n, const GLuint * buffers);
extern void (GLAPIENTRY *glDeleteCommandListsNV) (GLsizei  n, const GLuint * lists);
extern void (GLAPIENTRY *glDeleteFencesAPPLE) (GLsizei  n, const GLuint * fences);
extern void (GLAPIENTRY *glDeleteFencesNV) (GLsizei  n, const GLuint * fences);
extern void (GLAPIENTRY *glDeleteFramebuffers) (GLsizei  n, const GLuint * framebuffers);
extern void (GLAPIENTRY *glDeleteFramebuffersEXT) (GLsizei  n, const GLuint * framebuffers);
extern void (GLAPIENTRY *glDeleteLists) (GLuint  list, GLsizei  range);
extern void (GLAPIENTRY *glDeleteMemoryObjectsEXT) (GLsizei  n, const GLuint * memoryObjects);
extern void (GLAPIENTRY *glDeleteNamedStringARB) (GLint  namelen, const GLchar * name);
extern void (GLAPIENTRY *glDeleteNamesAMD) (GLenum  identifier, GLuint  num, const GLuint * names);
extern void (GLAPIENTRY *glDeleteObjectARB) (GLhandleARB  obj);
extern void (GLAPIENTRY *glDeleteOcclusionQueriesNV) (GLsizei  n, const GLuint * ids);
extern void (GLAPIENTRY *glDeletePathsNV) (GLuint  path, GLsizei  range);
extern void (GLAPIENTRY *glDeletePerfMonitorsAMD) (GLsizei  n, GLuint * monitors);
extern void (GLAPIENTRY *glDeletePerfQueryINTEL) (GLuint  queryHandle);
extern void (GLAPIENTRY *glDeleteProgram) (GLuint  program);
extern void (GLAPIENTRY *glDeleteProgramPipelines) (GLsizei  n, const GLuint * pipelines);
extern void (GLAPIENTRY *glDeleteProgramPipelinesEXT) (GLsizei  n, const GLuint * pipelines);
extern void (GLAPIENTRY *glDeleteProgramsARB) (GLsizei  n, const GLuint * programs);
extern void (GLAPIENTRY *glDeleteProgramsNV) (GLsizei  n, const GLuint * programs);
extern void (GLAPIENTRY *glDeleteQueries) (GLsizei  n, const GLuint * ids);
extern void (GLAPIENTRY *glDeleteQueriesARB) (GLsizei  n, const GLuint * ids);
extern void (GLAPIENTRY *glDeleteQueryResourceTagNV) (GLsizei  n, const GLint * tagIds);
extern void (GLAPIENTRY *glDeleteRenderbuffers) (GLsizei  n, const GLuint * renderbuffers);
extern void (GLAPIENTRY *glDeleteRenderbuffersEXT) (GLsizei  n, const GLuint * renderbuffers);
extern void (GLAPIENTRY *glDeleteSamplers) (GLsizei  count, const GLuint * samplers);
extern void (GLAPIENTRY *glDeleteSemaphoresEXT) (GLsizei  n, const GLuint * semaphores);
extern void (GLAPIENTRY *glDeleteShader) (GLuint  shader);
extern void (GLAPIENTRY *glDeleteStatesNV) (GLsizei  n, const GLuint * states);
extern void (GLAPIENTRY *glDeleteSync) (GLsync  sync);
extern void (GLAPIENTRY *glDeleteTextures) (GLsizei  n, const GLuint * textures);
extern void (GLAPIENTRY *glDeleteTexturesEXT) (GLsizei  n, const GLuint * textures);
extern void (GLAPIENTRY *glDeleteTransformFeedbacks) (GLsizei  n, const GLuint * ids);
extern void (GLAPIENTRY *glDeleteTransformFeedbacksNV) (GLsizei  n, const GLuint * ids);
extern void (GLAPIENTRY *glDeleteVertexArrays) (GLsizei  n, const GLuint * arrays);
extern void (GLAPIENTRY *glDeleteVertexArraysAPPLE) (GLsizei  n, const GLuint * arrays);
extern void (GLAPIENTRY *glDeleteVertexShaderEXT) (GLuint  id);
extern void (GLAPIENTRY *glDepthBoundsEXT) (GLclampd  zmin, GLclampd  zmax);
extern void (GLAPIENTRY *glDepthBoundsdNV) (GLdouble  zmin, GLdouble  zmax);
extern void (GLAPIENTRY *glDepthFunc) (GLenum  func);
extern void (GLAPIENTRY *glDepthMask) (GLboolean  flag);
extern void (GLAPIENTRY *glDepthRange) (GLdouble  n, GLdouble  f);
extern void (GLAPIENTRY *glDepthRangeArraydvNV) (GLuint  first, GLsizei  count, const GLdouble * v);
extern void (GLAPIENTRY *glDepthRangeArrayv) (GLuint  first, GLsizei  count, const GLdouble * v);
extern void (GLAPIENTRY *glDepthRangeIndexed) (GLuint  index, GLdouble  n, GLdouble  f);
extern void (GLAPIENTRY *glDepthRangeIndexeddNV) (GLuint  index, GLdouble  n, GLdouble  f);
extern void (GLAPIENTRY *glDepthRangedNV) (GLdouble  zNear, GLdouble  zFar);
extern void (GLAPIENTRY *glDepthRangef) (GLfloat  n, GLfloat  f);
extern void (GLAPIENTRY *glDetachObjectARB) (GLhandleARB  containerObj, GLhandleARB  attachedObj);
extern void (GLAPIENTRY *glDetachShader) (GLuint  program, GLuint  shader);
extern void (GLAPIENTRY *glDisable) (GLenum  cap);
extern void (GLAPIENTRY *glDisableClientState) (GLenum  array);
extern void (GLAPIENTRY *glDisableClientStateIndexedEXT) (GLenum  array, GLuint  index);
extern void (GLAPIENTRY *glDisableClientStateiEXT) (GLenum  array, GLuint  index);
extern void (GLAPIENTRY *glDisableIndexedEXT) (GLenum  target, GLuint  index);
extern void (GLAPIENTRY *glDisableVariantClientStateEXT) (GLuint  id);
extern void (GLAPIENTRY *glDisableVertexArrayAttrib) (GLuint  vaobj, GLuint  index);
extern void (GLAPIENTRY *glDisableVertexArrayAttribEXT) (GLuint  vaobj, GLuint  index);
extern void (GLAPIENTRY *glDisableVertexArrayEXT) (GLuint  vaobj, GLenum  array);
extern void (GLAPIENTRY *glDisableVertexAttribAPPLE) (GLuint  index, GLenum  pname);
extern void (GLAPIENTRY *glDisableVertexAttribArray) (GLuint  index);
extern void (GLAPIENTRY *glDisableVertexAttribArrayARB) (GLuint  index);
extern void (GLAPIENTRY *glDisablei) (GLenum  target, GLuint  index);
extern void (GLAPIENTRY *glDispatchCompute) (GLuint  num_groups_x, GLuint  num_groups_y, GLuint  num_groups_z);
extern void (GLAPIENTRY *glDispatchComputeGroupSizeARB) (GLuint  num_groups_x, GLuint  num_groups_y, GLuint  num_groups_z, GLuint  group_size_x, GLuint  group_size_y, GLuint  group_size_z);
extern void (GLAPIENTRY *glDispatchComputeIndirect) (GLintptr  indirect);
extern void (GLAPIENTRY *glDrawArrays) (GLenum  mode, GLint  first, GLsizei  count);
extern void (GLAPIENTRY *glDrawArraysEXT) (GLenum  mode, GLint  first, GLsizei  count);
extern void (GLAPIENTRY *glDrawArraysIndirect) (GLenum  mode, const void * indirect);
extern void (GLAPIENTRY *glDrawArraysInstanced) (GLenum  mode, GLint  first, GLsizei  count, GLsizei  instancecount);
extern void (GLAPIENTRY *glDrawArraysInstancedARB) (GLenum  mode, GLint  first, GLsizei  count, GLsizei  primcount);
extern void (GLAPIENTRY *glDrawArraysInstancedBaseInstance) (GLenum  mode, GLint  first, GLsizei  count, GLsizei  instancecount, GLuint  baseinstance);
extern void (GLAPIENTRY *glDrawArraysInstancedEXT) (GLenum  mode, GLint  start, GLsizei  count, GLsizei  primcount);
extern void (GLAPIENTRY *glDrawBuffer) (GLenum  buf);
extern void (GLAPIENTRY *glDrawBuffers) (GLsizei  n, const GLenum * bufs);
extern void (GLAPIENTRY *glDrawBuffersARB) (GLsizei  n, const GLenum * bufs);
extern void (GLAPIENTRY *glDrawCommandsAddressNV) (GLenum  primitiveMode, const GLuint64 * indirects, const GLsizei * sizes, GLuint  count);
extern void (GLAPIENTRY *glDrawCommandsNV) (GLenum  primitiveMode, GLuint  buffer, const GLintptr * indirects, const GLsizei * sizes, GLuint  count);
extern void (GLAPIENTRY *glDrawCommandsStatesAddressNV) (const GLuint64 * indirects, const GLsizei * sizes, const GLuint * states, const GLuint * fbos, GLuint  count);
extern void (GLAPIENTRY *glDrawCommandsStatesNV) (GLuint  buffer, const GLintptr * indirects, const GLsizei * sizes, const GLuint * states, const GLuint * fbos, GLuint  count);
extern void (GLAPIENTRY *glDrawElementArrayAPPLE) (GLenum  mode, GLint  first, GLsizei  count);
extern void (GLAPIENTRY *glDrawElements) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices);
extern void (GLAPIENTRY *glDrawElementsBaseVertex) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLint  basevertex);
extern void (GLAPIENTRY *glDrawElementsIndirect) (GLenum  mode, GLenum  type, const void * indirect);
extern void (GLAPIENTRY *glDrawElementsInstanced) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  instancecount);
extern void (GLAPIENTRY *glDrawElementsInstancedARB) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  primcount);
extern void (GLAPIENTRY *glDrawElementsInstancedBaseInstance) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  instancecount, GLuint  baseinstance);
extern void (GLAPIENTRY *glDrawElementsInstancedBaseVertex) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  instancecount, GLint  basevertex);
extern void (GLAPIENTRY *glDrawElementsInstancedBaseVertexBaseInstance) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  instancecount, GLint  basevertex, GLuint  baseinstance);
extern void (GLAPIENTRY *glDrawElementsInstancedEXT) (GLenum  mode, GLsizei  count, GLenum  type, const void * indices, GLsizei  primcount);
extern void (GLAPIENTRY *glDrawMeshTasksNV) (GLuint  first, GLuint  count);
extern void (GLAPIENTRY *glDrawMeshTasksIndirectNV) (GLintptr  indirect);
extern void (GLAPIENTRY *glDrawPixels) (GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glDrawRangeElementArrayAPPLE) (GLenum  mode, GLuint  start, GLuint  end, GLint  first, GLsizei  count);
extern void (GLAPIENTRY *glDrawRangeElements) (GLenum  mode, GLuint  start, GLuint  end, GLsizei  count, GLenum  type, const void * indices);
extern void (GLAPIENTRY *glDrawRangeElementsBaseVertex) (GLenum  mode, GLuint  start, GLuint  end, GLsizei  count, GLenum  type, const void * indices, GLint  basevertex);
extern void (GLAPIENTRY *glDrawRangeElementsEXT) (GLenum  mode, GLuint  start, GLuint  end, GLsizei  count, GLenum  type, const void * indices);
extern void (GLAPIENTRY *glDrawTextureNV) (GLuint  texture, GLuint  sampler, GLfloat  x0, GLfloat  y0, GLfloat  x1, GLfloat  y1, GLfloat  z, GLfloat  s0, GLfloat  t0, GLfloat  s1, GLfloat  t1);
extern void (GLAPIENTRY *glDrawTransformFeedback) (GLenum  mode, GLuint  id);
extern void (GLAPIENTRY *glDrawTransformFeedbackInstanced) (GLenum  mode, GLuint  id, GLsizei  instancecount);
extern void (GLAPIENTRY *glDrawTransformFeedbackNV) (GLenum  mode, GLuint  id);
extern void (GLAPIENTRY *glDrawTransformFeedbackStream) (GLenum  mode, GLuint  id, GLuint  stream);
extern void (GLAPIENTRY *glDrawTransformFeedbackStreamInstanced) (GLenum  mode, GLuint  id, GLuint  stream, GLsizei  instancecount);
extern void (GLAPIENTRY *glEGLImageTargetTexStorageEXT) (GLenum  target, GLeglImageOES  image, const GLint*  attrib_list);
extern void (GLAPIENTRY *glEGLImageTargetTextureStorageEXT) (GLuint  texture, GLeglImageOES  image, const GLint*  attrib_list);
extern void (GLAPIENTRY *glEdgeFlag) (GLboolean  flag);
extern void (GLAPIENTRY *glEdgeFlagFormatNV) (GLsizei  stride);
extern void (GLAPIENTRY *glEdgeFlagPointer) (GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glEdgeFlagPointerEXT) (GLsizei  stride, GLsizei  count, const GLboolean * pointer);
extern void (GLAPIENTRY *glEdgeFlagv) (const GLboolean * flag);
extern void (GLAPIENTRY *glElementPointerAPPLE) (GLenum  type, const void * pointer);
extern void (GLAPIENTRY *glEnable) (GLenum  cap);
extern void (GLAPIENTRY *glEnableClientState) (GLenum  array);
extern void (GLAPIENTRY *glEnableClientStateIndexedEXT) (GLenum  array, GLuint  index);
extern void (GLAPIENTRY *glEnableClientStateiEXT) (GLenum  array, GLuint  index);
extern void (GLAPIENTRY *glEnableIndexedEXT) (GLenum  target, GLuint  index);
extern void (GLAPIENTRY *glEnableVariantClientStateEXT) (GLuint  id);
extern void (GLAPIENTRY *glEnableVertexArrayAttrib) (GLuint  vaobj, GLuint  index);
extern void (GLAPIENTRY *glEnableVertexArrayAttribEXT) (GLuint  vaobj, GLuint  index);
extern void (GLAPIENTRY *glEnableVertexArrayEXT) (GLuint  vaobj, GLenum  array);
extern void (GLAPIENTRY *glEnableVertexAttribAPPLE) (GLuint  index, GLenum  pname);
extern void (GLAPIENTRY *glEnableVertexAttribArray) (GLuint  index);
extern void (GLAPIENTRY *glEnableVertexAttribArrayARB) (GLuint  index);
extern void (GLAPIENTRY *glEnablei) (GLenum  target, GLuint  index);
extern void (GLAPIENTRY *glEnd) ();
extern void (GLAPIENTRY *glEndConditionalRender) ();
extern void (GLAPIENTRY *glEndConditionalRenderNV) ();
extern void (GLAPIENTRY *glEndList) ();
extern void (GLAPIENTRY *glEndOcclusionQueryNV) ();
extern void (GLAPIENTRY *glEndPerfMonitorAMD) (GLuint  monitor);
extern void (GLAPIENTRY *glEndPerfQueryINTEL) (GLuint  queryHandle);
extern void (GLAPIENTRY *glEndQuery) (GLenum  target);
extern void (GLAPIENTRY *glEndQueryARB) (GLenum  target);
extern void (GLAPIENTRY *glEndQueryIndexed) (GLenum  target, GLuint  index);
extern void (GLAPIENTRY *glEndTransformFeedback) ();
extern void (GLAPIENTRY *glEndTransformFeedbackEXT) ();
extern void (GLAPIENTRY *glEndTransformFeedbackNV) ();
extern void (GLAPIENTRY *glEndVertexShaderEXT) ();
extern void (GLAPIENTRY *glEndVideoCaptureNV) (GLuint  video_capture_slot);
extern void (GLAPIENTRY *glEvalCoord1d) (GLdouble  u);
extern void (GLAPIENTRY *glEvalCoord1dv) (const GLdouble * u);
extern void (GLAPIENTRY *glEvalCoord1f) (GLfloat  u);
extern void (GLAPIENTRY *glEvalCoord1fv) (const GLfloat * u);
extern void (GLAPIENTRY *glEvalCoord2d) (GLdouble  u, GLdouble  v);
extern void (GLAPIENTRY *glEvalCoord2dv) (const GLdouble * u);
extern void (GLAPIENTRY *glEvalCoord2f) (GLfloat  u, GLfloat  v);
extern void (GLAPIENTRY *glEvalCoord2fv) (const GLfloat * u);
extern void (GLAPIENTRY *glEvalMapsNV) (GLenum  target, GLenum  mode);
extern void (GLAPIENTRY *glEvalMesh1) (GLenum  mode, GLint  i1, GLint  i2);
extern void (GLAPIENTRY *glEvalMesh2) (GLenum  mode, GLint  i1, GLint  i2, GLint  j1, GLint  j2);
extern void (GLAPIENTRY *glEvalPoint1) (GLint  i);
extern void (GLAPIENTRY *glEvalPoint2) (GLint  i, GLint  j);
extern void (GLAPIENTRY *glEvaluateDepthValuesARB) ();
extern void (GLAPIENTRY *glExecuteProgramNV) (GLenum  target, GLuint  id, const GLfloat * params);
extern void (GLAPIENTRY *glExtractComponentEXT) (GLuint  res, GLuint  src, GLuint  num);
extern void (GLAPIENTRY *glFeedbackBuffer) (GLsizei  size, GLenum  type, GLfloat * buffer);
extern GLsync (GLAPIENTRY *glFenceSync) (GLenum  condition, GLbitfield  flags);
extern void (GLAPIENTRY *glFinalCombinerInputNV) (GLenum  variable, GLenum  input, GLenum  mapping, GLenum  componentUsage);
extern void (GLAPIENTRY *glFinish) ();
extern void (GLAPIENTRY *glFinishFenceAPPLE) (GLuint  fence);
extern void (GLAPIENTRY *glFinishFenceNV) (GLuint  fence);
extern void (GLAPIENTRY *glFinishObjectAPPLE) (GLenum  object, GLint  name);
extern void (GLAPIENTRY *glFlush) ();
extern void (GLAPIENTRY *glFlushMappedBufferRange) (GLenum  target, GLintptr  offset, GLsizeiptr  length);
extern void (GLAPIENTRY *glFlushMappedBufferRangeAPPLE) (GLenum  target, GLintptr  offset, GLsizeiptr  size);
extern void (GLAPIENTRY *glFlushMappedNamedBufferRange) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length);
extern void (GLAPIENTRY *glFlushMappedNamedBufferRangeEXT) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length);
extern void (GLAPIENTRY *glFlushPixelDataRangeNV) (GLenum  target);
extern void (GLAPIENTRY *glFlushVertexArrayRangeAPPLE) (GLsizei  length, void * pointer);
extern void (GLAPIENTRY *glFlushVertexArrayRangeNV) ();
extern void (GLAPIENTRY *glFogCoordFormatNV) (GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glFogCoordPointer) (GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glFogCoordPointerEXT) (GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glFogCoordd) (GLdouble  coord);
extern void (GLAPIENTRY *glFogCoorddEXT) (GLdouble  coord);
extern void (GLAPIENTRY *glFogCoorddv) (const GLdouble * coord);
extern void (GLAPIENTRY *glFogCoorddvEXT) (const GLdouble * coord);
extern void (GLAPIENTRY *glFogCoordf) (GLfloat  coord);
extern void (GLAPIENTRY *glFogCoordfEXT) (GLfloat  coord);
extern void (GLAPIENTRY *glFogCoordfv) (const GLfloat * coord);
extern void (GLAPIENTRY *glFogCoordfvEXT) (const GLfloat * coord);
extern void (GLAPIENTRY *glFogCoordhNV) (GLhalfNV  fog);
extern void (GLAPIENTRY *glFogCoordhvNV) (const GLhalfNV * fog);
extern void (GLAPIENTRY *glFogf) (GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glFogfv) (GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glFogi) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glFogiv) (GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glFragmentCoverageColorNV) (GLuint  color);
extern void (GLAPIENTRY *glFramebufferDrawBufferEXT) (GLuint  framebuffer, GLenum  mode);
extern void (GLAPIENTRY *glFramebufferDrawBuffersEXT) (GLuint  framebuffer, GLsizei  n, const GLenum * bufs);
extern void (GLAPIENTRY *glFramebufferFetchBarrierEXT) ();
extern void (GLAPIENTRY *glFramebufferParameteri) (GLenum  target, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glFramebufferReadBufferEXT) (GLuint  framebuffer, GLenum  mode);
extern void (GLAPIENTRY *glFramebufferRenderbuffer) (GLenum  target, GLenum  attachment, GLenum  renderbuffertarget, GLuint  renderbuffer);
extern void (GLAPIENTRY *glFramebufferRenderbufferEXT) (GLenum  target, GLenum  attachment, GLenum  renderbuffertarget, GLuint  renderbuffer);
extern void (GLAPIENTRY *glFramebufferSampleLocationsfvARB) (GLenum  target, GLuint  start, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glFramebufferSampleLocationsfvNV) (GLenum  target, GLuint  start, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glFramebufferSamplePositionsfvAMD) (GLenum  target, GLuint  numsamples, GLuint  pixelindex, const GLfloat * values);
extern void (GLAPIENTRY *glFramebufferTexture) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glFramebufferTexture1D) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glFramebufferTexture1DEXT) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glFramebufferTexture2D) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glFramebufferTexture2DEXT) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glFramebufferTexture3D) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level, GLint  zoffset);
extern void (GLAPIENTRY *glFramebufferTexture3DEXT) (GLenum  target, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level, GLint  zoffset);
extern void (GLAPIENTRY *glFramebufferTextureARB) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glFramebufferTextureEXT) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glFramebufferTextureFaceARB) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLenum  face);
extern void (GLAPIENTRY *glFramebufferTextureFaceEXT) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLenum  face);
extern void (GLAPIENTRY *glFramebufferTextureLayer) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
extern void (GLAPIENTRY *glFramebufferTextureLayerARB) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
extern void (GLAPIENTRY *glFramebufferTextureLayerEXT) (GLenum  target, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
extern void (GLAPIENTRY *glFrontFace) (GLenum  mode);
extern void (GLAPIENTRY *glFrustum) (GLdouble  left, GLdouble  right, GLdouble  bottom, GLdouble  top, GLdouble  zNear, GLdouble  zFar);
extern void (GLAPIENTRY *glGenBuffers) (GLsizei  n, GLuint * buffers);
extern void (GLAPIENTRY *glGenBuffersARB) (GLsizei  n, GLuint * buffers);
extern void (GLAPIENTRY *glGenFencesAPPLE) (GLsizei  n, GLuint * fences);
extern void (GLAPIENTRY *glGenFencesNV) (GLsizei  n, GLuint * fences);
extern void (GLAPIENTRY *glGenFramebuffers) (GLsizei  n, GLuint * framebuffers);
extern void (GLAPIENTRY *glGenFramebuffersEXT) (GLsizei  n, GLuint * framebuffers);
extern GLuint (GLAPIENTRY *glGenLists) (GLsizei  range);
extern void (GLAPIENTRY *glGenNamesAMD) (GLenum  identifier, GLuint  num, GLuint * names);
extern void (GLAPIENTRY *glGenOcclusionQueriesNV) (GLsizei  n, GLuint * ids);
extern GLuint (GLAPIENTRY *glGenPathsNV) (GLsizei  range);
extern void (GLAPIENTRY *glGenPerfMonitorsAMD) (GLsizei  n, GLuint * monitors);
extern void (GLAPIENTRY *glGenProgramPipelines) (GLsizei  n, GLuint * pipelines);
extern void (GLAPIENTRY *glGenProgramPipelinesEXT) (GLsizei  n, GLuint * pipelines);
extern void (GLAPIENTRY *glGenProgramsARB) (GLsizei  n, GLuint * programs);
extern void (GLAPIENTRY *glGenProgramsNV) (GLsizei  n, GLuint * programs);
extern void (GLAPIENTRY *glGenQueries) (GLsizei  n, GLuint * ids);
extern void (GLAPIENTRY *glGenQueriesARB) (GLsizei  n, GLuint * ids);
extern void (GLAPIENTRY *glGenQueryResourceTagNV) (GLsizei  n, GLint * tagIds);
extern void (GLAPIENTRY *glGenRenderbuffers) (GLsizei  n, GLuint * renderbuffers);
extern void (GLAPIENTRY *glGenRenderbuffersEXT) (GLsizei  n, GLuint * renderbuffers);
extern void (GLAPIENTRY *glGenSamplers) (GLsizei  count, GLuint * samplers);
extern void (GLAPIENTRY *glGenSemaphoresEXT) (GLsizei  n, GLuint * semaphores);
extern GLuint (GLAPIENTRY *glGenSymbolsEXT) (GLenum  datatype, GLenum  storagetype, GLenum  range, GLuint  components);
extern void (GLAPIENTRY *glGenTextures) (GLsizei  n, GLuint * textures);
extern void (GLAPIENTRY *glGenTexturesEXT) (GLsizei  n, GLuint * textures);
extern void (GLAPIENTRY *glGenTransformFeedbacks) (GLsizei  n, GLuint * ids);
extern void (GLAPIENTRY *glGenTransformFeedbacksNV) (GLsizei  n, GLuint * ids);
extern void (GLAPIENTRY *glGenVertexArrays) (GLsizei  n, GLuint * arrays);
extern void (GLAPIENTRY *glGenVertexArraysAPPLE) (GLsizei  n, GLuint * arrays);
extern GLuint (GLAPIENTRY *glGenVertexShadersEXT) (GLuint  range);
extern void (GLAPIENTRY *glGenerateMipmap) (GLenum  target);
extern void (GLAPIENTRY *glGenerateMipmapEXT) (GLenum  target);
extern void (GLAPIENTRY *glGenerateMultiTexMipmapEXT) (GLenum  texunit, GLenum  target);
extern void (GLAPIENTRY *glGenerateTextureMipmap) (GLuint  texture);
extern void (GLAPIENTRY *glGenerateTextureMipmapEXT) (GLuint  texture, GLenum  target);
extern void (GLAPIENTRY *glGetActiveAtomicCounterBufferiv) (GLuint  program, GLuint  bufferIndex, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetActiveAttrib) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLint * size, GLenum * type, GLchar * name);
extern void (GLAPIENTRY *glGetActiveAttribARB) (GLhandleARB  programObj, GLuint  index, GLsizei  maxLength, GLsizei * length, GLint * size, GLenum * type, GLcharARB * name);
extern void (GLAPIENTRY *glGetActiveSubroutineName) (GLuint  program, GLenum  shadertype, GLuint  index, GLsizei  bufSize, GLsizei * length, GLchar * name);
extern void (GLAPIENTRY *glGetActiveSubroutineUniformName) (GLuint  program, GLenum  shadertype, GLuint  index, GLsizei  bufSize, GLsizei * length, GLchar * name);
extern void (GLAPIENTRY *glGetActiveSubroutineUniformiv) (GLuint  program, GLenum  shadertype, GLuint  index, GLenum  pname, GLint * values);
extern void (GLAPIENTRY *glGetActiveUniform) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLint * size, GLenum * type, GLchar * name);
extern void (GLAPIENTRY *glGetActiveUniformARB) (GLhandleARB  programObj, GLuint  index, GLsizei  maxLength, GLsizei * length, GLint * size, GLenum * type, GLcharARB * name);
extern void (GLAPIENTRY *glGetActiveUniformBlockName) (GLuint  program, GLuint  uniformBlockIndex, GLsizei  bufSize, GLsizei * length, GLchar * uniformBlockName);
extern void (GLAPIENTRY *glGetActiveUniformBlockiv) (GLuint  program, GLuint  uniformBlockIndex, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetActiveUniformName) (GLuint  program, GLuint  uniformIndex, GLsizei  bufSize, GLsizei * length, GLchar * uniformName);
extern void (GLAPIENTRY *glGetActiveUniformsiv) (GLuint  program, GLsizei  uniformCount, const GLuint * uniformIndices, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetActiveVaryingNV) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
extern void (GLAPIENTRY *glGetAttachedObjectsARB) (GLhandleARB  containerObj, GLsizei  maxCount, GLsizei * count, GLhandleARB * obj);
extern void (GLAPIENTRY *glGetAttachedShaders) (GLuint  program, GLsizei  maxCount, GLsizei * count, GLuint * shaders);
extern GLint (GLAPIENTRY *glGetAttribLocation) (GLuint  program, const GLchar * name);
extern GLint (GLAPIENTRY *glGetAttribLocationARB) (GLhandleARB  programObj, const GLcharARB * name);
extern void (GLAPIENTRY *glGetBooleanIndexedvEXT) (GLenum  target, GLuint  index, GLboolean * data);
extern void (GLAPIENTRY *glGetBooleani_v) (GLenum  target, GLuint  index, GLboolean * data);
extern void (GLAPIENTRY *glGetBooleanv) (GLenum  pname, GLboolean * data);
extern void (GLAPIENTRY *glGetBufferParameteri64v) (GLenum  target, GLenum  pname, GLint64 * params);
extern void (GLAPIENTRY *glGetBufferParameteriv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetBufferParameterivARB) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetBufferParameterui64vNV) (GLenum  target, GLenum  pname, GLuint64EXT * params);
extern void (GLAPIENTRY *glGetBufferPointerv) (GLenum  target, GLenum  pname, void ** params);
extern void (GLAPIENTRY *glGetBufferPointervARB) (GLenum  target, GLenum  pname, void ** params);
extern void (GLAPIENTRY *glGetBufferSubData) (GLenum  target, GLintptr  offset, GLsizeiptr  size, void * data);
extern void (GLAPIENTRY *glGetBufferSubDataARB) (GLenum  target, GLintptrARB  offset, GLsizeiptrARB  size, void * data);
extern void (GLAPIENTRY *glGetClipPlane) (GLenum  plane, GLdouble * equation);
extern void (GLAPIENTRY *glGetColorTable) (GLenum  target, GLenum  format, GLenum  type, void * table);
extern void (GLAPIENTRY *glGetColorTableEXT) (GLenum  target, GLenum  format, GLenum  type, void * data);
extern void (GLAPIENTRY *glGetColorTableParameterfv) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetColorTableParameterfvEXT) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetColorTableParameteriv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetColorTableParameterivEXT) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetCombinerInputParameterfvNV) (GLenum  stage, GLenum  portion, GLenum  variable, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetCombinerInputParameterivNV) (GLenum  stage, GLenum  portion, GLenum  variable, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetCombinerOutputParameterfvNV) (GLenum  stage, GLenum  portion, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetCombinerOutputParameterivNV) (GLenum  stage, GLenum  portion, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetCombinerStageParameterfvNV) (GLenum  stage, GLenum  pname, GLfloat * params);
extern GLuint (GLAPIENTRY *glGetCommandHeaderNV) (GLenum  tokenID, GLuint  size);
extern void (GLAPIENTRY *glGetCompressedMultiTexImageEXT) (GLenum  texunit, GLenum  target, GLint  lod, void * img);
extern void (GLAPIENTRY *glGetCompressedTexImage) (GLenum  target, GLint  level, void * img);
extern void (GLAPIENTRY *glGetCompressedTexImageARB) (GLenum  target, GLint  level, void * img);
extern void (GLAPIENTRY *glGetCompressedTextureImage) (GLuint  texture, GLint  level, GLsizei  bufSize, void * pixels);
extern void (GLAPIENTRY *glGetCompressedTextureImageEXT) (GLuint  texture, GLenum  target, GLint  lod, void * img);
extern void (GLAPIENTRY *glGetCompressedTextureSubImage) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLsizei  bufSize, void * pixels);
extern void (GLAPIENTRY *glGetConvolutionFilter) (GLenum  target, GLenum  format, GLenum  type, void * image);
extern void (GLAPIENTRY *glGetConvolutionFilterEXT) (GLenum  target, GLenum  format, GLenum  type, void * image);
extern void (GLAPIENTRY *glGetConvolutionParameterfv) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetConvolutionParameterfvEXT) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetConvolutionParameteriv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetConvolutionParameterivEXT) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetCoverageModulationTableNV) (GLsizei  bufSize, GLfloat * v);
extern GLuint (GLAPIENTRY *glGetDebugMessageLog) (GLuint  count, GLsizei  bufSize, GLenum * sources, GLenum * types, GLuint * ids, GLenum * severities, GLsizei * lengths, GLchar * messageLog);
extern GLuint (GLAPIENTRY *glGetDebugMessageLogAMD) (GLuint  count, GLsizei  bufSize, GLenum * categories, GLuint * severities, GLuint * ids, GLsizei * lengths, GLchar * message);
extern GLuint (GLAPIENTRY *glGetDebugMessageLogARB) (GLuint  count, GLsizei  bufSize, GLenum * sources, GLenum * types, GLuint * ids, GLenum * severities, GLsizei * lengths, GLchar * messageLog);
extern GLuint (GLAPIENTRY *glGetDebugMessageLogKHR) (GLuint  count, GLsizei  bufSize, GLenum * sources, GLenum * types, GLuint * ids, GLenum * severities, GLsizei * lengths, GLchar * messageLog);
extern void (GLAPIENTRY *glGetDoubleIndexedvEXT) (GLenum  target, GLuint  index, GLdouble * data);
extern void (GLAPIENTRY *glGetDoublei_v) (GLenum  target, GLuint  index, GLdouble * data);
extern void (GLAPIENTRY *glGetDoublei_vEXT) (GLenum  pname, GLuint  index, GLdouble * params);
extern void (GLAPIENTRY *glGetDoublev) (GLenum  pname, GLdouble * data);
extern GLenum (GLAPIENTRY *glGetError) ();
extern void (GLAPIENTRY *glGetFenceivNV) (GLuint  fence, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetFinalCombinerInputParameterfvNV) (GLenum  variable, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetFinalCombinerInputParameterivNV) (GLenum  variable, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetFirstPerfQueryIdINTEL) (GLuint * queryId);
extern void (GLAPIENTRY *glGetFloatIndexedvEXT) (GLenum  target, GLuint  index, GLfloat * data);
extern void (GLAPIENTRY *glGetFloati_v) (GLenum  target, GLuint  index, GLfloat * data);
extern void (GLAPIENTRY *glGetFloati_vEXT) (GLenum  pname, GLuint  index, GLfloat * params);
extern void (GLAPIENTRY *glGetFloatv) (GLenum  pname, GLfloat * data);
extern GLint (GLAPIENTRY *glGetFragDataIndex) (GLuint  program, const GLchar * name);
extern GLint (GLAPIENTRY *glGetFragDataLocation) (GLuint  program, const GLchar * name);
extern GLint (GLAPIENTRY *glGetFragDataLocationEXT) (GLuint  program, const GLchar * name);
extern void (GLAPIENTRY *glGetFramebufferAttachmentParameteriv) (GLenum  target, GLenum  attachment, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetFramebufferAttachmentParameterivEXT) (GLenum  target, GLenum  attachment, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetFramebufferParameterfvAMD) (GLenum  target, GLenum  pname, GLuint  numsamples, GLuint  pixelindex, GLsizei  size, GLfloat * values);
extern void (GLAPIENTRY *glGetFramebufferParameteriv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetFramebufferParameterivEXT) (GLuint  framebuffer, GLenum  pname, GLint * params);
extern GLenum (GLAPIENTRY *glGetGraphicsResetStatus) ();
extern GLenum (GLAPIENTRY *glGetGraphicsResetStatusARB) ();
extern GLenum (GLAPIENTRY *glGetGraphicsResetStatusKHR) ();
extern GLhandleARB (GLAPIENTRY *glGetHandleARB) (GLenum  pname);
extern void (GLAPIENTRY *glGetHistogram) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, void * values);
extern void (GLAPIENTRY *glGetHistogramEXT) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, void * values);
extern void (GLAPIENTRY *glGetHistogramParameterfv) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetHistogramParameterfvEXT) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetHistogramParameteriv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetHistogramParameterivEXT) (GLenum  target, GLenum  pname, GLint * params);
extern GLuint64 (GLAPIENTRY *glGetImageHandleARB) (GLuint  texture, GLint  level, GLboolean  layered, GLint  layer, GLenum  format);
extern GLuint64 (GLAPIENTRY *glGetImageHandleNV) (GLuint  texture, GLint  level, GLboolean  layered, GLint  layer, GLenum  format);
extern void (GLAPIENTRY *glGetInfoLogARB) (GLhandleARB  obj, GLsizei  maxLength, GLsizei * length, GLcharARB * infoLog);
extern void (GLAPIENTRY *glGetInteger64i_v) (GLenum  target, GLuint  index, GLint64 * data);
extern void (GLAPIENTRY *glGetInteger64v) (GLenum  pname, GLint64 * data);
extern void (GLAPIENTRY *glGetIntegerIndexedvEXT) (GLenum  target, GLuint  index, GLint * data);
extern void (GLAPIENTRY *glGetIntegeri_v) (GLenum  target, GLuint  index, GLint * data);
extern void (GLAPIENTRY *glGetIntegerui64i_vNV) (GLenum  value, GLuint  index, GLuint64EXT * result);
extern void (GLAPIENTRY *glGetIntegerui64vNV) (GLenum  value, GLuint64EXT * result);
extern void (GLAPIENTRY *glGetIntegerv) (GLenum  pname, GLint * data);
extern void (GLAPIENTRY *glGetInternalformatSampleivNV) (GLenum  target, GLenum  internalformat, GLsizei  samples, GLenum  pname, GLsizei  count, GLint * params);
extern void (GLAPIENTRY *glGetInternalformati64v) (GLenum  target, GLenum  internalformat, GLenum  pname, GLsizei  count, GLint64 * params);
extern void (GLAPIENTRY *glGetInternalformativ) (GLenum  target, GLenum  internalformat, GLenum  pname, GLsizei  count, GLint * params);
extern void (GLAPIENTRY *glGetInvariantBooleanvEXT) (GLuint  id, GLenum  value, GLboolean * data);
extern void (GLAPIENTRY *glGetInvariantFloatvEXT) (GLuint  id, GLenum  value, GLfloat * data);
extern void (GLAPIENTRY *glGetInvariantIntegervEXT) (GLuint  id, GLenum  value, GLint * data);
extern void (GLAPIENTRY *glGetLightfv) (GLenum  light, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetLightiv) (GLenum  light, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetLocalConstantBooleanvEXT) (GLuint  id, GLenum  value, GLboolean * data);
extern void (GLAPIENTRY *glGetLocalConstantFloatvEXT) (GLuint  id, GLenum  value, GLfloat * data);
extern void (GLAPIENTRY *glGetLocalConstantIntegervEXT) (GLuint  id, GLenum  value, GLint * data);
extern void (GLAPIENTRY *glGetMapAttribParameterfvNV) (GLenum  target, GLuint  index, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMapAttribParameterivNV) (GLenum  target, GLuint  index, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMapControlPointsNV) (GLenum  target, GLuint  index, GLenum  type, GLsizei  ustride, GLsizei  vstride, GLboolean  packed, void * points);
extern void (GLAPIENTRY *glGetMapParameterfvNV) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMapParameterivNV) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMapdv) (GLenum  target, GLenum  query, GLdouble * v);
extern void (GLAPIENTRY *glGetMapfv) (GLenum  target, GLenum  query, GLfloat * v);
extern void (GLAPIENTRY *glGetMapiv) (GLenum  target, GLenum  query, GLint * v);
extern void (GLAPIENTRY *glGetMaterialfv) (GLenum  face, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMaterialiv) (GLenum  face, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMemoryObjectDetachedResourcesuivNV) (GLuint  memory, GLenum  pname, GLint  first, GLsizei  count, GLuint * params);
extern void (GLAPIENTRY *glGetMemoryObjectParameterivEXT) (GLuint  memoryObject, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMinmax) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, void * values);
extern void (GLAPIENTRY *glGetMinmaxEXT) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, void * values);
extern void (GLAPIENTRY *glGetMinmaxParameterfv) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMinmaxParameterfvEXT) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMinmaxParameteriv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMinmaxParameterivEXT) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMultiTexEnvfvEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMultiTexEnvivEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMultiTexGendvEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetMultiTexGenfvEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMultiTexGenivEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMultiTexImageEXT) (GLenum  texunit, GLenum  target, GLint  level, GLenum  format, GLenum  type, void * pixels);
extern void (GLAPIENTRY *glGetMultiTexLevelParameterfvEXT) (GLenum  texunit, GLenum  target, GLint  level, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMultiTexLevelParameterivEXT) (GLenum  texunit, GLenum  target, GLint  level, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMultiTexParameterIivEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMultiTexParameterIuivEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetMultiTexParameterfvEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetMultiTexParameterivEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetMultisamplefv) (GLenum  pname, GLuint  index, GLfloat * val);
extern void (GLAPIENTRY *glGetMultisamplefvNV) (GLenum  pname, GLuint  index, GLfloat * val);
extern void (GLAPIENTRY *glGetNamedBufferParameteri64v) (GLuint  buffer, GLenum  pname, GLint64 * params);
extern void (GLAPIENTRY *glGetNamedBufferParameteriv) (GLuint  buffer, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNamedBufferParameterivEXT) (GLuint  buffer, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNamedBufferParameterui64vNV) (GLuint  buffer, GLenum  pname, GLuint64EXT * params);
extern void (GLAPIENTRY *glGetNamedBufferPointerv) (GLuint  buffer, GLenum  pname, void ** params);
extern void (GLAPIENTRY *glGetNamedBufferPointervEXT) (GLuint  buffer, GLenum  pname, void ** params);
extern void (GLAPIENTRY *glGetNamedBufferSubData) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, void * data);
extern void (GLAPIENTRY *glGetNamedBufferSubDataEXT) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, void * data);
extern void (GLAPIENTRY *glGetNamedFramebufferParameterfvAMD) (GLuint  framebuffer, GLenum  pname, GLuint  numsamples, GLuint  pixelindex, GLsizei  size, GLfloat * values);
extern void (GLAPIENTRY *glGetNamedFramebufferAttachmentParameteriv) (GLuint  framebuffer, GLenum  attachment, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNamedFramebufferAttachmentParameterivEXT) (GLuint  framebuffer, GLenum  attachment, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNamedFramebufferParameteriv) (GLuint  framebuffer, GLenum  pname, GLint * param);
extern void (GLAPIENTRY *glGetNamedFramebufferParameterivEXT) (GLuint  framebuffer, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNamedProgramLocalParameterIivEXT) (GLuint  program, GLenum  target, GLuint  index, GLint * params);
extern void (GLAPIENTRY *glGetNamedProgramLocalParameterIuivEXT) (GLuint  program, GLenum  target, GLuint  index, GLuint * params);
extern void (GLAPIENTRY *glGetNamedProgramLocalParameterdvEXT) (GLuint  program, GLenum  target, GLuint  index, GLdouble * params);
extern void (GLAPIENTRY *glGetNamedProgramLocalParameterfvEXT) (GLuint  program, GLenum  target, GLuint  index, GLfloat * params);
extern void (GLAPIENTRY *glGetNamedProgramStringEXT) (GLuint  program, GLenum  target, GLenum  pname, void * string);
extern void (GLAPIENTRY *glGetNamedProgramivEXT) (GLuint  program, GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNamedRenderbufferParameteriv) (GLuint  renderbuffer, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNamedRenderbufferParameterivEXT) (GLuint  renderbuffer, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNamedStringARB) (GLint  namelen, const GLchar * name, GLsizei  bufSize, GLint * stringlen, GLchar * string);
extern void (GLAPIENTRY *glGetNamedStringivARB) (GLint  namelen, const GLchar * name, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetNextPerfQueryIdINTEL) (GLuint  queryId, GLuint * nextQueryId);
extern void (GLAPIENTRY *glGetObjectLabel) (GLenum  identifier, GLuint  name, GLsizei  bufSize, GLsizei * length, GLchar * label);
extern void (GLAPIENTRY *glGetObjectLabelEXT) (GLenum  type, GLuint  object, GLsizei  bufSize, GLsizei * length, GLchar * label);
extern void (GLAPIENTRY *glGetObjectLabelKHR) (GLenum  identifier, GLuint  name, GLsizei  bufSize, GLsizei * length, GLchar * label);
extern void (GLAPIENTRY *glGetObjectParameterfvARB) (GLhandleARB  obj, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetObjectParameterivAPPLE) (GLenum  objectType, GLuint  name, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetObjectParameterivARB) (GLhandleARB  obj, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetObjectPtrLabel) (const void * ptr, GLsizei  bufSize, GLsizei * length, GLchar * label);
extern void (GLAPIENTRY *glGetObjectPtrLabelKHR) (const void * ptr, GLsizei  bufSize, GLsizei * length, GLchar * label);
extern void (GLAPIENTRY *glGetOcclusionQueryivNV) (GLuint  id, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetOcclusionQueryuivNV) (GLuint  id, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetPathColorGenfvNV) (GLenum  color, GLenum  pname, GLfloat * value);
extern void (GLAPIENTRY *glGetPathColorGenivNV) (GLenum  color, GLenum  pname, GLint * value);
extern void (GLAPIENTRY *glGetPathCommandsNV) (GLuint  path, GLubyte * commands);
extern void (GLAPIENTRY *glGetPathCoordsNV) (GLuint  path, GLfloat * coords);
extern void (GLAPIENTRY *glGetPathDashArrayNV) (GLuint  path, GLfloat * dashArray);
extern GLfloat (GLAPIENTRY *glGetPathLengthNV) (GLuint  path, GLsizei  startSegment, GLsizei  numSegments);
extern void (GLAPIENTRY *glGetPathMetricRangeNV) (GLbitfield  metricQueryMask, GLuint  firstPathName, GLsizei  numPaths, GLsizei  stride, GLfloat * metrics);
extern void (GLAPIENTRY *glGetPathMetricsNV) (GLbitfield  metricQueryMask, GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLsizei  stride, GLfloat * metrics);
extern void (GLAPIENTRY *glGetPathParameterfvNV) (GLuint  path, GLenum  pname, GLfloat * value);
extern void (GLAPIENTRY *glGetPathParameterivNV) (GLuint  path, GLenum  pname, GLint * value);
extern void (GLAPIENTRY *glGetPathSpacingNV) (GLenum  pathListMode, GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLfloat  advanceScale, GLfloat  kerningScale, GLenum  transformType, GLfloat * returnedSpacing);
extern void (GLAPIENTRY *glGetPathTexGenfvNV) (GLenum  texCoordSet, GLenum  pname, GLfloat * value);
extern void (GLAPIENTRY *glGetPathTexGenivNV) (GLenum  texCoordSet, GLenum  pname, GLint * value);
extern void (GLAPIENTRY *glGetPerfCounterInfoINTEL) (GLuint  queryId, GLuint  counterId, GLuint  counterNameLength, GLchar * counterName, GLuint  counterDescLength, GLchar * counterDesc, GLuint * counterOffset, GLuint * counterDataSize, GLuint * counterTypeEnum, GLuint * counterDataTypeEnum, GLuint64 * rawCounterMaxValue);
extern void (GLAPIENTRY *glGetPerfMonitorCounterDataAMD) (GLuint  monitor, GLenum  pname, GLsizei  dataSize, GLuint * data, GLint * bytesWritten);
extern void (GLAPIENTRY *glGetPerfMonitorCounterInfoAMD) (GLuint  group, GLuint  counter, GLenum  pname, void * data);
extern void (GLAPIENTRY *glGetPerfMonitorCounterStringAMD) (GLuint  group, GLuint  counter, GLsizei  bufSize, GLsizei * length, GLchar * counterString);
extern void (GLAPIENTRY *glGetPerfMonitorCountersAMD) (GLuint  group, GLint * numCounters, GLint * maxActiveCounters, GLsizei  counterSize, GLuint * counters);
extern void (GLAPIENTRY *glGetPerfMonitorGroupStringAMD) (GLuint  group, GLsizei  bufSize, GLsizei * length, GLchar * groupString);
extern void (GLAPIENTRY *glGetPerfMonitorGroupsAMD) (GLint * numGroups, GLsizei  groupsSize, GLuint * groups);
extern void (GLAPIENTRY *glGetPerfQueryDataINTEL) (GLuint  queryHandle, GLuint  flags, GLsizei  dataSize, void * data, GLuint * bytesWritten);
extern void (GLAPIENTRY *glGetPerfQueryIdByNameINTEL) (GLchar * queryName, GLuint * queryId);
extern void (GLAPIENTRY *glGetPerfQueryInfoINTEL) (GLuint  queryId, GLuint  queryNameLength, GLchar * queryName, GLuint * dataSize, GLuint * noCounters, GLuint * noInstances, GLuint * capsMask);
extern void (GLAPIENTRY *glGetPixelMapfv) (GLenum  map, GLfloat * values);
extern void (GLAPIENTRY *glGetPixelMapuiv) (GLenum  map, GLuint * values);
extern void (GLAPIENTRY *glGetPixelMapusv) (GLenum  map, GLushort * values);
extern void (GLAPIENTRY *glGetPixelTransformParameterfvEXT) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetPixelTransformParameterivEXT) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetPointerIndexedvEXT) (GLenum  target, GLuint  index, void ** data);
extern void (GLAPIENTRY *glGetPointeri_vEXT) (GLenum  pname, GLuint  index, void ** params);
extern void (GLAPIENTRY *glGetPointerv) (GLenum  pname, void ** params);
extern void (GLAPIENTRY *glGetPointervEXT) (GLenum  pname, void ** params);
extern void (GLAPIENTRY *glGetPointervKHR) (GLenum  pname, void ** params);
extern void (GLAPIENTRY *glGetPolygonStipple) (GLubyte * mask);
extern void (GLAPIENTRY *glGetProgramBinary) (GLuint  program, GLsizei  bufSize, GLsizei * length, GLenum * binaryFormat, void * binary);
extern void (GLAPIENTRY *glGetProgramEnvParameterIivNV) (GLenum  target, GLuint  index, GLint * params);
extern void (GLAPIENTRY *glGetProgramEnvParameterIuivNV) (GLenum  target, GLuint  index, GLuint * params);
extern void (GLAPIENTRY *glGetProgramEnvParameterdvARB) (GLenum  target, GLuint  index, GLdouble * params);
extern void (GLAPIENTRY *glGetProgramEnvParameterfvARB) (GLenum  target, GLuint  index, GLfloat * params);
extern void (GLAPIENTRY *glGetProgramInfoLog) (GLuint  program, GLsizei  bufSize, GLsizei * length, GLchar * infoLog);
extern void (GLAPIENTRY *glGetProgramInterfaceiv) (GLuint  program, GLenum  programInterface, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetProgramLocalParameterIivNV) (GLenum  target, GLuint  index, GLint * params);
extern void (GLAPIENTRY *glGetProgramLocalParameterIuivNV) (GLenum  target, GLuint  index, GLuint * params);
extern void (GLAPIENTRY *glGetProgramLocalParameterdvARB) (GLenum  target, GLuint  index, GLdouble * params);
extern void (GLAPIENTRY *glGetProgramLocalParameterfvARB) (GLenum  target, GLuint  index, GLfloat * params);
extern void (GLAPIENTRY *glGetProgramNamedParameterdvNV) (GLuint  id, GLsizei  len, const GLubyte * name, GLdouble * params);
extern void (GLAPIENTRY *glGetProgramNamedParameterfvNV) (GLuint  id, GLsizei  len, const GLubyte * name, GLfloat * params);
extern void (GLAPIENTRY *glGetProgramParameterdvNV) (GLenum  target, GLuint  index, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetProgramParameterfvNV) (GLenum  target, GLuint  index, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetProgramPipelineInfoLog) (GLuint  pipeline, GLsizei  bufSize, GLsizei * length, GLchar * infoLog);
extern void (GLAPIENTRY *glGetProgramPipelineInfoLogEXT) (GLuint  pipeline, GLsizei  bufSize, GLsizei * length, GLchar * infoLog);
extern void (GLAPIENTRY *glGetProgramPipelineiv) (GLuint  pipeline, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetProgramPipelineivEXT) (GLuint  pipeline, GLenum  pname, GLint * params);
extern GLuint (GLAPIENTRY *glGetProgramResourceIndex) (GLuint  program, GLenum  programInterface, const GLchar * name);
extern GLint (GLAPIENTRY *glGetProgramResourceLocation) (GLuint  program, GLenum  programInterface, const GLchar * name);
extern GLint (GLAPIENTRY *glGetProgramResourceLocationIndex) (GLuint  program, GLenum  programInterface, const GLchar * name);
extern void (GLAPIENTRY *glGetProgramResourceName) (GLuint  program, GLenum  programInterface, GLuint  index, GLsizei  bufSize, GLsizei * length, GLchar * name);
extern void (GLAPIENTRY *glGetProgramResourcefvNV) (GLuint  program, GLenum  programInterface, GLuint  index, GLsizei  propCount, const GLenum * props, GLsizei  count, GLsizei * length, GLfloat * params);
extern void (GLAPIENTRY *glGetProgramResourceiv) (GLuint  program, GLenum  programInterface, GLuint  index, GLsizei  propCount, const GLenum * props, GLsizei  count, GLsizei * length, GLint * params);
extern void (GLAPIENTRY *glGetProgramStageiv) (GLuint  program, GLenum  shadertype, GLenum  pname, GLint * values);
extern void (GLAPIENTRY *glGetProgramStringARB) (GLenum  target, GLenum  pname, void * string);
extern void (GLAPIENTRY *glGetProgramStringNV) (GLuint  id, GLenum  pname, GLubyte * program);
extern void (GLAPIENTRY *glGetProgramSubroutineParameteruivNV) (GLenum  target, GLuint  index, GLuint * param);
extern void (GLAPIENTRY *glGetProgramiv) (GLuint  program, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetProgramivARB) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetProgramivNV) (GLuint  id, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetQueryBufferObjecti64v) (GLuint  id, GLuint  buffer, GLenum  pname, GLintptr  offset);
extern void (GLAPIENTRY *glGetQueryBufferObjectiv) (GLuint  id, GLuint  buffer, GLenum  pname, GLintptr  offset);
extern void (GLAPIENTRY *glGetQueryBufferObjectui64v) (GLuint  id, GLuint  buffer, GLenum  pname, GLintptr  offset);
extern void (GLAPIENTRY *glGetQueryBufferObjectuiv) (GLuint  id, GLuint  buffer, GLenum  pname, GLintptr  offset);
extern void (GLAPIENTRY *glGetQueryIndexediv) (GLenum  target, GLuint  index, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetQueryObjecti64v) (GLuint  id, GLenum  pname, GLint64 * params);
extern void (GLAPIENTRY *glGetQueryObjecti64vEXT) (GLuint  id, GLenum  pname, GLint64 * params);
extern void (GLAPIENTRY *glGetQueryObjectiv) (GLuint  id, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetQueryObjectivARB) (GLuint  id, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetQueryObjectui64v) (GLuint  id, GLenum  pname, GLuint64 * params);
extern void (GLAPIENTRY *glGetQueryObjectui64vEXT) (GLuint  id, GLenum  pname, GLuint64 * params);
extern void (GLAPIENTRY *glGetQueryObjectuiv) (GLuint  id, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetQueryObjectuivARB) (GLuint  id, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetQueryiv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetQueryivARB) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetRenderbufferParameteriv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetRenderbufferParameterivEXT) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetSamplerParameterIiv) (GLuint  sampler, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetSamplerParameterIuiv) (GLuint  sampler, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetSamplerParameterfv) (GLuint  sampler, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetSamplerParameteriv) (GLuint  sampler, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetSemaphoreParameterui64vEXT) (GLuint  semaphore, GLenum  pname, GLuint64 * params);
extern void (GLAPIENTRY *glGetSeparableFilter) (GLenum  target, GLenum  format, GLenum  type, void * row, void * column, void * span);
extern void (GLAPIENTRY *glGetSeparableFilterEXT) (GLenum  target, GLenum  format, GLenum  type, void * row, void * column, void * span);
extern void (GLAPIENTRY *glGetShaderInfoLog) (GLuint  shader, GLsizei  bufSize, GLsizei * length, GLchar * infoLog);
extern void (GLAPIENTRY *glGetShaderPrecisionFormat) (GLenum  shadertype, GLenum  precisiontype, GLint * range, GLint * precision);
extern void (GLAPIENTRY *glGetShaderSource) (GLuint  shader, GLsizei  bufSize, GLsizei * length, GLchar * source);
extern void (GLAPIENTRY *glGetShaderSourceARB) (GLhandleARB  obj, GLsizei  maxLength, GLsizei * length, GLcharARB * source);
extern void (GLAPIENTRY *glGetShaderiv) (GLuint  shader, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetShadingRateImagePaletteNV) (GLuint  viewport, GLuint  entry, GLenum * rate);
extern void (GLAPIENTRY *glGetShadingRateSampleLocationivNV) (GLenum  rate, GLuint  samples, GLuint  index, GLint * location);
extern GLushort (GLAPIENTRY *glGetStageIndexNV) (GLenum  shadertype);
extern const GLubyte *(GLAPIENTRY *glGetString) (GLenum  name);
extern const GLubyte *(GLAPIENTRY *glGetStringi) (GLenum  name, GLuint  index);
extern GLuint (GLAPIENTRY *glGetSubroutineIndex) (GLuint  program, GLenum  shadertype, const GLchar * name);
extern GLint (GLAPIENTRY *glGetSubroutineUniformLocation) (GLuint  program, GLenum  shadertype, const GLchar * name);
extern void (GLAPIENTRY *glGetSynciv) (GLsync  sync, GLenum  pname, GLsizei  count, GLsizei * length, GLint * values);
extern void (GLAPIENTRY *glGetTexEnvfv) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetTexEnviv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTexGendv) (GLenum  coord, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetTexGenfv) (GLenum  coord, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetTexGeniv) (GLenum  coord, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTexImage) (GLenum  target, GLint  level, GLenum  format, GLenum  type, void * pixels);
extern void (GLAPIENTRY *glGetTexLevelParameterfv) (GLenum  target, GLint  level, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetTexLevelParameteriv) (GLenum  target, GLint  level, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTexParameterIiv) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTexParameterIivEXT) (GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTexParameterIuiv) (GLenum  target, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetTexParameterIuivEXT) (GLenum  target, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetTexParameterPointervAPPLE) (GLenum  target, GLenum  pname, void ** params);
extern void (GLAPIENTRY *glGetTexParameterfv) (GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetTexParameteriv) (GLenum  target, GLenum  pname, GLint * params);
extern GLuint64 (GLAPIENTRY *glGetTextureHandleARB) (GLuint  texture);
extern GLuint64 (GLAPIENTRY *glGetTextureHandleNV) (GLuint  texture);
extern void (GLAPIENTRY *glGetTextureImage) (GLuint  texture, GLint  level, GLenum  format, GLenum  type, GLsizei  bufSize, void * pixels);
extern void (GLAPIENTRY *glGetTextureImageEXT) (GLuint  texture, GLenum  target, GLint  level, GLenum  format, GLenum  type, void * pixels);
extern void (GLAPIENTRY *glGetTextureLevelParameterfv) (GLuint  texture, GLint  level, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetTextureLevelParameterfvEXT) (GLuint  texture, GLenum  target, GLint  level, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetTextureLevelParameteriv) (GLuint  texture, GLint  level, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTextureLevelParameterivEXT) (GLuint  texture, GLenum  target, GLint  level, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTextureParameterIiv) (GLuint  texture, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTextureParameterIivEXT) (GLuint  texture, GLenum  target, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTextureParameterIuiv) (GLuint  texture, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetTextureParameterIuivEXT) (GLuint  texture, GLenum  target, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetTextureParameterfv) (GLuint  texture, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetTextureParameterfvEXT) (GLuint  texture, GLenum  target, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetTextureParameteriv) (GLuint  texture, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTextureParameterivEXT) (GLuint  texture, GLenum  target, GLenum  pname, GLint * params);
extern GLuint64 (GLAPIENTRY *glGetTextureSamplerHandleARB) (GLuint  texture, GLuint  sampler);
extern GLuint64 (GLAPIENTRY *glGetTextureSamplerHandleNV) (GLuint  texture, GLuint  sampler);
extern void (GLAPIENTRY *glGetTextureSubImage) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, GLsizei  bufSize, void * pixels);
extern void (GLAPIENTRY *glGetTrackMatrixivNV) (GLenum  target, GLuint  address, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetTransformFeedbackVarying) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
extern void (GLAPIENTRY *glGetTransformFeedbackVaryingEXT) (GLuint  program, GLuint  index, GLsizei  bufSize, GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
extern void (GLAPIENTRY *glGetTransformFeedbackVaryingNV) (GLuint  program, GLuint  index, GLint * location);
extern void (GLAPIENTRY *glGetTransformFeedbacki64_v) (GLuint  xfb, GLenum  pname, GLuint  index, GLint64 * param);
extern void (GLAPIENTRY *glGetTransformFeedbacki_v) (GLuint  xfb, GLenum  pname, GLuint  index, GLint * param);
extern void (GLAPIENTRY *glGetTransformFeedbackiv) (GLuint  xfb, GLenum  pname, GLint * param);
extern GLuint (GLAPIENTRY *glGetUniformBlockIndex) (GLuint  program, const GLchar * uniformBlockName);
extern GLint (GLAPIENTRY *glGetUniformBufferSizeEXT) (GLuint  program, GLint  location);
extern void (GLAPIENTRY *glGetUniformIndices) (GLuint  program, GLsizei  uniformCount, const GLchar *const* uniformNames, GLuint * uniformIndices);
extern GLint (GLAPIENTRY *glGetUniformLocation) (GLuint  program, const GLchar * name);
extern GLint (GLAPIENTRY *glGetUniformLocationARB) (GLhandleARB  programObj, const GLcharARB * name);
extern GLintptr (GLAPIENTRY *glGetUniformOffsetEXT) (GLuint  program, GLint  location);
extern void (GLAPIENTRY *glGetUniformSubroutineuiv) (GLenum  shadertype, GLint  location, GLuint * params);
extern void (GLAPIENTRY *glGetUniformdv) (GLuint  program, GLint  location, GLdouble * params);
extern void (GLAPIENTRY *glGetUniformfv) (GLuint  program, GLint  location, GLfloat * params);
extern void (GLAPIENTRY *glGetUniformfvARB) (GLhandleARB  programObj, GLint  location, GLfloat * params);
extern void (GLAPIENTRY *glGetUniformi64vARB) (GLuint  program, GLint  location, GLint64 * params);
extern void (GLAPIENTRY *glGetUniformi64vNV) (GLuint  program, GLint  location, GLint64EXT * params);
extern void (GLAPIENTRY *glGetUniformiv) (GLuint  program, GLint  location, GLint * params);
extern void (GLAPIENTRY *glGetUniformivARB) (GLhandleARB  programObj, GLint  location, GLint * params);
extern void (GLAPIENTRY *glGetUniformui64vARB) (GLuint  program, GLint  location, GLuint64 * params);
extern void (GLAPIENTRY *glGetUniformui64vNV) (GLuint  program, GLint  location, GLuint64EXT * params);
extern void (GLAPIENTRY *glGetUniformuiv) (GLuint  program, GLint  location, GLuint * params);
extern void (GLAPIENTRY *glGetUniformuivEXT) (GLuint  program, GLint  location, GLuint * params);
extern void (GLAPIENTRY *glGetUnsignedBytevEXT) (GLenum  pname, GLubyte * data);
extern void (GLAPIENTRY *glGetUnsignedBytei_vEXT) (GLenum  target, GLuint  index, GLubyte * data);
extern void (GLAPIENTRY *glGetVariantBooleanvEXT) (GLuint  id, GLenum  value, GLboolean * data);
extern void (GLAPIENTRY *glGetVariantFloatvEXT) (GLuint  id, GLenum  value, GLfloat * data);
extern void (GLAPIENTRY *glGetVariantIntegervEXT) (GLuint  id, GLenum  value, GLint * data);
extern void (GLAPIENTRY *glGetVariantPointervEXT) (GLuint  id, GLenum  value, void ** data);
extern GLint (GLAPIENTRY *glGetVaryingLocationNV) (GLuint  program, const GLchar * name);
extern void (GLAPIENTRY *glGetVertexArrayIndexed64iv) (GLuint  vaobj, GLuint  index, GLenum  pname, GLint64 * param);
extern void (GLAPIENTRY *glGetVertexArrayIndexediv) (GLuint  vaobj, GLuint  index, GLenum  pname, GLint * param);
extern void (GLAPIENTRY *glGetVertexArrayIntegeri_vEXT) (GLuint  vaobj, GLuint  index, GLenum  pname, GLint * param);
extern void (GLAPIENTRY *glGetVertexArrayIntegervEXT) (GLuint  vaobj, GLenum  pname, GLint * param);
extern void (GLAPIENTRY *glGetVertexArrayPointeri_vEXT) (GLuint  vaobj, GLuint  index, GLenum  pname, void ** param);
extern void (GLAPIENTRY *glGetVertexArrayPointervEXT) (GLuint  vaobj, GLenum  pname, void ** param);
extern void (GLAPIENTRY *glGetVertexArrayiv) (GLuint  vaobj, GLenum  pname, GLint * param);
extern void (GLAPIENTRY *glGetVertexAttribIiv) (GLuint  index, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetVertexAttribIivEXT) (GLuint  index, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetVertexAttribIuiv) (GLuint  index, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetVertexAttribIuivEXT) (GLuint  index, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetVertexAttribLdv) (GLuint  index, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetVertexAttribLdvEXT) (GLuint  index, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetVertexAttribLi64vNV) (GLuint  index, GLenum  pname, GLint64EXT * params);
extern void (GLAPIENTRY *glGetVertexAttribLui64vARB) (GLuint  index, GLenum  pname, GLuint64EXT * params);
extern void (GLAPIENTRY *glGetVertexAttribLui64vNV) (GLuint  index, GLenum  pname, GLuint64EXT * params);
extern void (GLAPIENTRY *glGetVertexAttribPointerv) (GLuint  index, GLenum  pname, void ** pointer);
extern void (GLAPIENTRY *glGetVertexAttribPointervARB) (GLuint  index, GLenum  pname, void ** pointer);
extern void (GLAPIENTRY *glGetVertexAttribPointervNV) (GLuint  index, GLenum  pname, void ** pointer);
extern void (GLAPIENTRY *glGetVertexAttribdv) (GLuint  index, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetVertexAttribdvARB) (GLuint  index, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetVertexAttribdvNV) (GLuint  index, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetVertexAttribfv) (GLuint  index, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetVertexAttribfvARB) (GLuint  index, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetVertexAttribfvNV) (GLuint  index, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetVertexAttribiv) (GLuint  index, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetVertexAttribivARB) (GLuint  index, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetVertexAttribivNV) (GLuint  index, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetVideoCaptureStreamdvNV) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, GLdouble * params);
extern void (GLAPIENTRY *glGetVideoCaptureStreamfvNV) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, GLfloat * params);
extern void (GLAPIENTRY *glGetVideoCaptureStreamivNV) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetVideoCaptureivNV) (GLuint  video_capture_slot, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetVideoi64vNV) (GLuint  video_slot, GLenum  pname, GLint64EXT * params);
extern void (GLAPIENTRY *glGetVideoivNV) (GLuint  video_slot, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glGetVideoui64vNV) (GLuint  video_slot, GLenum  pname, GLuint64EXT * params);
extern void (GLAPIENTRY *glGetVideouivNV) (GLuint  video_slot, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glGetnColorTable) (GLenum  target, GLenum  format, GLenum  type, GLsizei  bufSize, void * table);
extern void (GLAPIENTRY *glGetnColorTableARB) (GLenum  target, GLenum  format, GLenum  type, GLsizei  bufSize, void * table);
extern void (GLAPIENTRY *glGetnCompressedTexImage) (GLenum  target, GLint  lod, GLsizei  bufSize, void * pixels);
extern void (GLAPIENTRY *glGetnCompressedTexImageARB) (GLenum  target, GLint  lod, GLsizei  bufSize, void * img);
extern void (GLAPIENTRY *glGetnConvolutionFilter) (GLenum  target, GLenum  format, GLenum  type, GLsizei  bufSize, void * image);
extern void (GLAPIENTRY *glGetnConvolutionFilterARB) (GLenum  target, GLenum  format, GLenum  type, GLsizei  bufSize, void * image);
extern void (GLAPIENTRY *glGetnHistogram) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, GLsizei  bufSize, void * values);
extern void (GLAPIENTRY *glGetnHistogramARB) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, GLsizei  bufSize, void * values);
extern void (GLAPIENTRY *glGetnMapdv) (GLenum  target, GLenum  query, GLsizei  bufSize, GLdouble * v);
extern void (GLAPIENTRY *glGetnMapdvARB) (GLenum  target, GLenum  query, GLsizei  bufSize, GLdouble * v);
extern void (GLAPIENTRY *glGetnMapfv) (GLenum  target, GLenum  query, GLsizei  bufSize, GLfloat * v);
extern void (GLAPIENTRY *glGetnMapfvARB) (GLenum  target, GLenum  query, GLsizei  bufSize, GLfloat * v);
extern void (GLAPIENTRY *glGetnMapiv) (GLenum  target, GLenum  query, GLsizei  bufSize, GLint * v);
extern void (GLAPIENTRY *glGetnMapivARB) (GLenum  target, GLenum  query, GLsizei  bufSize, GLint * v);
extern void (GLAPIENTRY *glGetnMinmax) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, GLsizei  bufSize, void * values);
extern void (GLAPIENTRY *glGetnMinmaxARB) (GLenum  target, GLboolean  reset, GLenum  format, GLenum  type, GLsizei  bufSize, void * values);
extern void (GLAPIENTRY *glGetnPixelMapfv) (GLenum  map, GLsizei  bufSize, GLfloat * values);
extern void (GLAPIENTRY *glGetnPixelMapfvARB) (GLenum  map, GLsizei  bufSize, GLfloat * values);
extern void (GLAPIENTRY *glGetnPixelMapuiv) (GLenum  map, GLsizei  bufSize, GLuint * values);
extern void (GLAPIENTRY *glGetnPixelMapuivARB) (GLenum  map, GLsizei  bufSize, GLuint * values);
extern void (GLAPIENTRY *glGetnPixelMapusv) (GLenum  map, GLsizei  bufSize, GLushort * values);
extern void (GLAPIENTRY *glGetnPixelMapusvARB) (GLenum  map, GLsizei  bufSize, GLushort * values);
extern void (GLAPIENTRY *glGetnPolygonStipple) (GLsizei  bufSize, GLubyte * pattern);
extern void (GLAPIENTRY *glGetnPolygonStippleARB) (GLsizei  bufSize, GLubyte * pattern);
extern void (GLAPIENTRY *glGetnSeparableFilter) (GLenum  target, GLenum  format, GLenum  type, GLsizei  rowBufSize, void * row, GLsizei  columnBufSize, void * column, void * span);
extern void (GLAPIENTRY *glGetnSeparableFilterARB) (GLenum  target, GLenum  format, GLenum  type, GLsizei  rowBufSize, void * row, GLsizei  columnBufSize, void * column, void * span);
extern void (GLAPIENTRY *glGetnTexImage) (GLenum  target, GLint  level, GLenum  format, GLenum  type, GLsizei  bufSize, void * pixels);
extern void (GLAPIENTRY *glGetnTexImageARB) (GLenum  target, GLint  level, GLenum  format, GLenum  type, GLsizei  bufSize, void * img);
extern void (GLAPIENTRY *glGetnUniformdv) (GLuint  program, GLint  location, GLsizei  bufSize, GLdouble * params);
extern void (GLAPIENTRY *glGetnUniformdvARB) (GLuint  program, GLint  location, GLsizei  bufSize, GLdouble * params);
extern void (GLAPIENTRY *glGetnUniformfv) (GLuint  program, GLint  location, GLsizei  bufSize, GLfloat * params);
extern void (GLAPIENTRY *glGetnUniformfvARB) (GLuint  program, GLint  location, GLsizei  bufSize, GLfloat * params);
extern void (GLAPIENTRY *glGetnUniformfvKHR) (GLuint  program, GLint  location, GLsizei  bufSize, GLfloat * params);
extern void (GLAPIENTRY *glGetnUniformi64vARB) (GLuint  program, GLint  location, GLsizei  bufSize, GLint64 * params);
extern void (GLAPIENTRY *glGetnUniformiv) (GLuint  program, GLint  location, GLsizei  bufSize, GLint * params);
extern void (GLAPIENTRY *glGetnUniformivARB) (GLuint  program, GLint  location, GLsizei  bufSize, GLint * params);
extern void (GLAPIENTRY *glGetnUniformivKHR) (GLuint  program, GLint  location, GLsizei  bufSize, GLint * params);
extern void (GLAPIENTRY *glGetnUniformui64vARB) (GLuint  program, GLint  location, GLsizei  bufSize, GLuint64 * params);
extern void (GLAPIENTRY *glGetnUniformuiv) (GLuint  program, GLint  location, GLsizei  bufSize, GLuint * params);
extern void (GLAPIENTRY *glGetnUniformuivARB) (GLuint  program, GLint  location, GLsizei  bufSize, GLuint * params);
extern void (GLAPIENTRY *glGetnUniformuivKHR) (GLuint  program, GLint  location, GLsizei  bufSize, GLuint * params);
extern void (GLAPIENTRY *glHint) (GLenum  target, GLenum  mode);
extern void (GLAPIENTRY *glHistogram) (GLenum  target, GLsizei  width, GLenum  internalformat, GLboolean  sink);
extern void (GLAPIENTRY *glHistogramEXT) (GLenum  target, GLsizei  width, GLenum  internalformat, GLboolean  sink);
extern void (GLAPIENTRY *glImportMemoryFdEXT) (GLuint  memory, GLuint64  size, GLenum  handleType, GLint  fd);
extern void (GLAPIENTRY *glImportMemoryWin32HandleEXT) (GLuint  memory, GLuint64  size, GLenum  handleType, void * handle);
extern void (GLAPIENTRY *glImportMemoryWin32NameEXT) (GLuint  memory, GLuint64  size, GLenum  handleType, const void * name);
extern void (GLAPIENTRY *glImportSemaphoreFdEXT) (GLuint  semaphore, GLenum  handleType, GLint  fd);
extern void (GLAPIENTRY *glImportSemaphoreWin32HandleEXT) (GLuint  semaphore, GLenum  handleType, void * handle);
extern void (GLAPIENTRY *glImportSemaphoreWin32NameEXT) (GLuint  semaphore, GLenum  handleType, const void * name);
extern GLsync (GLAPIENTRY *glImportSyncEXT) (GLenum  external_sync_type, GLintptr  external_sync, GLbitfield  flags);
extern void (GLAPIENTRY *glIndexFormatNV) (GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glIndexFuncEXT) (GLenum  func, GLclampf  ref);
extern void (GLAPIENTRY *glIndexMask) (GLuint  mask);
extern void (GLAPIENTRY *glIndexMaterialEXT) (GLenum  face, GLenum  mode);
extern void (GLAPIENTRY *glIndexPointer) (GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glIndexPointerEXT) (GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
extern void (GLAPIENTRY *glIndexd) (GLdouble  c);
extern void (GLAPIENTRY *glIndexdv) (const GLdouble * c);
extern void (GLAPIENTRY *glIndexf) (GLfloat  c);
extern void (GLAPIENTRY *glIndexfv) (const GLfloat * c);
extern void (GLAPIENTRY *glIndexi) (GLint  c);
extern void (GLAPIENTRY *glIndexiv) (const GLint * c);
extern void (GLAPIENTRY *glIndexs) (GLshort  c);
extern void (GLAPIENTRY *glIndexsv) (const GLshort * c);
extern void (GLAPIENTRY *glIndexub) (GLubyte  c);
extern void (GLAPIENTRY *glIndexubv) (const GLubyte * c);
extern void (GLAPIENTRY *glInitNames) ();
extern void (GLAPIENTRY *glInsertComponentEXT) (GLuint  res, GLuint  src, GLuint  num);
extern void (GLAPIENTRY *glInsertEventMarkerEXT) (GLsizei  length, const GLchar * marker);
extern void (GLAPIENTRY *glInterleavedArrays) (GLenum  format, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glInterpolatePathsNV) (GLuint  resultPath, GLuint  pathA, GLuint  pathB, GLfloat  weight);
extern void (GLAPIENTRY *glInvalidateBufferData) (GLuint  buffer);
extern void (GLAPIENTRY *glInvalidateBufferSubData) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length);
extern void (GLAPIENTRY *glInvalidateFramebuffer) (GLenum  target, GLsizei  numAttachments, const GLenum * attachments);
extern void (GLAPIENTRY *glInvalidateNamedFramebufferData) (GLuint  framebuffer, GLsizei  numAttachments, const GLenum * attachments);
extern void (GLAPIENTRY *glInvalidateNamedFramebufferSubData) (GLuint  framebuffer, GLsizei  numAttachments, const GLenum * attachments, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glInvalidateSubFramebuffer) (GLenum  target, GLsizei  numAttachments, const GLenum * attachments, GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glInvalidateTexImage) (GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glInvalidateTexSubImage) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth);
extern GLboolean (GLAPIENTRY *glIsBuffer) (GLuint  buffer);
extern GLboolean (GLAPIENTRY *glIsBufferARB) (GLuint  buffer);
extern GLboolean (GLAPIENTRY *glIsBufferResidentNV) (GLenum  target);
extern GLboolean (GLAPIENTRY *glIsCommandListNV) (GLuint  list);
extern GLboolean (GLAPIENTRY *glIsEnabled) (GLenum  cap);
extern GLboolean (GLAPIENTRY *glIsEnabledIndexedEXT) (GLenum  target, GLuint  index);
extern GLboolean (GLAPIENTRY *glIsEnabledi) (GLenum  target, GLuint  index);
extern GLboolean (GLAPIENTRY *glIsFenceAPPLE) (GLuint  fence);
extern GLboolean (GLAPIENTRY *glIsFenceNV) (GLuint  fence);
extern GLboolean (GLAPIENTRY *glIsFramebuffer) (GLuint  framebuffer);
extern GLboolean (GLAPIENTRY *glIsFramebufferEXT) (GLuint  framebuffer);
extern GLboolean (GLAPIENTRY *glIsImageHandleResidentARB) (GLuint64  handle);
extern GLboolean (GLAPIENTRY *glIsImageHandleResidentNV) (GLuint64  handle);
extern GLboolean (GLAPIENTRY *glIsList) (GLuint  list);
extern GLboolean (GLAPIENTRY *glIsMemoryObjectEXT) (GLuint  memoryObject);
extern GLboolean (GLAPIENTRY *glIsNameAMD) (GLenum  identifier, GLuint  name);
extern GLboolean (GLAPIENTRY *glIsNamedBufferResidentNV) (GLuint  buffer);
extern GLboolean (GLAPIENTRY *glIsNamedStringARB) (GLint  namelen, const GLchar * name);
extern GLboolean (GLAPIENTRY *glIsOcclusionQueryNV) (GLuint  id);
extern GLboolean (GLAPIENTRY *glIsPathNV) (GLuint  path);
extern GLboolean (GLAPIENTRY *glIsPointInFillPathNV) (GLuint  path, GLuint  mask, GLfloat  x, GLfloat  y);
extern GLboolean (GLAPIENTRY *glIsPointInStrokePathNV) (GLuint  path, GLfloat  x, GLfloat  y);
extern GLboolean (GLAPIENTRY *glIsProgram) (GLuint  program);
extern GLboolean (GLAPIENTRY *glIsProgramARB) (GLuint  program);
extern GLboolean (GLAPIENTRY *glIsProgramNV) (GLuint  id);
extern GLboolean (GLAPIENTRY *glIsProgramPipeline) (GLuint  pipeline);
extern GLboolean (GLAPIENTRY *glIsProgramPipelineEXT) (GLuint  pipeline);
extern GLboolean (GLAPIENTRY *glIsQuery) (GLuint  id);
extern GLboolean (GLAPIENTRY *glIsQueryARB) (GLuint  id);
extern GLboolean (GLAPIENTRY *glIsRenderbuffer) (GLuint  renderbuffer);
extern GLboolean (GLAPIENTRY *glIsRenderbufferEXT) (GLuint  renderbuffer);
extern GLboolean (GLAPIENTRY *glIsSemaphoreEXT) (GLuint  semaphore);
extern GLboolean (GLAPIENTRY *glIsSampler) (GLuint  sampler);
extern GLboolean (GLAPIENTRY *glIsShader) (GLuint  shader);
extern GLboolean (GLAPIENTRY *glIsStateNV) (GLuint  state);
extern GLboolean (GLAPIENTRY *glIsSync) (GLsync  sync);
extern GLboolean (GLAPIENTRY *glIsTexture) (GLuint  texture);
extern GLboolean (GLAPIENTRY *glIsTextureEXT) (GLuint  texture);
extern GLboolean (GLAPIENTRY *glIsTextureHandleResidentARB) (GLuint64  handle);
extern GLboolean (GLAPIENTRY *glIsTextureHandleResidentNV) (GLuint64  handle);
extern GLboolean (GLAPIENTRY *glIsTransformFeedback) (GLuint  id);
extern GLboolean (GLAPIENTRY *glIsTransformFeedbackNV) (GLuint  id);
extern GLboolean (GLAPIENTRY *glIsVariantEnabledEXT) (GLuint  id, GLenum  cap);
extern GLboolean (GLAPIENTRY *glIsVertexArray) (GLuint  array);
extern GLboolean (GLAPIENTRY *glIsVertexArrayAPPLE) (GLuint  array);
extern GLboolean (GLAPIENTRY *glIsVertexAttribEnabledAPPLE) (GLuint  index, GLenum  pname);
extern void (GLAPIENTRY *glLabelObjectEXT) (GLenum  type, GLuint  object, GLsizei  length, const GLchar * label);
extern void (GLAPIENTRY *glLightModelf) (GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glLightModelfv) (GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glLightModeli) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glLightModeliv) (GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glLightf) (GLenum  light, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glLightfv) (GLenum  light, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glLighti) (GLenum  light, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glLightiv) (GLenum  light, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glLineStipple) (GLint  factor, GLushort  pattern);
extern void (GLAPIENTRY *glLineWidth) (GLfloat  width);
extern void (GLAPIENTRY *glLinkProgram) (GLuint  program);
extern void (GLAPIENTRY *glLinkProgramARB) (GLhandleARB  programObj);
extern void (GLAPIENTRY *glListBase) (GLuint  base);
extern void (GLAPIENTRY *glListDrawCommandsStatesClientNV) (GLuint  list, GLuint  segment, const void ** indirects, const GLsizei * sizes, const GLuint * states, const GLuint * fbos, GLuint  count);
extern void (GLAPIENTRY *glLoadIdentity) ();
extern void (GLAPIENTRY *glLoadMatrixd) (const GLdouble * m);
extern void (GLAPIENTRY *glLoadMatrixf) (const GLfloat * m);
extern void (GLAPIENTRY *glLoadName) (GLuint  name);
extern void (GLAPIENTRY *glLoadProgramNV) (GLenum  target, GLuint  id, GLsizei  len, const GLubyte * program);
extern void (GLAPIENTRY *glLoadTransposeMatrixd) (const GLdouble * m);
extern void (GLAPIENTRY *glLoadTransposeMatrixdARB) (const GLdouble * m);
extern void (GLAPIENTRY *glLoadTransposeMatrixf) (const GLfloat * m);
extern void (GLAPIENTRY *glLoadTransposeMatrixfARB) (const GLfloat * m);
extern void (GLAPIENTRY *glLockArraysEXT) (GLint  first, GLsizei  count);
extern void (GLAPIENTRY *glLogicOp) (GLenum  opcode);
extern void (GLAPIENTRY *glMakeBufferNonResidentNV) (GLenum  target);
extern void (GLAPIENTRY *glMakeBufferResidentNV) (GLenum  target, GLenum  access);
extern void (GLAPIENTRY *glMakeImageHandleNonResidentARB) (GLuint64  handle);
extern void (GLAPIENTRY *glMakeImageHandleNonResidentNV) (GLuint64  handle);
extern void (GLAPIENTRY *glMakeImageHandleResidentARB) (GLuint64  handle, GLenum  access);
extern void (GLAPIENTRY *glMakeImageHandleResidentNV) (GLuint64  handle, GLenum  access);
extern void (GLAPIENTRY *glMakeNamedBufferNonResidentNV) (GLuint  buffer);
extern void (GLAPIENTRY *glMakeNamedBufferResidentNV) (GLuint  buffer, GLenum  access);
extern void (GLAPIENTRY *glMakeTextureHandleNonResidentARB) (GLuint64  handle);
extern void (GLAPIENTRY *glMakeTextureHandleNonResidentNV) (GLuint64  handle);
extern void (GLAPIENTRY *glMakeTextureHandleResidentARB) (GLuint64  handle);
extern void (GLAPIENTRY *glMakeTextureHandleResidentNV) (GLuint64  handle);
extern void (GLAPIENTRY *glMap1d) (GLenum  target, GLdouble  u1, GLdouble  u2, GLint  stride, GLint  order, const GLdouble * points);
extern void (GLAPIENTRY *glMap1f) (GLenum  target, GLfloat  u1, GLfloat  u2, GLint  stride, GLint  order, const GLfloat * points);
extern void (GLAPIENTRY *glMap2d) (GLenum  target, GLdouble  u1, GLdouble  u2, GLint  ustride, GLint  uorder, GLdouble  v1, GLdouble  v2, GLint  vstride, GLint  vorder, const GLdouble * points);
extern void (GLAPIENTRY *glMap2f) (GLenum  target, GLfloat  u1, GLfloat  u2, GLint  ustride, GLint  uorder, GLfloat  v1, GLfloat  v2, GLint  vstride, GLint  vorder, const GLfloat * points);
extern void *(GLAPIENTRY *glMapBuffer) (GLenum  target, GLenum  access);
extern void *(GLAPIENTRY *glMapBufferARB) (GLenum  target, GLenum  access);
extern void *(GLAPIENTRY *glMapBufferRange) (GLenum  target, GLintptr  offset, GLsizeiptr  length, GLbitfield  access);
extern void (GLAPIENTRY *glMapControlPointsNV) (GLenum  target, GLuint  index, GLenum  type, GLsizei  ustride, GLsizei  vstride, GLint  uorder, GLint  vorder, GLboolean  packed, const void * points);
extern void (GLAPIENTRY *glMapGrid1d) (GLint  un, GLdouble  u1, GLdouble  u2);
extern void (GLAPIENTRY *glMapGrid1f) (GLint  un, GLfloat  u1, GLfloat  u2);
extern void (GLAPIENTRY *glMapGrid2d) (GLint  un, GLdouble  u1, GLdouble  u2, GLint  vn, GLdouble  v1, GLdouble  v2);
extern void (GLAPIENTRY *glMapGrid2f) (GLint  un, GLfloat  u1, GLfloat  u2, GLint  vn, GLfloat  v1, GLfloat  v2);
extern void *(GLAPIENTRY *glMapNamedBuffer) (GLuint  buffer, GLenum  access);
extern void *(GLAPIENTRY *glMapNamedBufferEXT) (GLuint  buffer, GLenum  access);
extern void *(GLAPIENTRY *glMapNamedBufferRange) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length, GLbitfield  access);
extern void *(GLAPIENTRY *glMapNamedBufferRangeEXT) (GLuint  buffer, GLintptr  offset, GLsizeiptr  length, GLbitfield  access);
extern void (GLAPIENTRY *glMapParameterfvNV) (GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glMapParameterivNV) (GLenum  target, GLenum  pname, const GLint * params);
extern void *(GLAPIENTRY *glMapTexture2DINTEL) (GLuint  texture, GLint  level, GLbitfield  access, GLint * stride, GLenum * layout);
extern void (GLAPIENTRY *glMapVertexAttrib1dAPPLE) (GLuint  index, GLuint  size, GLdouble  u1, GLdouble  u2, GLint  stride, GLint  order, const GLdouble * points);
extern void (GLAPIENTRY *glMapVertexAttrib1fAPPLE) (GLuint  index, GLuint  size, GLfloat  u1, GLfloat  u2, GLint  stride, GLint  order, const GLfloat * points);
extern void (GLAPIENTRY *glMapVertexAttrib2dAPPLE) (GLuint  index, GLuint  size, GLdouble  u1, GLdouble  u2, GLint  ustride, GLint  uorder, GLdouble  v1, GLdouble  v2, GLint  vstride, GLint  vorder, const GLdouble * points);
extern void (GLAPIENTRY *glMapVertexAttrib2fAPPLE) (GLuint  index, GLuint  size, GLfloat  u1, GLfloat  u2, GLint  ustride, GLint  uorder, GLfloat  v1, GLfloat  v2, GLint  vstride, GLint  vorder, const GLfloat * points);
extern void (GLAPIENTRY *glMaterialf) (GLenum  face, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glMaterialfv) (GLenum  face, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glMateriali) (GLenum  face, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glMaterialiv) (GLenum  face, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glMatrixFrustumEXT) (GLenum  mode, GLdouble  left, GLdouble  right, GLdouble  bottom, GLdouble  top, GLdouble  zNear, GLdouble  zFar);
extern void (GLAPIENTRY *glMatrixIndexPointerARB) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glMatrixIndexubvARB) (GLint  size, const GLubyte * indices);
extern void (GLAPIENTRY *glMatrixIndexuivARB) (GLint  size, const GLuint * indices);
extern void (GLAPIENTRY *glMatrixIndexusvARB) (GLint  size, const GLushort * indices);
extern void (GLAPIENTRY *glMatrixLoad3x2fNV) (GLenum  matrixMode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixLoad3x3fNV) (GLenum  matrixMode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixLoadIdentityEXT) (GLenum  mode);
extern void (GLAPIENTRY *glMatrixLoadTranspose3x3fNV) (GLenum  matrixMode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixLoadTransposedEXT) (GLenum  mode, const GLdouble * m);
extern void (GLAPIENTRY *glMatrixLoadTransposefEXT) (GLenum  mode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixLoaddEXT) (GLenum  mode, const GLdouble * m);
extern void (GLAPIENTRY *glMatrixLoadfEXT) (GLenum  mode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixMode) (GLenum  mode);
extern void (GLAPIENTRY *glMatrixMult3x2fNV) (GLenum  matrixMode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixMult3x3fNV) (GLenum  matrixMode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixMultTranspose3x3fNV) (GLenum  matrixMode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixMultTransposedEXT) (GLenum  mode, const GLdouble * m);
extern void (GLAPIENTRY *glMatrixMultTransposefEXT) (GLenum  mode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixMultdEXT) (GLenum  mode, const GLdouble * m);
extern void (GLAPIENTRY *glMatrixMultfEXT) (GLenum  mode, const GLfloat * m);
extern void (GLAPIENTRY *glMatrixOrthoEXT) (GLenum  mode, GLdouble  left, GLdouble  right, GLdouble  bottom, GLdouble  top, GLdouble  zNear, GLdouble  zFar);
extern void (GLAPIENTRY *glMatrixPopEXT) (GLenum  mode);
extern void (GLAPIENTRY *glMatrixPushEXT) (GLenum  mode);
extern void (GLAPIENTRY *glMatrixRotatedEXT) (GLenum  mode, GLdouble  angle, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glMatrixRotatefEXT) (GLenum  mode, GLfloat  angle, GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glMatrixScaledEXT) (GLenum  mode, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glMatrixScalefEXT) (GLenum  mode, GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glMatrixTranslatedEXT) (GLenum  mode, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glMatrixTranslatefEXT) (GLenum  mode, GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glMaxShaderCompilerThreadsKHR) (GLuint  count);
extern void (GLAPIENTRY *glMaxShaderCompilerThreadsARB) (GLuint  count);
extern void (GLAPIENTRY *glMemoryBarrier) (GLbitfield  barriers);
extern void (GLAPIENTRY *glMemoryBarrierByRegion) (GLbitfield  barriers);
extern void (GLAPIENTRY *glMemoryBarrierEXT) (GLbitfield  barriers);
extern void (GLAPIENTRY *glMemoryObjectParameterivEXT) (GLuint  memoryObject, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glMinSampleShading) (GLfloat  value);
extern void (GLAPIENTRY *glMinSampleShadingARB) (GLfloat  value);
extern void (GLAPIENTRY *glMinmax) (GLenum  target, GLenum  internalformat, GLboolean  sink);
extern void (GLAPIENTRY *glMinmaxEXT) (GLenum  target, GLenum  internalformat, GLboolean  sink);
extern void (GLAPIENTRY *glMultMatrixd) (const GLdouble * m);
extern void (GLAPIENTRY *glMultMatrixf) (const GLfloat * m);
extern void (GLAPIENTRY *glMultTransposeMatrixd) (const GLdouble * m);
extern void (GLAPIENTRY *glMultTransposeMatrixdARB) (const GLdouble * m);
extern void (GLAPIENTRY *glMultTransposeMatrixf) (const GLfloat * m);
extern void (GLAPIENTRY *glMultTransposeMatrixfARB) (const GLfloat * m);
extern void (GLAPIENTRY *glMultiDrawArrays) (GLenum  mode, const GLint * first, const GLsizei * count, GLsizei  drawcount);
extern void (GLAPIENTRY *glMultiDrawArraysEXT) (GLenum  mode, const GLint * first, const GLsizei * count, GLsizei  primcount);
extern void (GLAPIENTRY *glMultiDrawArraysIndirect) (GLenum  mode, const void * indirect, GLsizei  drawcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawArraysIndirectAMD) (GLenum  mode, const void * indirect, GLsizei  primcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawArraysIndirectBindlessCountNV) (GLenum  mode, const void * indirect, GLsizei  drawCount, GLsizei  maxDrawCount, GLsizei  stride, GLint  vertexBufferCount);
extern void (GLAPIENTRY *glMultiDrawArraysIndirectBindlessNV) (GLenum  mode, const void * indirect, GLsizei  drawCount, GLsizei  stride, GLint  vertexBufferCount);
extern void (GLAPIENTRY *glMultiDrawArraysIndirectCount) (GLenum  mode, const void * indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawArraysIndirectCountARB) (GLenum  mode, const void * indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawElementArrayAPPLE) (GLenum  mode, const GLint * first, const GLsizei * count, GLsizei  primcount);
extern void (GLAPIENTRY *glMultiDrawElements) (GLenum  mode, const GLsizei * count, GLenum  type, const void *const* indices, GLsizei  drawcount);
extern void (GLAPIENTRY *glMultiDrawElementsBaseVertex) (GLenum  mode, const GLsizei * count, GLenum  type, const void *const* indices, GLsizei  drawcount, const GLint * basevertex);
extern void (GLAPIENTRY *glMultiDrawElementsEXT) (GLenum  mode, const GLsizei * count, GLenum  type, const void *const* indices, GLsizei  primcount);
extern void (GLAPIENTRY *glMultiDrawElementsIndirect) (GLenum  mode, GLenum  type, const void * indirect, GLsizei  drawcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawElementsIndirectAMD) (GLenum  mode, GLenum  type, const void * indirect, GLsizei  primcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawElementsIndirectBindlessCountNV) (GLenum  mode, GLenum  type, const void * indirect, GLsizei  drawCount, GLsizei  maxDrawCount, GLsizei  stride, GLint  vertexBufferCount);
extern void (GLAPIENTRY *glMultiDrawElementsIndirectBindlessNV) (GLenum  mode, GLenum  type, const void * indirect, GLsizei  drawCount, GLsizei  stride, GLint  vertexBufferCount);
extern void (GLAPIENTRY *glMultiDrawElementsIndirectCount) (GLenum  mode, GLenum  type, const void * indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawElementsIndirectCountARB) (GLenum  mode, GLenum  type, const void * indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawMeshTasksIndirectNV) (GLintptr  indirect, GLsizei  drawcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawMeshTasksIndirectCountNV) (GLintptr  indirect, GLintptr  drawcount, GLsizei  maxdrawcount, GLsizei  stride);
extern void (GLAPIENTRY *glMultiDrawRangeElementArrayAPPLE) (GLenum  mode, GLuint  start, GLuint  end, const GLint * first, const GLsizei * count, GLsizei  primcount);
extern void (GLAPIENTRY *glMultiTexBufferEXT) (GLenum  texunit, GLenum  target, GLenum  internalformat, GLuint  buffer);
extern void (GLAPIENTRY *glMultiTexCoord1d) (GLenum  target, GLdouble  s);
extern void (GLAPIENTRY *glMultiTexCoord1dARB) (GLenum  target, GLdouble  s);
extern void (GLAPIENTRY *glMultiTexCoord1dv) (GLenum  target, const GLdouble * v);
extern void (GLAPIENTRY *glMultiTexCoord1dvARB) (GLenum  target, const GLdouble * v);
extern void (GLAPIENTRY *glMultiTexCoord1f) (GLenum  target, GLfloat  s);
extern void (GLAPIENTRY *glMultiTexCoord1fARB) (GLenum  target, GLfloat  s);
extern void (GLAPIENTRY *glMultiTexCoord1fv) (GLenum  target, const GLfloat * v);
extern void (GLAPIENTRY *glMultiTexCoord1fvARB) (GLenum  target, const GLfloat * v);
extern void (GLAPIENTRY *glMultiTexCoord1hNV) (GLenum  target, GLhalfNV  s);
extern void (GLAPIENTRY *glMultiTexCoord1hvNV) (GLenum  target, const GLhalfNV * v);
extern void (GLAPIENTRY *glMultiTexCoord1i) (GLenum  target, GLint  s);
extern void (GLAPIENTRY *glMultiTexCoord1iARB) (GLenum  target, GLint  s);
extern void (GLAPIENTRY *glMultiTexCoord1iv) (GLenum  target, const GLint * v);
extern void (GLAPIENTRY *glMultiTexCoord1ivARB) (GLenum  target, const GLint * v);
extern void (GLAPIENTRY *glMultiTexCoord1s) (GLenum  target, GLshort  s);
extern void (GLAPIENTRY *glMultiTexCoord1sARB) (GLenum  target, GLshort  s);
extern void (GLAPIENTRY *glMultiTexCoord1sv) (GLenum  target, const GLshort * v);
extern void (GLAPIENTRY *glMultiTexCoord1svARB) (GLenum  target, const GLshort * v);
extern void (GLAPIENTRY *glMultiTexCoord2d) (GLenum  target, GLdouble  s, GLdouble  t);
extern void (GLAPIENTRY *glMultiTexCoord2dARB) (GLenum  target, GLdouble  s, GLdouble  t);
extern void (GLAPIENTRY *glMultiTexCoord2dv) (GLenum  target, const GLdouble * v);
extern void (GLAPIENTRY *glMultiTexCoord2dvARB) (GLenum  target, const GLdouble * v);
extern void (GLAPIENTRY *glMultiTexCoord2f) (GLenum  target, GLfloat  s, GLfloat  t);
extern void (GLAPIENTRY *glMultiTexCoord2fARB) (GLenum  target, GLfloat  s, GLfloat  t);
extern void (GLAPIENTRY *glMultiTexCoord2fv) (GLenum  target, const GLfloat * v);
extern void (GLAPIENTRY *glMultiTexCoord2fvARB) (GLenum  target, const GLfloat * v);
extern void (GLAPIENTRY *glMultiTexCoord2hNV) (GLenum  target, GLhalfNV  s, GLhalfNV  t);
extern void (GLAPIENTRY *glMultiTexCoord2hvNV) (GLenum  target, const GLhalfNV * v);
extern void (GLAPIENTRY *glMultiTexCoord2i) (GLenum  target, GLint  s, GLint  t);
extern void (GLAPIENTRY *glMultiTexCoord2iARB) (GLenum  target, GLint  s, GLint  t);
extern void (GLAPIENTRY *glMultiTexCoord2iv) (GLenum  target, const GLint * v);
extern void (GLAPIENTRY *glMultiTexCoord2ivARB) (GLenum  target, const GLint * v);
extern void (GLAPIENTRY *glMultiTexCoord2s) (GLenum  target, GLshort  s, GLshort  t);
extern void (GLAPIENTRY *glMultiTexCoord2sARB) (GLenum  target, GLshort  s, GLshort  t);
extern void (GLAPIENTRY *glMultiTexCoord2sv) (GLenum  target, const GLshort * v);
extern void (GLAPIENTRY *glMultiTexCoord2svARB) (GLenum  target, const GLshort * v);
extern void (GLAPIENTRY *glMultiTexCoord3d) (GLenum  target, GLdouble  s, GLdouble  t, GLdouble  r);
extern void (GLAPIENTRY *glMultiTexCoord3dARB) (GLenum  target, GLdouble  s, GLdouble  t, GLdouble  r);
extern void (GLAPIENTRY *glMultiTexCoord3dv) (GLenum  target, const GLdouble * v);
extern void (GLAPIENTRY *glMultiTexCoord3dvARB) (GLenum  target, const GLdouble * v);
extern void (GLAPIENTRY *glMultiTexCoord3f) (GLenum  target, GLfloat  s, GLfloat  t, GLfloat  r);
extern void (GLAPIENTRY *glMultiTexCoord3fARB) (GLenum  target, GLfloat  s, GLfloat  t, GLfloat  r);
extern void (GLAPIENTRY *glMultiTexCoord3fv) (GLenum  target, const GLfloat * v);
extern void (GLAPIENTRY *glMultiTexCoord3fvARB) (GLenum  target, const GLfloat * v);
extern void (GLAPIENTRY *glMultiTexCoord3hNV) (GLenum  target, GLhalfNV  s, GLhalfNV  t, GLhalfNV  r);
extern void (GLAPIENTRY *glMultiTexCoord3hvNV) (GLenum  target, const GLhalfNV * v);
extern void (GLAPIENTRY *glMultiTexCoord3i) (GLenum  target, GLint  s, GLint  t, GLint  r);
extern void (GLAPIENTRY *glMultiTexCoord3iARB) (GLenum  target, GLint  s, GLint  t, GLint  r);
extern void (GLAPIENTRY *glMultiTexCoord3iv) (GLenum  target, const GLint * v);
extern void (GLAPIENTRY *glMultiTexCoord3ivARB) (GLenum  target, const GLint * v);
extern void (GLAPIENTRY *glMultiTexCoord3s) (GLenum  target, GLshort  s, GLshort  t, GLshort  r);
extern void (GLAPIENTRY *glMultiTexCoord3sARB) (GLenum  target, GLshort  s, GLshort  t, GLshort  r);
extern void (GLAPIENTRY *glMultiTexCoord3sv) (GLenum  target, const GLshort * v);
extern void (GLAPIENTRY *glMultiTexCoord3svARB) (GLenum  target, const GLshort * v);
extern void (GLAPIENTRY *glMultiTexCoord4d) (GLenum  target, GLdouble  s, GLdouble  t, GLdouble  r, GLdouble  q);
extern void (GLAPIENTRY *glMultiTexCoord4dARB) (GLenum  target, GLdouble  s, GLdouble  t, GLdouble  r, GLdouble  q);
extern void (GLAPIENTRY *glMultiTexCoord4dv) (GLenum  target, const GLdouble * v);
extern void (GLAPIENTRY *glMultiTexCoord4dvARB) (GLenum  target, const GLdouble * v);
extern void (GLAPIENTRY *glMultiTexCoord4f) (GLenum  target, GLfloat  s, GLfloat  t, GLfloat  r, GLfloat  q);
extern void (GLAPIENTRY *glMultiTexCoord4fARB) (GLenum  target, GLfloat  s, GLfloat  t, GLfloat  r, GLfloat  q);
extern void (GLAPIENTRY *glMultiTexCoord4fv) (GLenum  target, const GLfloat * v);
extern void (GLAPIENTRY *glMultiTexCoord4fvARB) (GLenum  target, const GLfloat * v);
extern void (GLAPIENTRY *glMultiTexCoord4hNV) (GLenum  target, GLhalfNV  s, GLhalfNV  t, GLhalfNV  r, GLhalfNV  q);
extern void (GLAPIENTRY *glMultiTexCoord4hvNV) (GLenum  target, const GLhalfNV * v);
extern void (GLAPIENTRY *glMultiTexCoord4i) (GLenum  target, GLint  s, GLint  t, GLint  r, GLint  q);
extern void (GLAPIENTRY *glMultiTexCoord4iARB) (GLenum  target, GLint  s, GLint  t, GLint  r, GLint  q);
extern void (GLAPIENTRY *glMultiTexCoord4iv) (GLenum  target, const GLint * v);
extern void (GLAPIENTRY *glMultiTexCoord4ivARB) (GLenum  target, const GLint * v);
extern void (GLAPIENTRY *glMultiTexCoord4s) (GLenum  target, GLshort  s, GLshort  t, GLshort  r, GLshort  q);
extern void (GLAPIENTRY *glMultiTexCoord4sARB) (GLenum  target, GLshort  s, GLshort  t, GLshort  r, GLshort  q);
extern void (GLAPIENTRY *glMultiTexCoord4sv) (GLenum  target, const GLshort * v);
extern void (GLAPIENTRY *glMultiTexCoord4svARB) (GLenum  target, const GLshort * v);
extern void (GLAPIENTRY *glMultiTexCoordP1ui) (GLenum  texture, GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glMultiTexCoordP1uiv) (GLenum  texture, GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glMultiTexCoordP2ui) (GLenum  texture, GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glMultiTexCoordP2uiv) (GLenum  texture, GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glMultiTexCoordP3ui) (GLenum  texture, GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glMultiTexCoordP3uiv) (GLenum  texture, GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glMultiTexCoordP4ui) (GLenum  texture, GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glMultiTexCoordP4uiv) (GLenum  texture, GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glMultiTexCoordPointerEXT) (GLenum  texunit, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glMultiTexEnvfEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glMultiTexEnvfvEXT) (GLenum  texunit, GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glMultiTexEnviEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glMultiTexEnvivEXT) (GLenum  texunit, GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glMultiTexGendEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, GLdouble  param);
extern void (GLAPIENTRY *glMultiTexGendvEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, const GLdouble * params);
extern void (GLAPIENTRY *glMultiTexGenfEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glMultiTexGenfvEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glMultiTexGeniEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glMultiTexGenivEXT) (GLenum  texunit, GLenum  coord, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glMultiTexImage1DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glMultiTexImage2DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glMultiTexImage3DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glMultiTexParameterIivEXT) (GLenum  texunit, GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glMultiTexParameterIuivEXT) (GLenum  texunit, GLenum  target, GLenum  pname, const GLuint * params);
extern void (GLAPIENTRY *glMultiTexParameterfEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glMultiTexParameterfvEXT) (GLenum  texunit, GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glMultiTexParameteriEXT) (GLenum  texunit, GLenum  target, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glMultiTexParameterivEXT) (GLenum  texunit, GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glMultiTexRenderbufferEXT) (GLenum  texunit, GLenum  target, GLuint  renderbuffer);
extern void (GLAPIENTRY *glMultiTexSubImage1DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glMultiTexSubImage2DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glMultiTexSubImage3DEXT) (GLenum  texunit, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glMulticastBarrierNV) ();
extern void (GLAPIENTRY *glMulticastBlitFramebufferNV) (GLuint  srcGpu, GLuint  dstGpu, GLint  srcX0, GLint  srcY0, GLint  srcX1, GLint  srcY1, GLint  dstX0, GLint  dstY0, GLint  dstX1, GLint  dstY1, GLbitfield  mask, GLenum  filter);
extern void (GLAPIENTRY *glMulticastBufferSubDataNV) (GLbitfield  gpuMask, GLuint  buffer, GLintptr  offset, GLsizeiptr  size, const void * data);
extern void (GLAPIENTRY *glMulticastCopyBufferSubDataNV) (GLuint  readGpu, GLbitfield  writeGpuMask, GLuint  readBuffer, GLuint  writeBuffer, GLintptr  readOffset, GLintptr  writeOffset, GLsizeiptr  size);
extern void (GLAPIENTRY *glMulticastCopyImageSubDataNV) (GLuint  srcGpu, GLbitfield  dstGpuMask, GLuint  srcName, GLenum  srcTarget, GLint  srcLevel, GLint  srcX, GLint  srcY, GLint  srcZ, GLuint  dstName, GLenum  dstTarget, GLint  dstLevel, GLint  dstX, GLint  dstY, GLint  dstZ, GLsizei  srcWidth, GLsizei  srcHeight, GLsizei  srcDepth);
extern void (GLAPIENTRY *glMulticastFramebufferSampleLocationsfvNV) (GLuint  gpu, GLuint  framebuffer, GLuint  start, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glMulticastGetQueryObjecti64vNV) (GLuint  gpu, GLuint  id, GLenum  pname, GLint64 * params);
extern void (GLAPIENTRY *glMulticastGetQueryObjectivNV) (GLuint  gpu, GLuint  id, GLenum  pname, GLint * params);
extern void (GLAPIENTRY *glMulticastGetQueryObjectui64vNV) (GLuint  gpu, GLuint  id, GLenum  pname, GLuint64 * params);
extern void (GLAPIENTRY *glMulticastGetQueryObjectuivNV) (GLuint  gpu, GLuint  id, GLenum  pname, GLuint * params);
extern void (GLAPIENTRY *glMulticastWaitSyncNV) (GLuint  signalGpu, GLbitfield  waitGpuMask);
extern void (GLAPIENTRY *glNamedBufferAttachMemoryNV) (GLuint  buffer, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glNamedBufferData) (GLuint  buffer, GLsizeiptr  size, const void * data, GLenum  usage);
extern void (GLAPIENTRY *glNamedBufferDataEXT) (GLuint  buffer, GLsizeiptr  size, const void * data, GLenum  usage);
extern void (GLAPIENTRY *glNamedBufferPageCommitmentARB) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, GLboolean  commit);
extern void (GLAPIENTRY *glNamedBufferPageCommitmentEXT) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, GLboolean  commit);
extern void (GLAPIENTRY *glNamedBufferStorage) (GLuint  buffer, GLsizeiptr  size, const void * data, GLbitfield  flags);
extern void (GLAPIENTRY *glNamedBufferStorageExternalEXT) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, GLeglClientBufferEXT  clientBuffer, GLbitfield  flags);
extern void (GLAPIENTRY *glNamedBufferStorageEXT) (GLuint  buffer, GLsizeiptr  size, const void * data, GLbitfield  flags);
extern void (GLAPIENTRY *glNamedBufferStorageMemEXT) (GLuint  buffer, GLsizeiptr  size, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glNamedBufferSubData) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, const void * data);
extern void (GLAPIENTRY *glNamedBufferSubDataEXT) (GLuint  buffer, GLintptr  offset, GLsizeiptr  size, const void * data);
extern void (GLAPIENTRY *glNamedCopyBufferSubDataEXT) (GLuint  readBuffer, GLuint  writeBuffer, GLintptr  readOffset, GLintptr  writeOffset, GLsizeiptr  size);
extern void (GLAPIENTRY *glNamedFramebufferDrawBuffer) (GLuint  framebuffer, GLenum  buf);
extern void (GLAPIENTRY *glNamedFramebufferDrawBuffers) (GLuint  framebuffer, GLsizei  n, const GLenum * bufs);
extern void (GLAPIENTRY *glNamedFramebufferParameteri) (GLuint  framebuffer, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glNamedFramebufferParameteriEXT) (GLuint  framebuffer, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glNamedFramebufferReadBuffer) (GLuint  framebuffer, GLenum  src);
extern void (GLAPIENTRY *glNamedFramebufferRenderbuffer) (GLuint  framebuffer, GLenum  attachment, GLenum  renderbuffertarget, GLuint  renderbuffer);
extern void (GLAPIENTRY *glNamedFramebufferRenderbufferEXT) (GLuint  framebuffer, GLenum  attachment, GLenum  renderbuffertarget, GLuint  renderbuffer);
extern void (GLAPIENTRY *glNamedFramebufferSampleLocationsfvARB) (GLuint  framebuffer, GLuint  start, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glNamedFramebufferSampleLocationsfvNV) (GLuint  framebuffer, GLuint  start, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glNamedFramebufferTexture) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glNamedFramebufferSamplePositionsfvAMD) (GLuint  framebuffer, GLuint  numsamples, GLuint  pixelindex, const GLfloat * values);
extern void (GLAPIENTRY *glNamedFramebufferTexture1DEXT) (GLuint  framebuffer, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glNamedFramebufferTexture2DEXT) (GLuint  framebuffer, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glNamedFramebufferTexture3DEXT) (GLuint  framebuffer, GLenum  attachment, GLenum  textarget, GLuint  texture, GLint  level, GLint  zoffset);
extern void (GLAPIENTRY *glNamedFramebufferTextureEXT) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glNamedFramebufferTextureFaceEXT) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level, GLenum  face);
extern void (GLAPIENTRY *glNamedFramebufferTextureLayer) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
extern void (GLAPIENTRY *glNamedFramebufferTextureLayerEXT) (GLuint  framebuffer, GLenum  attachment, GLuint  texture, GLint  level, GLint  layer);
extern void (GLAPIENTRY *glNamedProgramLocalParameter4dEXT) (GLuint  program, GLenum  target, GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glNamedProgramLocalParameter4dvEXT) (GLuint  program, GLenum  target, GLuint  index, const GLdouble * params);
extern void (GLAPIENTRY *glNamedProgramLocalParameter4fEXT) (GLuint  program, GLenum  target, GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glNamedProgramLocalParameter4fvEXT) (GLuint  program, GLenum  target, GLuint  index, const GLfloat * params);
extern void (GLAPIENTRY *glNamedProgramLocalParameterI4iEXT) (GLuint  program, GLenum  target, GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
extern void (GLAPIENTRY *glNamedProgramLocalParameterI4ivEXT) (GLuint  program, GLenum  target, GLuint  index, const GLint * params);
extern void (GLAPIENTRY *glNamedProgramLocalParameterI4uiEXT) (GLuint  program, GLenum  target, GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
extern void (GLAPIENTRY *glNamedProgramLocalParameterI4uivEXT) (GLuint  program, GLenum  target, GLuint  index, const GLuint * params);
extern void (GLAPIENTRY *glNamedProgramLocalParameters4fvEXT) (GLuint  program, GLenum  target, GLuint  index, GLsizei  count, const GLfloat * params);
extern void (GLAPIENTRY *glNamedProgramLocalParametersI4ivEXT) (GLuint  program, GLenum  target, GLuint  index, GLsizei  count, const GLint * params);
extern void (GLAPIENTRY *glNamedProgramLocalParametersI4uivEXT) (GLuint  program, GLenum  target, GLuint  index, GLsizei  count, const GLuint * params);
extern void (GLAPIENTRY *glNamedProgramStringEXT) (GLuint  program, GLenum  target, GLenum  format, GLsizei  len, const void * string);
extern void (GLAPIENTRY *glNamedRenderbufferStorage) (GLuint  renderbuffer, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glNamedRenderbufferStorageEXT) (GLuint  renderbuffer, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glNamedRenderbufferStorageMultisample) (GLuint  renderbuffer, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glNamedRenderbufferStorageMultisampleAdvancedAMD) (GLuint  renderbuffer, GLsizei  samples, GLsizei  storageSamples, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glNamedRenderbufferStorageMultisampleCoverageEXT) (GLuint  renderbuffer, GLsizei  coverageSamples, GLsizei  colorSamples, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glNamedRenderbufferStorageMultisampleEXT) (GLuint  renderbuffer, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glNamedStringARB) (GLenum  type, GLint  namelen, const GLchar * name, GLint  stringlen, const GLchar * string);
extern void (GLAPIENTRY *glNewList) (GLuint  list, GLenum  mode);
extern void (GLAPIENTRY *glNormal3b) (GLbyte  nx, GLbyte  ny, GLbyte  nz);
extern void (GLAPIENTRY *glNormal3bv) (const GLbyte * v);
extern void (GLAPIENTRY *glNormal3d) (GLdouble  nx, GLdouble  ny, GLdouble  nz);
extern void (GLAPIENTRY *glNormal3dv) (const GLdouble * v);
extern void (GLAPIENTRY *glNormal3f) (GLfloat  nx, GLfloat  ny, GLfloat  nz);
extern void (GLAPIENTRY *glNormal3fv) (const GLfloat * v);
extern void (GLAPIENTRY *glNormal3hNV) (GLhalfNV  nx, GLhalfNV  ny, GLhalfNV  nz);
extern void (GLAPIENTRY *glNormal3hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glNormal3i) (GLint  nx, GLint  ny, GLint  nz);
extern void (GLAPIENTRY *glNormal3iv) (const GLint * v);
extern void (GLAPIENTRY *glNormal3s) (GLshort  nx, GLshort  ny, GLshort  nz);
extern void (GLAPIENTRY *glNormal3sv) (const GLshort * v);
extern void (GLAPIENTRY *glNormalFormatNV) (GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glNormalP3ui) (GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glNormalP3uiv) (GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glNormalPointer) (GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glNormalPointerEXT) (GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
extern void (GLAPIENTRY *glNormalPointervINTEL) (GLenum  type, const void ** pointer);
extern void (GLAPIENTRY *glObjectLabel) (GLenum  identifier, GLuint  name, GLsizei  length, const GLchar * label);
extern void (GLAPIENTRY *glObjectLabelKHR) (GLenum  identifier, GLuint  name, GLsizei  length, const GLchar * label);
extern void (GLAPIENTRY *glObjectPtrLabel) (const void * ptr, GLsizei  length, const GLchar * label);
extern void (GLAPIENTRY *glObjectPtrLabelKHR) (const void * ptr, GLsizei  length, const GLchar * label);
extern GLenum (GLAPIENTRY *glObjectPurgeableAPPLE) (GLenum  objectType, GLuint  name, GLenum  option);
extern GLenum (GLAPIENTRY *glObjectUnpurgeableAPPLE) (GLenum  objectType, GLuint  name, GLenum  option);
extern void (GLAPIENTRY *glOrtho) (GLdouble  left, GLdouble  right, GLdouble  bottom, GLdouble  top, GLdouble  zNear, GLdouble  zFar);
extern void (GLAPIENTRY *glPassThrough) (GLfloat  token);
extern void (GLAPIENTRY *glPatchParameterfv) (GLenum  pname, const GLfloat * values);
extern void (GLAPIENTRY *glPatchParameteri) (GLenum  pname, GLint  value);
extern void (GLAPIENTRY *glPathColorGenNV) (GLenum  color, GLenum  genMode, GLenum  colorFormat, const GLfloat * coeffs);
extern void (GLAPIENTRY *glPathCommandsNV) (GLuint  path, GLsizei  numCommands, const GLubyte * commands, GLsizei  numCoords, GLenum  coordType, const void * coords);
extern void (GLAPIENTRY *glPathCoordsNV) (GLuint  path, GLsizei  numCoords, GLenum  coordType, const void * coords);
extern void (GLAPIENTRY *glPathCoverDepthFuncNV) (GLenum  func);
extern void (GLAPIENTRY *glPathDashArrayNV) (GLuint  path, GLsizei  dashCount, const GLfloat * dashArray);
extern void (GLAPIENTRY *glPathFogGenNV) (GLenum  genMode);
extern GLenum (GLAPIENTRY *glPathGlyphIndexArrayNV) (GLuint  firstPathName, GLenum  fontTarget, const void * fontName, GLbitfield  fontStyle, GLuint  firstGlyphIndex, GLsizei  numGlyphs, GLuint  pathParameterTemplate, GLfloat  emScale);
extern GLenum (GLAPIENTRY *glPathGlyphIndexRangeNV) (GLenum  fontTarget, const void * fontName, GLbitfield  fontStyle, GLuint  pathParameterTemplate, GLfloat  emScale, GLuint  baseAndCount);
extern void (GLAPIENTRY *glPathGlyphRangeNV) (GLuint  firstPathName, GLenum  fontTarget, const void * fontName, GLbitfield  fontStyle, GLuint  firstGlyph, GLsizei  numGlyphs, GLenum  handleMissingGlyphs, GLuint  pathParameterTemplate, GLfloat  emScale);
extern void (GLAPIENTRY *glPathGlyphsNV) (GLuint  firstPathName, GLenum  fontTarget, const void * fontName, GLbitfield  fontStyle, GLsizei  numGlyphs, GLenum  type, const void * charcodes, GLenum  handleMissingGlyphs, GLuint  pathParameterTemplate, GLfloat  emScale);
extern GLenum (GLAPIENTRY *glPathMemoryGlyphIndexArrayNV) (GLuint  firstPathName, GLenum  fontTarget, GLsizeiptr  fontSize, const void * fontData, GLsizei  faceIndex, GLuint  firstGlyphIndex, GLsizei  numGlyphs, GLuint  pathParameterTemplate, GLfloat  emScale);
extern void (GLAPIENTRY *glPathParameterfNV) (GLuint  path, GLenum  pname, GLfloat  value);
extern void (GLAPIENTRY *glPathParameterfvNV) (GLuint  path, GLenum  pname, const GLfloat * value);
extern void (GLAPIENTRY *glPathParameteriNV) (GLuint  path, GLenum  pname, GLint  value);
extern void (GLAPIENTRY *glPathParameterivNV) (GLuint  path, GLenum  pname, const GLint * value);
extern void (GLAPIENTRY *glPathStencilDepthOffsetNV) (GLfloat  factor, GLfloat  units);
extern void (GLAPIENTRY *glPathStencilFuncNV) (GLenum  func, GLint  ref, GLuint  mask);
extern void (GLAPIENTRY *glPathStringNV) (GLuint  path, GLenum  format, GLsizei  length, const void * pathString);
extern void (GLAPIENTRY *glPathSubCommandsNV) (GLuint  path, GLsizei  commandStart, GLsizei  commandsToDelete, GLsizei  numCommands, const GLubyte * commands, GLsizei  numCoords, GLenum  coordType, const void * coords);
extern void (GLAPIENTRY *glPathSubCoordsNV) (GLuint  path, GLsizei  coordStart, GLsizei  numCoords, GLenum  coordType, const void * coords);
extern void (GLAPIENTRY *glPathTexGenNV) (GLenum  texCoordSet, GLenum  genMode, GLint  components, const GLfloat * coeffs);
extern void (GLAPIENTRY *glPauseTransformFeedback) ();
extern void (GLAPIENTRY *glPauseTransformFeedbackNV) ();
extern void (GLAPIENTRY *glPixelDataRangeNV) (GLenum  target, GLsizei  length, const void * pointer);
extern void (GLAPIENTRY *glPixelMapfv) (GLenum  map, GLsizei  mapsize, const GLfloat * values);
extern void (GLAPIENTRY *glPixelMapuiv) (GLenum  map, GLsizei  mapsize, const GLuint * values);
extern void (GLAPIENTRY *glPixelMapusv) (GLenum  map, GLsizei  mapsize, const GLushort * values);
extern void (GLAPIENTRY *glPixelStoref) (GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glPixelStorei) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glPixelTransferf) (GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glPixelTransferi) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glPixelTransformParameterfEXT) (GLenum  target, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glPixelTransformParameterfvEXT) (GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glPixelTransformParameteriEXT) (GLenum  target, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glPixelTransformParameterivEXT) (GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glPixelZoom) (GLfloat  xfactor, GLfloat  yfactor);
extern GLboolean (GLAPIENTRY *glPointAlongPathNV) (GLuint  path, GLsizei  startSegment, GLsizei  numSegments, GLfloat  distance, GLfloat * x, GLfloat * y, GLfloat * tangentX, GLfloat * tangentY);
extern void (GLAPIENTRY *glPointParameterf) (GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glPointParameterfARB) (GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glPointParameterfEXT) (GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glPointParameterfv) (GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glPointParameterfvARB) (GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glPointParameterfvEXT) (GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glPointParameteri) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glPointParameteriNV) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glPointParameteriv) (GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glPointParameterivNV) (GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glPointSize) (GLfloat  size);
extern void (GLAPIENTRY *glPolygonMode) (GLenum  face, GLenum  mode);
extern void (GLAPIENTRY *glPolygonOffset) (GLfloat  factor, GLfloat  units);
extern void (GLAPIENTRY *glPolygonOffsetClamp) (GLfloat  factor, GLfloat  units, GLfloat  clamp);
extern void (GLAPIENTRY *glPolygonOffsetClampEXT) (GLfloat  factor, GLfloat  units, GLfloat  clamp);
extern void (GLAPIENTRY *glPolygonOffsetEXT) (GLfloat  factor, GLfloat  bias);
extern void (GLAPIENTRY *glPolygonStipple) (const GLubyte * mask);
extern void (GLAPIENTRY *glPopAttrib) ();
extern void (GLAPIENTRY *glPopClientAttrib) ();
extern void (GLAPIENTRY *glPopDebugGroup) ();
extern void (GLAPIENTRY *glPopDebugGroupKHR) ();
extern void (GLAPIENTRY *glPopGroupMarkerEXT) ();
extern void (GLAPIENTRY *glPopMatrix) ();
extern void (GLAPIENTRY *glPopName) ();
extern void (GLAPIENTRY *glPresentFrameDualFillNV) (GLuint  video_slot, GLuint64EXT  minPresentTime, GLuint  beginPresentTimeId, GLuint  presentDurationId, GLenum  type, GLenum  target0, GLuint  fill0, GLenum  target1, GLuint  fill1, GLenum  target2, GLuint  fill2, GLenum  target3, GLuint  fill3);
extern void (GLAPIENTRY *glPresentFrameKeyedNV) (GLuint  video_slot, GLuint64EXT  minPresentTime, GLuint  beginPresentTimeId, GLuint  presentDurationId, GLenum  type, GLenum  target0, GLuint  fill0, GLuint  key0, GLenum  target1, GLuint  fill1, GLuint  key1);
extern void (GLAPIENTRY *glPrimitiveBoundingBoxARB) (GLfloat  minX, GLfloat  minY, GLfloat  minZ, GLfloat  minW, GLfloat  maxX, GLfloat  maxY, GLfloat  maxZ, GLfloat  maxW);
extern void (GLAPIENTRY *glPrimitiveRestartIndex) (GLuint  index);
extern void (GLAPIENTRY *glPrimitiveRestartIndexNV) (GLuint  index);
extern void (GLAPIENTRY *glPrimitiveRestartNV) ();
extern void (GLAPIENTRY *glPrioritizeTextures) (GLsizei  n, const GLuint * textures, const GLfloat * priorities);
extern void (GLAPIENTRY *glPrioritizeTexturesEXT) (GLsizei  n, const GLuint * textures, const GLclampf * priorities);
extern void (GLAPIENTRY *glProgramBinary) (GLuint  program, GLenum  binaryFormat, const void * binary, GLsizei  length);
extern void (GLAPIENTRY *glProgramBufferParametersIivNV) (GLenum  target, GLuint  bindingIndex, GLuint  wordIndex, GLsizei  count, const GLint * params);
extern void (GLAPIENTRY *glProgramBufferParametersIuivNV) (GLenum  target, GLuint  bindingIndex, GLuint  wordIndex, GLsizei  count, const GLuint * params);
extern void (GLAPIENTRY *glProgramBufferParametersfvNV) (GLenum  target, GLuint  bindingIndex, GLuint  wordIndex, GLsizei  count, const GLfloat * params);
extern void (GLAPIENTRY *glProgramEnvParameter4dARB) (GLenum  target, GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glProgramEnvParameter4dvARB) (GLenum  target, GLuint  index, const GLdouble * params);
extern void (GLAPIENTRY *glProgramEnvParameter4fARB) (GLenum  target, GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glProgramEnvParameter4fvARB) (GLenum  target, GLuint  index, const GLfloat * params);
extern void (GLAPIENTRY *glProgramEnvParameterI4iNV) (GLenum  target, GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
extern void (GLAPIENTRY *glProgramEnvParameterI4ivNV) (GLenum  target, GLuint  index, const GLint * params);
extern void (GLAPIENTRY *glProgramEnvParameterI4uiNV) (GLenum  target, GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
extern void (GLAPIENTRY *glProgramEnvParameterI4uivNV) (GLenum  target, GLuint  index, const GLuint * params);
extern void (GLAPIENTRY *glProgramEnvParameters4fvEXT) (GLenum  target, GLuint  index, GLsizei  count, const GLfloat * params);
extern void (GLAPIENTRY *glProgramEnvParametersI4ivNV) (GLenum  target, GLuint  index, GLsizei  count, const GLint * params);
extern void (GLAPIENTRY *glProgramEnvParametersI4uivNV) (GLenum  target, GLuint  index, GLsizei  count, const GLuint * params);
extern void (GLAPIENTRY *glProgramLocalParameter4dARB) (GLenum  target, GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glProgramLocalParameter4dvARB) (GLenum  target, GLuint  index, const GLdouble * params);
extern void (GLAPIENTRY *glProgramLocalParameter4fARB) (GLenum  target, GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glProgramLocalParameter4fvARB) (GLenum  target, GLuint  index, const GLfloat * params);
extern void (GLAPIENTRY *glProgramLocalParameterI4iNV) (GLenum  target, GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
extern void (GLAPIENTRY *glProgramLocalParameterI4ivNV) (GLenum  target, GLuint  index, const GLint * params);
extern void (GLAPIENTRY *glProgramLocalParameterI4uiNV) (GLenum  target, GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
extern void (GLAPIENTRY *glProgramLocalParameterI4uivNV) (GLenum  target, GLuint  index, const GLuint * params);
extern void (GLAPIENTRY *glProgramLocalParameters4fvEXT) (GLenum  target, GLuint  index, GLsizei  count, const GLfloat * params);
extern void (GLAPIENTRY *glProgramLocalParametersI4ivNV) (GLenum  target, GLuint  index, GLsizei  count, const GLint * params);
extern void (GLAPIENTRY *glProgramLocalParametersI4uivNV) (GLenum  target, GLuint  index, GLsizei  count, const GLuint * params);
extern void (GLAPIENTRY *glProgramNamedParameter4dNV) (GLuint  id, GLsizei  len, const GLubyte * name, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glProgramNamedParameter4dvNV) (GLuint  id, GLsizei  len, const GLubyte * name, const GLdouble * v);
extern void (GLAPIENTRY *glProgramNamedParameter4fNV) (GLuint  id, GLsizei  len, const GLubyte * name, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glProgramNamedParameter4fvNV) (GLuint  id, GLsizei  len, const GLubyte * name, const GLfloat * v);
extern void (GLAPIENTRY *glProgramParameter4dNV) (GLenum  target, GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glProgramParameter4dvNV) (GLenum  target, GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glProgramParameter4fNV) (GLenum  target, GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glProgramParameter4fvNV) (GLenum  target, GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glProgramParameteri) (GLuint  program, GLenum  pname, GLint  value);
extern void (GLAPIENTRY *glProgramParameteriARB) (GLuint  program, GLenum  pname, GLint  value);
extern void (GLAPIENTRY *glProgramParameteriEXT) (GLuint  program, GLenum  pname, GLint  value);
extern void (GLAPIENTRY *glProgramParameters4dvNV) (GLenum  target, GLuint  index, GLsizei  count, const GLdouble * v);
extern void (GLAPIENTRY *glProgramParameters4fvNV) (GLenum  target, GLuint  index, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glProgramPathFragmentInputGenNV) (GLuint  program, GLint  location, GLenum  genMode, GLint  components, const GLfloat * coeffs);
extern void (GLAPIENTRY *glProgramStringARB) (GLenum  target, GLenum  format, GLsizei  len, const void * string);
extern void (GLAPIENTRY *glProgramSubroutineParametersuivNV) (GLenum  target, GLsizei  count, const GLuint * params);
extern void (GLAPIENTRY *glProgramUniform1d) (GLuint  program, GLint  location, GLdouble  v0);
extern void (GLAPIENTRY *glProgramUniform1dEXT) (GLuint  program, GLint  location, GLdouble  x);
extern void (GLAPIENTRY *glProgramUniform1dv) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniform1dvEXT) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniform1f) (GLuint  program, GLint  location, GLfloat  v0);
extern void (GLAPIENTRY *glProgramUniform1fEXT) (GLuint  program, GLint  location, GLfloat  v0);
extern void (GLAPIENTRY *glProgramUniform1fv) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniform1fvEXT) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniform1i) (GLuint  program, GLint  location, GLint  v0);
extern void (GLAPIENTRY *glProgramUniform1i64ARB) (GLuint  program, GLint  location, GLint64  x);
extern void (GLAPIENTRY *glProgramUniform1i64NV) (GLuint  program, GLint  location, GLint64EXT  x);
extern void (GLAPIENTRY *glProgramUniform1i64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLint64 * value);
extern void (GLAPIENTRY *glProgramUniform1i64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLint64EXT * value);
extern void (GLAPIENTRY *glProgramUniform1iEXT) (GLuint  program, GLint  location, GLint  v0);
extern void (GLAPIENTRY *glProgramUniform1iv) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glProgramUniform1ivEXT) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glProgramUniform1ui) (GLuint  program, GLint  location, GLuint  v0);
extern void (GLAPIENTRY *glProgramUniform1ui64ARB) (GLuint  program, GLint  location, GLuint64  x);
extern void (GLAPIENTRY *glProgramUniform1ui64NV) (GLuint  program, GLint  location, GLuint64EXT  x);
extern void (GLAPIENTRY *glProgramUniform1ui64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glProgramUniform1ui64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glProgramUniform1uiEXT) (GLuint  program, GLint  location, GLuint  v0);
extern void (GLAPIENTRY *glProgramUniform1uiv) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glProgramUniform1uivEXT) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glProgramUniform2d) (GLuint  program, GLint  location, GLdouble  v0, GLdouble  v1);
extern void (GLAPIENTRY *glProgramUniform2dEXT) (GLuint  program, GLint  location, GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glProgramUniform2dv) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniform2dvEXT) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniform2f) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1);
extern void (GLAPIENTRY *glProgramUniform2fEXT) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1);
extern void (GLAPIENTRY *glProgramUniform2fv) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniform2fvEXT) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniform2i) (GLuint  program, GLint  location, GLint  v0, GLint  v1);
extern void (GLAPIENTRY *glProgramUniform2i64ARB) (GLuint  program, GLint  location, GLint64  x, GLint64  y);
extern void (GLAPIENTRY *glProgramUniform2i64NV) (GLuint  program, GLint  location, GLint64EXT  x, GLint64EXT  y);
extern void (GLAPIENTRY *glProgramUniform2i64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLint64 * value);
extern void (GLAPIENTRY *glProgramUniform2i64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLint64EXT * value);
extern void (GLAPIENTRY *glProgramUniform2iEXT) (GLuint  program, GLint  location, GLint  v0, GLint  v1);
extern void (GLAPIENTRY *glProgramUniform2iv) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glProgramUniform2ivEXT) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glProgramUniform2ui) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1);
extern void (GLAPIENTRY *glProgramUniform2ui64ARB) (GLuint  program, GLint  location, GLuint64  x, GLuint64  y);
extern void (GLAPIENTRY *glProgramUniform2ui64NV) (GLuint  program, GLint  location, GLuint64EXT  x, GLuint64EXT  y);
extern void (GLAPIENTRY *glProgramUniform2ui64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glProgramUniform2ui64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glProgramUniform2uiEXT) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1);
extern void (GLAPIENTRY *glProgramUniform2uiv) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glProgramUniform2uivEXT) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glProgramUniform3d) (GLuint  program, GLint  location, GLdouble  v0, GLdouble  v1, GLdouble  v2);
extern void (GLAPIENTRY *glProgramUniform3dEXT) (GLuint  program, GLint  location, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glProgramUniform3dv) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniform3dvEXT) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniform3f) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2);
extern void (GLAPIENTRY *glProgramUniform3fEXT) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2);
extern void (GLAPIENTRY *glProgramUniform3fv) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniform3fvEXT) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniform3i) (GLuint  program, GLint  location, GLint  v0, GLint  v1, GLint  v2);
extern void (GLAPIENTRY *glProgramUniform3i64ARB) (GLuint  program, GLint  location, GLint64  x, GLint64  y, GLint64  z);
extern void (GLAPIENTRY *glProgramUniform3i64NV) (GLuint  program, GLint  location, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z);
extern void (GLAPIENTRY *glProgramUniform3i64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLint64 * value);
extern void (GLAPIENTRY *glProgramUniform3i64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLint64EXT * value);
extern void (GLAPIENTRY *glProgramUniform3iEXT) (GLuint  program, GLint  location, GLint  v0, GLint  v1, GLint  v2);
extern void (GLAPIENTRY *glProgramUniform3iv) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glProgramUniform3ivEXT) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glProgramUniform3ui) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1, GLuint  v2);
extern void (GLAPIENTRY *glProgramUniform3ui64ARB) (GLuint  program, GLint  location, GLuint64  x, GLuint64  y, GLuint64  z);
extern void (GLAPIENTRY *glProgramUniform3ui64NV) (GLuint  program, GLint  location, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z);
extern void (GLAPIENTRY *glProgramUniform3ui64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glProgramUniform3ui64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glProgramUniform3uiEXT) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1, GLuint  v2);
extern void (GLAPIENTRY *glProgramUniform3uiv) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glProgramUniform3uivEXT) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glProgramUniform4d) (GLuint  program, GLint  location, GLdouble  v0, GLdouble  v1, GLdouble  v2, GLdouble  v3);
extern void (GLAPIENTRY *glProgramUniform4dEXT) (GLuint  program, GLint  location, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glProgramUniform4dv) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniform4dvEXT) (GLuint  program, GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniform4f) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2, GLfloat  v3);
extern void (GLAPIENTRY *glProgramUniform4fEXT) (GLuint  program, GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2, GLfloat  v3);
extern void (GLAPIENTRY *glProgramUniform4fv) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniform4fvEXT) (GLuint  program, GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniform4i) (GLuint  program, GLint  location, GLint  v0, GLint  v1, GLint  v2, GLint  v3);
extern void (GLAPIENTRY *glProgramUniform4i64ARB) (GLuint  program, GLint  location, GLint64  x, GLint64  y, GLint64  z, GLint64  w);
extern void (GLAPIENTRY *glProgramUniform4i64NV) (GLuint  program, GLint  location, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z, GLint64EXT  w);
extern void (GLAPIENTRY *glProgramUniform4i64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLint64 * value);
extern void (GLAPIENTRY *glProgramUniform4i64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLint64EXT * value);
extern void (GLAPIENTRY *glProgramUniform4iEXT) (GLuint  program, GLint  location, GLint  v0, GLint  v1, GLint  v2, GLint  v3);
extern void (GLAPIENTRY *glProgramUniform4iv) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glProgramUniform4ivEXT) (GLuint  program, GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glProgramUniform4ui) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1, GLuint  v2, GLuint  v3);
extern void (GLAPIENTRY *glProgramUniform4ui64ARB) (GLuint  program, GLint  location, GLuint64  x, GLuint64  y, GLuint64  z, GLuint64  w);
extern void (GLAPIENTRY *glProgramUniform4ui64NV) (GLuint  program, GLint  location, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z, GLuint64EXT  w);
extern void (GLAPIENTRY *glProgramUniform4ui64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glProgramUniform4ui64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glProgramUniform4uiEXT) (GLuint  program, GLint  location, GLuint  v0, GLuint  v1, GLuint  v2, GLuint  v3);
extern void (GLAPIENTRY *glProgramUniform4uiv) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glProgramUniform4uivEXT) (GLuint  program, GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glProgramUniformHandleui64ARB) (GLuint  program, GLint  location, GLuint64  value);
extern void (GLAPIENTRY *glProgramUniformHandleui64NV) (GLuint  program, GLint  location, GLuint64  value);
extern void (GLAPIENTRY *glProgramUniformHandleui64vARB) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * values);
extern void (GLAPIENTRY *glProgramUniformHandleui64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLuint64 * values);
extern void (GLAPIENTRY *glProgramUniformMatrix2dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2x3dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2x3dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2x3fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2x3fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2x4dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2x4dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2x4fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix2x4fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3x2dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3x2dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3x2fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3x2fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3x4dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3x4dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3x4fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix3x4fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4x2dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4x2dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4x2fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4x2fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4x3dv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4x3dvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4x3fv) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformMatrix4x3fvEXT) (GLuint  program, GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glProgramUniformui64NV) (GLuint  program, GLint  location, GLuint64EXT  value);
extern void (GLAPIENTRY *glProgramUniformui64vNV) (GLuint  program, GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glProgramVertexLimitNV) (GLenum  target, GLint  limit);
extern void (GLAPIENTRY *glProvokingVertex) (GLenum  mode);
extern void (GLAPIENTRY *glProvokingVertexEXT) (GLenum  mode);
extern void (GLAPIENTRY *glPushAttrib) (GLbitfield  mask);
extern void (GLAPIENTRY *glPushClientAttrib) (GLbitfield  mask);
extern void (GLAPIENTRY *glPushClientAttribDefaultEXT) (GLbitfield  mask);
extern void (GLAPIENTRY *glPushDebugGroup) (GLenum  source, GLuint  id, GLsizei  length, const GLchar * message);
extern void (GLAPIENTRY *glPushDebugGroupKHR) (GLenum  source, GLuint  id, GLsizei  length, const GLchar * message);
extern void (GLAPIENTRY *glPushGroupMarkerEXT) (GLsizei  length, const GLchar * marker);
extern void (GLAPIENTRY *glPushMatrix) ();
extern void (GLAPIENTRY *glPushName) (GLuint  name);
extern void (GLAPIENTRY *glQueryCounter) (GLuint  id, GLenum  target);
extern void (GLAPIENTRY *glQueryObjectParameteruiAMD) (GLenum  target, GLuint  id, GLenum  pname, GLuint  param);
extern GLint (GLAPIENTRY *glQueryResourceNV) (GLenum  queryType, GLint  tagId, GLuint  count, GLint * buffer);
extern void (GLAPIENTRY *glQueryResourceTagNV) (GLint  tagId, const GLchar * tagString);
extern void (GLAPIENTRY *glRasterPos2d) (GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glRasterPos2dv) (const GLdouble * v);
extern void (GLAPIENTRY *glRasterPos2f) (GLfloat  x, GLfloat  y);
extern void (GLAPIENTRY *glRasterPos2fv) (const GLfloat * v);
extern void (GLAPIENTRY *glRasterPos2i) (GLint  x, GLint  y);
extern void (GLAPIENTRY *glRasterPos2iv) (const GLint * v);
extern void (GLAPIENTRY *glRasterPos2s) (GLshort  x, GLshort  y);
extern void (GLAPIENTRY *glRasterPos2sv) (const GLshort * v);
extern void (GLAPIENTRY *glRasterPos3d) (GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glRasterPos3dv) (const GLdouble * v);
extern void (GLAPIENTRY *glRasterPos3f) (GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glRasterPos3fv) (const GLfloat * v);
extern void (GLAPIENTRY *glRasterPos3i) (GLint  x, GLint  y, GLint  z);
extern void (GLAPIENTRY *glRasterPos3iv) (const GLint * v);
extern void (GLAPIENTRY *glRasterPos3s) (GLshort  x, GLshort  y, GLshort  z);
extern void (GLAPIENTRY *glRasterPos3sv) (const GLshort * v);
extern void (GLAPIENTRY *glRasterPos4d) (GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glRasterPos4dv) (const GLdouble * v);
extern void (GLAPIENTRY *glRasterPos4f) (GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glRasterPos4fv) (const GLfloat * v);
extern void (GLAPIENTRY *glRasterPos4i) (GLint  x, GLint  y, GLint  z, GLint  w);
extern void (GLAPIENTRY *glRasterPos4iv) (const GLint * v);
extern void (GLAPIENTRY *glRasterPos4s) (GLshort  x, GLshort  y, GLshort  z, GLshort  w);
extern void (GLAPIENTRY *glRasterPos4sv) (const GLshort * v);
extern void (GLAPIENTRY *glRasterSamplesEXT) (GLuint  samples, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glReadBuffer) (GLenum  src);
extern void (GLAPIENTRY *glReadPixels) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, void * pixels);
extern void (GLAPIENTRY *glReadnPixels) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, GLsizei  bufSize, void * data);
extern void (GLAPIENTRY *glReadnPixelsARB) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, GLsizei  bufSize, void * data);
extern void (GLAPIENTRY *glReadnPixelsKHR) (GLint  x, GLint  y, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, GLsizei  bufSize, void * data);
extern GLboolean (GLAPIENTRY *glReleaseKeyedMutexWin32EXT) (GLuint  memory, GLuint64  key);
extern void (GLAPIENTRY *glRectd) (GLdouble  x1, GLdouble  y1, GLdouble  x2, GLdouble  y2);
extern void (GLAPIENTRY *glRectdv) (const GLdouble * v1, const GLdouble * v2);
extern void (GLAPIENTRY *glRectf) (GLfloat  x1, GLfloat  y1, GLfloat  x2, GLfloat  y2);
extern void (GLAPIENTRY *glRectfv) (const GLfloat * v1, const GLfloat * v2);
extern void (GLAPIENTRY *glRecti) (GLint  x1, GLint  y1, GLint  x2, GLint  y2);
extern void (GLAPIENTRY *glRectiv) (const GLint * v1, const GLint * v2);
extern void (GLAPIENTRY *glRects) (GLshort  x1, GLshort  y1, GLshort  x2, GLshort  y2);
extern void (GLAPIENTRY *glRectsv) (const GLshort * v1, const GLshort * v2);
extern void (GLAPIENTRY *glReleaseShaderCompiler) ();
extern void (GLAPIENTRY *glRenderGpuMaskNV) (GLbitfield  mask);
extern GLint (GLAPIENTRY *glRenderMode) (GLenum  mode);
extern void (GLAPIENTRY *glRenderbufferStorage) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glRenderbufferStorageEXT) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glRenderbufferStorageMultisample) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glRenderbufferStorageMultisampleAdvancedAMD) (GLenum  target, GLsizei  samples, GLsizei  storageSamples, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glRenderbufferStorageMultisampleCoverageNV) (GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glRenderbufferStorageMultisampleEXT) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glRequestResidentProgramsNV) (GLsizei  n, const GLuint * programs);
extern void (GLAPIENTRY *glResetHistogram) (GLenum  target);
extern void (GLAPIENTRY *glResetHistogramEXT) (GLenum  target);
extern void (GLAPIENTRY *glResetMemoryObjectParameterNV) (GLuint  memory, GLenum  pname);
extern void (GLAPIENTRY *glResetMinmax) (GLenum  target);
extern void (GLAPIENTRY *glResetMinmaxEXT) (GLenum  target);
extern void (GLAPIENTRY *glResolveDepthValuesNV) ();
extern void (GLAPIENTRY *glResumeTransformFeedback) ();
extern void (GLAPIENTRY *glResumeTransformFeedbackNV) ();
extern void (GLAPIENTRY *glRotated) (GLdouble  angle, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glRotatef) (GLfloat  angle, GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glSampleCoverage) (GLfloat  value, GLboolean  invert);
extern void (GLAPIENTRY *glSampleCoverageARB) (GLfloat  value, GLboolean  invert);
extern void (GLAPIENTRY *glSampleMaskEXT) (GLclampf  value, GLboolean  invert);
extern void (GLAPIENTRY *glSampleMaskIndexedNV) (GLuint  index, GLbitfield  mask);
extern void (GLAPIENTRY *glSampleMaski) (GLuint  maskNumber, GLbitfield  mask);
extern void (GLAPIENTRY *glSamplePatternEXT) (GLenum  pattern);
extern void (GLAPIENTRY *glSamplerParameterIiv) (GLuint  sampler, GLenum  pname, const GLint * param);
extern void (GLAPIENTRY *glSamplerParameterIuiv) (GLuint  sampler, GLenum  pname, const GLuint * param);
extern void (GLAPIENTRY *glSamplerParameterf) (GLuint  sampler, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glSamplerParameterfv) (GLuint  sampler, GLenum  pname, const GLfloat * param);
extern void (GLAPIENTRY *glSamplerParameteri) (GLuint  sampler, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glSamplerParameteriv) (GLuint  sampler, GLenum  pname, const GLint * param);
extern void (GLAPIENTRY *glScaled) (GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glScalef) (GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glScissor) (GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glScissorArrayv) (GLuint  first, GLsizei  count, const GLint * v);
extern void (GLAPIENTRY *glScissorExclusiveArrayvNV) (GLuint  first, GLsizei  count, const GLint * v);
extern void (GLAPIENTRY *glScissorExclusiveNV) (GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glScissorIndexed) (GLuint  index, GLint  left, GLint  bottom, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glScissorIndexedv) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glSecondaryColor3b) (GLbyte  red, GLbyte  green, GLbyte  blue);
extern void (GLAPIENTRY *glSecondaryColor3bEXT) (GLbyte  red, GLbyte  green, GLbyte  blue);
extern void (GLAPIENTRY *glSecondaryColor3bv) (const GLbyte * v);
extern void (GLAPIENTRY *glSecondaryColor3bvEXT) (const GLbyte * v);
extern void (GLAPIENTRY *glSecondaryColor3d) (GLdouble  red, GLdouble  green, GLdouble  blue);
extern void (GLAPIENTRY *glSecondaryColor3dEXT) (GLdouble  red, GLdouble  green, GLdouble  blue);
extern void (GLAPIENTRY *glSecondaryColor3dv) (const GLdouble * v);
extern void (GLAPIENTRY *glSecondaryColor3dvEXT) (const GLdouble * v);
extern void (GLAPIENTRY *glSecondaryColor3f) (GLfloat  red, GLfloat  green, GLfloat  blue);
extern void (GLAPIENTRY *glSecondaryColor3fEXT) (GLfloat  red, GLfloat  green, GLfloat  blue);
extern void (GLAPIENTRY *glSecondaryColor3fv) (const GLfloat * v);
extern void (GLAPIENTRY *glSecondaryColor3fvEXT) (const GLfloat * v);
extern void (GLAPIENTRY *glSecondaryColor3hNV) (GLhalfNV  red, GLhalfNV  green, GLhalfNV  blue);
extern void (GLAPIENTRY *glSecondaryColor3hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glSecondaryColor3i) (GLint  red, GLint  green, GLint  blue);
extern void (GLAPIENTRY *glSecondaryColor3iEXT) (GLint  red, GLint  green, GLint  blue);
extern void (GLAPIENTRY *glSecondaryColor3iv) (const GLint * v);
extern void (GLAPIENTRY *glSecondaryColor3ivEXT) (const GLint * v);
extern void (GLAPIENTRY *glSecondaryColor3s) (GLshort  red, GLshort  green, GLshort  blue);
extern void (GLAPIENTRY *glSecondaryColor3sEXT) (GLshort  red, GLshort  green, GLshort  blue);
extern void (GLAPIENTRY *glSecondaryColor3sv) (const GLshort * v);
extern void (GLAPIENTRY *glSecondaryColor3svEXT) (const GLshort * v);
extern void (GLAPIENTRY *glSecondaryColor3ub) (GLubyte  red, GLubyte  green, GLubyte  blue);
extern void (GLAPIENTRY *glSecondaryColor3ubEXT) (GLubyte  red, GLubyte  green, GLubyte  blue);
extern void (GLAPIENTRY *glSecondaryColor3ubv) (const GLubyte * v);
extern void (GLAPIENTRY *glSecondaryColor3ubvEXT) (const GLubyte * v);
extern void (GLAPIENTRY *glSecondaryColor3ui) (GLuint  red, GLuint  green, GLuint  blue);
extern void (GLAPIENTRY *glSecondaryColor3uiEXT) (GLuint  red, GLuint  green, GLuint  blue);
extern void (GLAPIENTRY *glSecondaryColor3uiv) (const GLuint * v);
extern void (GLAPIENTRY *glSecondaryColor3uivEXT) (const GLuint * v);
extern void (GLAPIENTRY *glSecondaryColor3us) (GLushort  red, GLushort  green, GLushort  blue);
extern void (GLAPIENTRY *glSecondaryColor3usEXT) (GLushort  red, GLushort  green, GLushort  blue);
extern void (GLAPIENTRY *glSecondaryColor3usv) (const GLushort * v);
extern void (GLAPIENTRY *glSecondaryColor3usvEXT) (const GLushort * v);
extern void (GLAPIENTRY *glSecondaryColorFormatNV) (GLint  size, GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glSecondaryColorP3ui) (GLenum  type, GLuint  color);
extern void (GLAPIENTRY *glSecondaryColorP3uiv) (GLenum  type, const GLuint * color);
extern void (GLAPIENTRY *glSecondaryColorPointer) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glSecondaryColorPointerEXT) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glSelectBuffer) (GLsizei  size, GLuint * buffer);
extern void (GLAPIENTRY *glSelectPerfMonitorCountersAMD) (GLuint  monitor, GLboolean  enable, GLuint  group, GLint  numCounters, GLuint * counterList);
extern void (GLAPIENTRY *glSemaphoreParameterui64vEXT) (GLuint  semaphore, GLenum  pname, const GLuint64 * params);
extern void (GLAPIENTRY *glSeparableFilter2D) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * row, const void * column);
extern void (GLAPIENTRY *glSeparableFilter2DEXT) (GLenum  target, GLenum  internalformat, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * row, const void * column);
extern void (GLAPIENTRY *glSetFenceAPPLE) (GLuint  fence);
extern void (GLAPIENTRY *glSetFenceNV) (GLuint  fence, GLenum  condition);
extern void (GLAPIENTRY *glSetInvariantEXT) (GLuint  id, GLenum  type, const void * addr);
extern void (GLAPIENTRY *glSetLocalConstantEXT) (GLuint  id, GLenum  type, const void * addr);
extern void (GLAPIENTRY *glSetMultisamplefvAMD) (GLenum  pname, GLuint  index, const GLfloat * val);
extern void (GLAPIENTRY *glShadeModel) (GLenum  mode);
extern void (GLAPIENTRY *glShaderBinary) (GLsizei  count, const GLuint * shaders, GLenum  binaryformat, const void * binary, GLsizei  length);
extern void (GLAPIENTRY *glShaderOp1EXT) (GLenum  op, GLuint  res, GLuint  arg1);
extern void (GLAPIENTRY *glShaderOp2EXT) (GLenum  op, GLuint  res, GLuint  arg1, GLuint  arg2);
extern void (GLAPIENTRY *glShaderOp3EXT) (GLenum  op, GLuint  res, GLuint  arg1, GLuint  arg2, GLuint  arg3);
extern void (GLAPIENTRY *glShaderSource) (GLuint  shader, GLsizei  count, const GLchar *const* string, const GLint * length);
extern void (GLAPIENTRY *glShaderSourceARB) (GLhandleARB  shaderObj, GLsizei  count, const GLcharARB ** string, const GLint * length);
extern void (GLAPIENTRY *glShaderStorageBlockBinding) (GLuint  program, GLuint  storageBlockIndex, GLuint  storageBlockBinding);
extern void (GLAPIENTRY *glShadingRateImageBarrierNV) (GLboolean  synchronize);
extern void (GLAPIENTRY *glShadingRateImagePaletteNV) (GLuint  viewport, GLuint  first, GLsizei  count, const GLenum * rates);
extern void (GLAPIENTRY *glShadingRateSampleOrderNV) (GLenum  order);
extern void (GLAPIENTRY *glShadingRateSampleOrderCustomNV) (GLenum  rate, GLuint  samples, const GLint * locations);
extern void (GLAPIENTRY *glSignalSemaphoreEXT) (GLuint  semaphore, GLuint  numBufferBarriers, const GLuint * buffers, GLuint  numTextureBarriers, const GLuint * textures, const GLenum * dstLayouts);
extern void (GLAPIENTRY *glSpecializeShader) (GLuint  shader, const GLchar * pEntryPoint, GLuint  numSpecializationConstants, const GLuint * pConstantIndex, const GLuint * pConstantValue);
extern void (GLAPIENTRY *glSpecializeShaderARB) (GLuint  shader, const GLchar * pEntryPoint, GLuint  numSpecializationConstants, const GLuint * pConstantIndex, const GLuint * pConstantValue);
extern void (GLAPIENTRY *glStateCaptureNV) (GLuint  state, GLenum  mode);
extern void (GLAPIENTRY *glStencilClearTagEXT) (GLsizei  stencilTagBits, GLuint  stencilClearTag);
extern void (GLAPIENTRY *glStencilFillPathInstancedNV) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLenum  fillMode, GLuint  mask, GLenum  transformType, const GLfloat * transformValues);
extern void (GLAPIENTRY *glStencilFillPathNV) (GLuint  path, GLenum  fillMode, GLuint  mask);
extern void (GLAPIENTRY *glStencilFunc) (GLenum  func, GLint  ref, GLuint  mask);
extern void (GLAPIENTRY *glStencilFuncSeparate) (GLenum  face, GLenum  func, GLint  ref, GLuint  mask);
extern void (GLAPIENTRY *glStencilMask) (GLuint  mask);
extern void (GLAPIENTRY *glStencilMaskSeparate) (GLenum  face, GLuint  mask);
extern void (GLAPIENTRY *glStencilOp) (GLenum  fail, GLenum  zfail, GLenum  zpass);
extern void (GLAPIENTRY *glStencilOpSeparate) (GLenum  face, GLenum  sfail, GLenum  dpfail, GLenum  dppass);
extern void (GLAPIENTRY *glStencilOpValueAMD) (GLenum  face, GLuint  value);
extern void (GLAPIENTRY *glStencilStrokePathInstancedNV) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLint  reference, GLuint  mask, GLenum  transformType, const GLfloat * transformValues);
extern void (GLAPIENTRY *glStencilStrokePathNV) (GLuint  path, GLint  reference, GLuint  mask);
extern void (GLAPIENTRY *glStencilThenCoverFillPathInstancedNV) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLenum  fillMode, GLuint  mask, GLenum  coverMode, GLenum  transformType, const GLfloat * transformValues);
extern void (GLAPIENTRY *glStencilThenCoverFillPathNV) (GLuint  path, GLenum  fillMode, GLuint  mask, GLenum  coverMode);
extern void (GLAPIENTRY *glStencilThenCoverStrokePathInstancedNV) (GLsizei  numPaths, GLenum  pathNameType, const void * paths, GLuint  pathBase, GLint  reference, GLuint  mask, GLenum  coverMode, GLenum  transformType, const GLfloat * transformValues);
extern void (GLAPIENTRY *glStencilThenCoverStrokePathNV) (GLuint  path, GLint  reference, GLuint  mask, GLenum  coverMode);
extern void (GLAPIENTRY *glSubpixelPrecisionBiasNV) (GLuint  xbits, GLuint  ybits);
extern void (GLAPIENTRY *glSwizzleEXT) (GLuint  res, GLuint  in, GLenum  outX, GLenum  outY, GLenum  outZ, GLenum  outW);
extern void (GLAPIENTRY *glSyncTextureINTEL) (GLuint  texture);
extern void (GLAPIENTRY *glTangent3bEXT) (GLbyte  tx, GLbyte  ty, GLbyte  tz);
extern void (GLAPIENTRY *glTangent3bvEXT) (const GLbyte * v);
extern void (GLAPIENTRY *glTangent3dEXT) (GLdouble  tx, GLdouble  ty, GLdouble  tz);
extern void (GLAPIENTRY *glTangent3dvEXT) (const GLdouble * v);
extern void (GLAPIENTRY *glTangent3fEXT) (GLfloat  tx, GLfloat  ty, GLfloat  tz);
extern void (GLAPIENTRY *glTangent3fvEXT) (const GLfloat * v);
extern void (GLAPIENTRY *glTangent3iEXT) (GLint  tx, GLint  ty, GLint  tz);
extern void (GLAPIENTRY *glTangent3ivEXT) (const GLint * v);
extern void (GLAPIENTRY *glTangent3sEXT) (GLshort  tx, GLshort  ty, GLshort  tz);
extern void (GLAPIENTRY *glTangent3svEXT) (const GLshort * v);
extern void (GLAPIENTRY *glTangentPointerEXT) (GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glTessellationFactorAMD) (GLfloat  factor);
extern void (GLAPIENTRY *glTessellationModeAMD) (GLenum  mode);
extern GLboolean (GLAPIENTRY *glTestFenceAPPLE) (GLuint  fence);
extern GLboolean (GLAPIENTRY *glTestFenceNV) (GLuint  fence);
extern GLboolean (GLAPIENTRY *glTestObjectAPPLE) (GLenum  object, GLuint  name);
extern void (GLAPIENTRY *glTexAttachMemoryNV) (GLenum  target, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTexBuffer) (GLenum  target, GLenum  internalformat, GLuint  buffer);
extern void (GLAPIENTRY *glTexBufferARB) (GLenum  target, GLenum  internalformat, GLuint  buffer);
extern void (GLAPIENTRY *glTexBufferEXT) (GLenum  target, GLenum  internalformat, GLuint  buffer);
extern void (GLAPIENTRY *glTexBufferRange) (GLenum  target, GLenum  internalformat, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
extern void (GLAPIENTRY *glTexCoord1d) (GLdouble  s);
extern void (GLAPIENTRY *glTexCoord1dv) (const GLdouble * v);
extern void (GLAPIENTRY *glTexCoord1f) (GLfloat  s);
extern void (GLAPIENTRY *glTexCoord1fv) (const GLfloat * v);
extern void (GLAPIENTRY *glTexCoord1hNV) (GLhalfNV  s);
extern void (GLAPIENTRY *glTexCoord1hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glTexCoord1i) (GLint  s);
extern void (GLAPIENTRY *glTexCoord1iv) (const GLint * v);
extern void (GLAPIENTRY *glTexCoord1s) (GLshort  s);
extern void (GLAPIENTRY *glTexCoord1sv) (const GLshort * v);
extern void (GLAPIENTRY *glTexCoord2d) (GLdouble  s, GLdouble  t);
extern void (GLAPIENTRY *glTexCoord2dv) (const GLdouble * v);
extern void (GLAPIENTRY *glTexCoord2f) (GLfloat  s, GLfloat  t);
extern void (GLAPIENTRY *glTexCoord2fv) (const GLfloat * v);
extern void (GLAPIENTRY *glTexCoord2hNV) (GLhalfNV  s, GLhalfNV  t);
extern void (GLAPIENTRY *glTexCoord2hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glTexCoord2i) (GLint  s, GLint  t);
extern void (GLAPIENTRY *glTexCoord2iv) (const GLint * v);
extern void (GLAPIENTRY *glTexCoord2s) (GLshort  s, GLshort  t);
extern void (GLAPIENTRY *glTexCoord2sv) (const GLshort * v);
extern void (GLAPIENTRY *glTexCoord3d) (GLdouble  s, GLdouble  t, GLdouble  r);
extern void (GLAPIENTRY *glTexCoord3dv) (const GLdouble * v);
extern void (GLAPIENTRY *glTexCoord3f) (GLfloat  s, GLfloat  t, GLfloat  r);
extern void (GLAPIENTRY *glTexCoord3fv) (const GLfloat * v);
extern void (GLAPIENTRY *glTexCoord3hNV) (GLhalfNV  s, GLhalfNV  t, GLhalfNV  r);
extern void (GLAPIENTRY *glTexCoord3hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glTexCoord3i) (GLint  s, GLint  t, GLint  r);
extern void (GLAPIENTRY *glTexCoord3iv) (const GLint * v);
extern void (GLAPIENTRY *glTexCoord3s) (GLshort  s, GLshort  t, GLshort  r);
extern void (GLAPIENTRY *glTexCoord3sv) (const GLshort * v);
extern void (GLAPIENTRY *glTexCoord4d) (GLdouble  s, GLdouble  t, GLdouble  r, GLdouble  q);
extern void (GLAPIENTRY *glTexCoord4dv) (const GLdouble * v);
extern void (GLAPIENTRY *glTexCoord4f) (GLfloat  s, GLfloat  t, GLfloat  r, GLfloat  q);
extern void (GLAPIENTRY *glTexCoord4fv) (const GLfloat * v);
extern void (GLAPIENTRY *glTexCoord4hNV) (GLhalfNV  s, GLhalfNV  t, GLhalfNV  r, GLhalfNV  q);
extern void (GLAPIENTRY *glTexCoord4hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glTexCoord4i) (GLint  s, GLint  t, GLint  r, GLint  q);
extern void (GLAPIENTRY *glTexCoord4iv) (const GLint * v);
extern void (GLAPIENTRY *glTexCoord4s) (GLshort  s, GLshort  t, GLshort  r, GLshort  q);
extern void (GLAPIENTRY *glTexCoord4sv) (const GLshort * v);
extern void (GLAPIENTRY *glTexCoordFormatNV) (GLint  size, GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glTexCoordP1ui) (GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glTexCoordP1uiv) (GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glTexCoordP2ui) (GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glTexCoordP2uiv) (GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glTexCoordP3ui) (GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glTexCoordP3uiv) (GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glTexCoordP4ui) (GLenum  type, GLuint  coords);
extern void (GLAPIENTRY *glTexCoordP4uiv) (GLenum  type, const GLuint * coords);
extern void (GLAPIENTRY *glTexCoordPointer) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glTexCoordPointerEXT) (GLint  size, GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
extern void (GLAPIENTRY *glTexCoordPointervINTEL) (GLint  size, GLenum  type, const void ** pointer);
extern void (GLAPIENTRY *glTexEnvf) (GLenum  target, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glTexEnvfv) (GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glTexEnvi) (GLenum  target, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glTexEnviv) (GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glTexGend) (GLenum  coord, GLenum  pname, GLdouble  param);
extern void (GLAPIENTRY *glTexGendv) (GLenum  coord, GLenum  pname, const GLdouble * params);
extern void (GLAPIENTRY *glTexGenf) (GLenum  coord, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glTexGenfv) (GLenum  coord, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glTexGeni) (GLenum  coord, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glTexGeniv) (GLenum  coord, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glTexImage1D) (GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexImage2D) (GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexImage2DMultisample) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glTexImage2DMultisampleCoverageNV) (GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations);
extern void (GLAPIENTRY *glTexImage3D) (GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexImage3DEXT) (GLenum  target, GLint  level, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexImage3DMultisample) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glTexImage3DMultisampleCoverageNV) (GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations);
extern void (GLAPIENTRY *glTexPageCommitmentARB) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  commit);
extern void (GLAPIENTRY *glTexParameterIiv) (GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glTexParameterIivEXT) (GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glTexParameterIuiv) (GLenum  target, GLenum  pname, const GLuint * params);
extern void (GLAPIENTRY *glTexParameterIuivEXT) (GLenum  target, GLenum  pname, const GLuint * params);
extern void (GLAPIENTRY *glTexParameterf) (GLenum  target, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glTexParameterfv) (GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glTexParameteri) (GLenum  target, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glTexParameteriv) (GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glTexRenderbufferNV) (GLenum  target, GLuint  renderbuffer);
extern void (GLAPIENTRY *glTexStorage1D) (GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width);
extern void (GLAPIENTRY *glTexStorage2D) (GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glTexStorage2DMultisample) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glTexStorage3D) (GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth);
extern void (GLAPIENTRY *glTexStorage3DMultisample) (GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glTexStorageMem1DEXT) (GLenum  target, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTexStorageMem2DEXT) (GLenum  target, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTexStorageMem2DMultisampleEXT) (GLenum  target, GLsizei  samples, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTexStorageMem3DEXT) (GLenum  target, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTexStorageMem3DMultisampleEXT) (GLenum  target, GLsizei  samples, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTexStorageSparseAMD) (GLenum  target, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLsizei  layers, GLbitfield  flags);
extern void (GLAPIENTRY *glTexSubImage1D) (GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexSubImage1DEXT) (GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexSubImage2D) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexSubImage2DEXT) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexSubImage3D) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTexSubImage3DEXT) (GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureAttachMemoryNV) (GLuint  texture, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTextureBarrier) ();
extern void (GLAPIENTRY *glTextureBarrierNV) ();
extern void (GLAPIENTRY *glTextureBuffer) (GLuint  texture, GLenum  internalformat, GLuint  buffer);
extern void (GLAPIENTRY *glTextureBufferEXT) (GLuint  texture, GLenum  target, GLenum  internalformat, GLuint  buffer);
extern void (GLAPIENTRY *glTextureBufferRange) (GLuint  texture, GLenum  internalformat, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
extern void (GLAPIENTRY *glTextureBufferRangeEXT) (GLuint  texture, GLenum  target, GLenum  internalformat, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
extern void (GLAPIENTRY *glTextureImage1DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureImage2DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureImage2DMultisampleCoverageNV) (GLuint  texture, GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations);
extern void (GLAPIENTRY *glTextureImage2DMultisampleNV) (GLuint  texture, GLenum  target, GLsizei  samples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations);
extern void (GLAPIENTRY *glTextureImage3DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLint  border, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureImage3DMultisampleCoverageNV) (GLuint  texture, GLenum  target, GLsizei  coverageSamples, GLsizei  colorSamples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations);
extern void (GLAPIENTRY *glTextureImage3DMultisampleNV) (GLuint  texture, GLenum  target, GLsizei  samples, GLint  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations);
extern void (GLAPIENTRY *glTextureLightEXT) (GLenum  pname);
extern void (GLAPIENTRY *glTextureMaterialEXT) (GLenum  face, GLenum  mode);
extern void (GLAPIENTRY *glTextureNormalEXT) (GLenum  mode);
extern void (GLAPIENTRY *glTexturePageCommitmentEXT) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  commit);
extern void (GLAPIENTRY *glTextureParameterIiv) (GLuint  texture, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glTextureParameterIivEXT) (GLuint  texture, GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glTextureParameterIuiv) (GLuint  texture, GLenum  pname, const GLuint * params);
extern void (GLAPIENTRY *glTextureParameterIuivEXT) (GLuint  texture, GLenum  target, GLenum  pname, const GLuint * params);
extern void (GLAPIENTRY *glTextureParameterf) (GLuint  texture, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glTextureParameterfEXT) (GLuint  texture, GLenum  target, GLenum  pname, GLfloat  param);
extern void (GLAPIENTRY *glTextureParameterfv) (GLuint  texture, GLenum  pname, const GLfloat * param);
extern void (GLAPIENTRY *glTextureParameterfvEXT) (GLuint  texture, GLenum  target, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glTextureParameteri) (GLuint  texture, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glTextureParameteriEXT) (GLuint  texture, GLenum  target, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glTextureParameteriv) (GLuint  texture, GLenum  pname, const GLint * param);
extern void (GLAPIENTRY *glTextureParameterivEXT) (GLuint  texture, GLenum  target, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glTextureRangeAPPLE) (GLenum  target, GLsizei  length, const void * pointer);
extern void (GLAPIENTRY *glTextureRenderbufferEXT) (GLuint  texture, GLenum  target, GLuint  renderbuffer);
extern void (GLAPIENTRY *glTextureStorage1D) (GLuint  texture, GLsizei  levels, GLenum  internalformat, GLsizei  width);
extern void (GLAPIENTRY *glTextureStorage1DEXT) (GLuint  texture, GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width);
extern void (GLAPIENTRY *glTextureStorage2D) (GLuint  texture, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glTextureStorage2DEXT) (GLuint  texture, GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glTextureStorage2DMultisample) (GLuint  texture, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glTextureStorage2DMultisampleEXT) (GLuint  texture, GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glTextureStorage3D) (GLuint  texture, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth);
extern void (GLAPIENTRY *glTextureStorage3DEXT) (GLuint  texture, GLenum  target, GLsizei  levels, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth);
extern void (GLAPIENTRY *glTextureStorage3DMultisample) (GLuint  texture, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glTextureStorage3DMultisampleEXT) (GLuint  texture, GLenum  target, GLsizei  samples, GLenum  internalformat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedsamplelocations);
extern void (GLAPIENTRY *glTextureStorageMem1DEXT) (GLuint  texture, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTextureStorageMem2DEXT) (GLuint  texture, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTextureStorageMem2DMultisampleEXT) (GLuint  texture, GLsizei  samples, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLboolean  fixedSampleLocations, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTextureStorageMem3DEXT) (GLuint  texture, GLsizei  levels, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTextureStorageMem3DMultisampleEXT) (GLuint  texture, GLsizei  samples, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLboolean  fixedSampleLocations, GLuint  memory, GLuint64  offset);
extern void (GLAPIENTRY *glTextureStorageSparseAMD) (GLuint  texture, GLenum  target, GLenum  internalFormat, GLsizei  width, GLsizei  height, GLsizei  depth, GLsizei  layers, GLbitfield  flags);
extern void (GLAPIENTRY *glTextureSubImage1D) (GLuint  texture, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureSubImage1DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLsizei  width, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureSubImage2D) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureSubImage2DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLsizei  width, GLsizei  height, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureSubImage3D) (GLuint  texture, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureSubImage3DEXT) (GLuint  texture, GLenum  target, GLint  level, GLint  xoffset, GLint  yoffset, GLint  zoffset, GLsizei  width, GLsizei  height, GLsizei  depth, GLenum  format, GLenum  type, const void * pixels);
extern void (GLAPIENTRY *glTextureView) (GLuint  texture, GLenum  target, GLuint  origtexture, GLenum  internalformat, GLuint  minlevel, GLuint  numlevels, GLuint  minlayer, GLuint  numlayers);
extern void (GLAPIENTRY *glTrackMatrixNV) (GLenum  target, GLuint  address, GLenum  matrix, GLenum  transform);
extern void (GLAPIENTRY *glTransformFeedbackAttribsNV) (GLsizei  count, const GLint * attribs, GLenum  bufferMode);
extern void (GLAPIENTRY *glTransformFeedbackBufferBase) (GLuint  xfb, GLuint  index, GLuint  buffer);
extern void (GLAPIENTRY *glTransformFeedbackBufferRange) (GLuint  xfb, GLuint  index, GLuint  buffer, GLintptr  offset, GLsizeiptr  size);
extern void (GLAPIENTRY *glTransformFeedbackStreamAttribsNV) (GLsizei  count, const GLint * attribs, GLsizei  nbuffers, const GLint * bufstreams, GLenum  bufferMode);
extern void (GLAPIENTRY *glTransformFeedbackVaryings) (GLuint  program, GLsizei  count, const GLchar *const* varyings, GLenum  bufferMode);
extern void (GLAPIENTRY *glTransformFeedbackVaryingsEXT) (GLuint  program, GLsizei  count, const GLchar *const* varyings, GLenum  bufferMode);
extern void (GLAPIENTRY *glTransformFeedbackVaryingsNV) (GLuint  program, GLsizei  count, const GLint * locations, GLenum  bufferMode);
extern void (GLAPIENTRY *glTransformPathNV) (GLuint  resultPath, GLuint  srcPath, GLenum  transformType, const GLfloat * transformValues);
extern void (GLAPIENTRY *glTranslated) (GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glTranslatef) (GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glUniform1d) (GLint  location, GLdouble  x);
extern void (GLAPIENTRY *glUniform1dv) (GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glUniform1f) (GLint  location, GLfloat  v0);
extern void (GLAPIENTRY *glUniform1fARB) (GLint  location, GLfloat  v0);
extern void (GLAPIENTRY *glUniform1fv) (GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glUniform1fvARB) (GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glUniform1i) (GLint  location, GLint  v0);
extern void (GLAPIENTRY *glUniform1i64ARB) (GLint  location, GLint64  x);
extern void (GLAPIENTRY *glUniform1i64NV) (GLint  location, GLint64EXT  x);
extern void (GLAPIENTRY *glUniform1i64vARB) (GLint  location, GLsizei  count, const GLint64 * value);
extern void (GLAPIENTRY *glUniform1i64vNV) (GLint  location, GLsizei  count, const GLint64EXT * value);
extern void (GLAPIENTRY *glUniform1iARB) (GLint  location, GLint  v0);
extern void (GLAPIENTRY *glUniform1iv) (GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glUniform1ivARB) (GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glUniform1ui) (GLint  location, GLuint  v0);
extern void (GLAPIENTRY *glUniform1ui64ARB) (GLint  location, GLuint64  x);
extern void (GLAPIENTRY *glUniform1ui64NV) (GLint  location, GLuint64EXT  x);
extern void (GLAPIENTRY *glUniform1ui64vARB) (GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glUniform1ui64vNV) (GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glUniform1uiEXT) (GLint  location, GLuint  v0);
extern void (GLAPIENTRY *glUniform1uiv) (GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glUniform1uivEXT) (GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glUniform2d) (GLint  location, GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glUniform2dv) (GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glUniform2f) (GLint  location, GLfloat  v0, GLfloat  v1);
extern void (GLAPIENTRY *glUniform2fARB) (GLint  location, GLfloat  v0, GLfloat  v1);
extern void (GLAPIENTRY *glUniform2fv) (GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glUniform2fvARB) (GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glUniform2i) (GLint  location, GLint  v0, GLint  v1);
extern void (GLAPIENTRY *glUniform2i64ARB) (GLint  location, GLint64  x, GLint64  y);
extern void (GLAPIENTRY *glUniform2i64NV) (GLint  location, GLint64EXT  x, GLint64EXT  y);
extern void (GLAPIENTRY *glUniform2i64vARB) (GLint  location, GLsizei  count, const GLint64 * value);
extern void (GLAPIENTRY *glUniform2i64vNV) (GLint  location, GLsizei  count, const GLint64EXT * value);
extern void (GLAPIENTRY *glUniform2iARB) (GLint  location, GLint  v0, GLint  v1);
extern void (GLAPIENTRY *glUniform2iv) (GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glUniform2ivARB) (GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glUniform2ui) (GLint  location, GLuint  v0, GLuint  v1);
extern void (GLAPIENTRY *glUniform2ui64ARB) (GLint  location, GLuint64  x, GLuint64  y);
extern void (GLAPIENTRY *glUniform2ui64NV) (GLint  location, GLuint64EXT  x, GLuint64EXT  y);
extern void (GLAPIENTRY *glUniform2ui64vARB) (GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glUniform2ui64vNV) (GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glUniform2uiEXT) (GLint  location, GLuint  v0, GLuint  v1);
extern void (GLAPIENTRY *glUniform2uiv) (GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glUniform2uivEXT) (GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glUniform3d) (GLint  location, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glUniform3dv) (GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glUniform3f) (GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2);
extern void (GLAPIENTRY *glUniform3fARB) (GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2);
extern void (GLAPIENTRY *glUniform3fv) (GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glUniform3fvARB) (GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glUniform3i) (GLint  location, GLint  v0, GLint  v1, GLint  v2);
extern void (GLAPIENTRY *glUniform3i64ARB) (GLint  location, GLint64  x, GLint64  y, GLint64  z);
extern void (GLAPIENTRY *glUniform3i64NV) (GLint  location, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z);
extern void (GLAPIENTRY *glUniform3i64vARB) (GLint  location, GLsizei  count, const GLint64 * value);
extern void (GLAPIENTRY *glUniform3i64vNV) (GLint  location, GLsizei  count, const GLint64EXT * value);
extern void (GLAPIENTRY *glUniform3iARB) (GLint  location, GLint  v0, GLint  v1, GLint  v2);
extern void (GLAPIENTRY *glUniform3iv) (GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glUniform3ivARB) (GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glUniform3ui) (GLint  location, GLuint  v0, GLuint  v1, GLuint  v2);
extern void (GLAPIENTRY *glUniform3ui64ARB) (GLint  location, GLuint64  x, GLuint64  y, GLuint64  z);
extern void (GLAPIENTRY *glUniform3ui64NV) (GLint  location, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z);
extern void (GLAPIENTRY *glUniform3ui64vARB) (GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glUniform3ui64vNV) (GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glUniform3uiEXT) (GLint  location, GLuint  v0, GLuint  v1, GLuint  v2);
extern void (GLAPIENTRY *glUniform3uiv) (GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glUniform3uivEXT) (GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glUniform4d) (GLint  location, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glUniform4dv) (GLint  location, GLsizei  count, const GLdouble * value);
extern void (GLAPIENTRY *glUniform4f) (GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2, GLfloat  v3);
extern void (GLAPIENTRY *glUniform4fARB) (GLint  location, GLfloat  v0, GLfloat  v1, GLfloat  v2, GLfloat  v3);
extern void (GLAPIENTRY *glUniform4fv) (GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glUniform4fvARB) (GLint  location, GLsizei  count, const GLfloat * value);
extern void (GLAPIENTRY *glUniform4i) (GLint  location, GLint  v0, GLint  v1, GLint  v2, GLint  v3);
extern void (GLAPIENTRY *glUniform4i64ARB) (GLint  location, GLint64  x, GLint64  y, GLint64  z, GLint64  w);
extern void (GLAPIENTRY *glUniform4i64NV) (GLint  location, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z, GLint64EXT  w);
extern void (GLAPIENTRY *glUniform4i64vARB) (GLint  location, GLsizei  count, const GLint64 * value);
extern void (GLAPIENTRY *glUniform4i64vNV) (GLint  location, GLsizei  count, const GLint64EXT * value);
extern void (GLAPIENTRY *glUniform4iARB) (GLint  location, GLint  v0, GLint  v1, GLint  v2, GLint  v3);
extern void (GLAPIENTRY *glUniform4iv) (GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glUniform4ivARB) (GLint  location, GLsizei  count, const GLint * value);
extern void (GLAPIENTRY *glUniform4ui) (GLint  location, GLuint  v0, GLuint  v1, GLuint  v2, GLuint  v3);
extern void (GLAPIENTRY *glUniform4ui64ARB) (GLint  location, GLuint64  x, GLuint64  y, GLuint64  z, GLuint64  w);
extern void (GLAPIENTRY *glUniform4ui64NV) (GLint  location, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z, GLuint64EXT  w);
extern void (GLAPIENTRY *glUniform4ui64vARB) (GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glUniform4ui64vNV) (GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glUniform4uiEXT) (GLint  location, GLuint  v0, GLuint  v1, GLuint  v2, GLuint  v3);
extern void (GLAPIENTRY *glUniform4uiv) (GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glUniform4uivEXT) (GLint  location, GLsizei  count, const GLuint * value);
extern void (GLAPIENTRY *glUniformBlockBinding) (GLuint  program, GLuint  uniformBlockIndex, GLuint  uniformBlockBinding);
extern void (GLAPIENTRY *glUniformBufferEXT) (GLuint  program, GLint  location, GLuint  buffer);
extern void (GLAPIENTRY *glUniformHandleui64ARB) (GLint  location, GLuint64  value);
extern void (GLAPIENTRY *glUniformHandleui64NV) (GLint  location, GLuint64  value);
extern void (GLAPIENTRY *glUniformHandleui64vARB) (GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glUniformHandleui64vNV) (GLint  location, GLsizei  count, const GLuint64 * value);
extern void (GLAPIENTRY *glUniformMatrix2dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix2fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix2fvARB) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix2x3dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix2x3fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix2x4dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix2x4fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix3dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix3fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix3fvARB) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix3x2dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix3x2fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix3x4dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix3x4fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix4dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix4fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix4fvARB) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix4x2dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix4x2fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformMatrix4x3dv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLdouble * value);
extern void (GLAPIENTRY *glUniformMatrix4x3fv) (GLint  location, GLsizei  count, GLboolean  transpose, const GLfloat * value);
extern void (GLAPIENTRY *glUniformSubroutinesuiv) (GLenum  shadertype, GLsizei  count, const GLuint * indices);
extern void (GLAPIENTRY *glUniformui64NV) (GLint  location, GLuint64EXT  value);
extern void (GLAPIENTRY *glUniformui64vNV) (GLint  location, GLsizei  count, const GLuint64EXT * value);
extern void (GLAPIENTRY *glUnlockArraysEXT) ();
extern GLboolean (GLAPIENTRY *glUnmapBuffer) (GLenum  target);
extern GLboolean (GLAPIENTRY *glUnmapBufferARB) (GLenum  target);
extern GLboolean (GLAPIENTRY *glUnmapNamedBuffer) (GLuint  buffer);
extern GLboolean (GLAPIENTRY *glUnmapNamedBufferEXT) (GLuint  buffer);
extern void (GLAPIENTRY *glUnmapTexture2DINTEL) (GLuint  texture, GLint  level);
extern void (GLAPIENTRY *glUseProgram) (GLuint  program);
extern void (GLAPIENTRY *glUseProgramObjectARB) (GLhandleARB  programObj);
extern void (GLAPIENTRY *glUseProgramStages) (GLuint  pipeline, GLbitfield  stages, GLuint  program);
extern void (GLAPIENTRY *glUseProgramStagesEXT) (GLuint  pipeline, GLbitfield  stages, GLuint  program);
extern void (GLAPIENTRY *glUseShaderProgramEXT) (GLenum  type, GLuint  program);
extern void (GLAPIENTRY *glVDPAUFiniNV) ();
extern void (GLAPIENTRY *glVDPAUGetSurfaceivNV) (GLvdpauSurfaceNV  surface, GLenum  pname, GLsizei  count, GLsizei * length, GLint * values);
extern void (GLAPIENTRY *glVDPAUInitNV) (const void * vdpDevice, const void * getProcAddress);
extern GLboolean (GLAPIENTRY *glVDPAUIsSurfaceNV) (GLvdpauSurfaceNV  surface);
extern void (GLAPIENTRY *glVDPAUMapSurfacesNV) (GLsizei  numSurfaces, const GLvdpauSurfaceNV * surfaces);
extern GLvdpauSurfaceNV (GLAPIENTRY *glVDPAURegisterOutputSurfaceNV) (const void * vdpSurface, GLenum  target, GLsizei  numTextureNames, const GLuint * textureNames);
extern GLvdpauSurfaceNV (GLAPIENTRY *glVDPAURegisterVideoSurfaceNV) (const void * vdpSurface, GLenum  target, GLsizei  numTextureNames, const GLuint * textureNames);
extern GLvdpauSurfaceNV (GLAPIENTRY *glVDPAURegisterVideoSurfaceWithPictureStructureNV) (const void * vdpSurface, GLenum  target, GLsizei  numTextureNames, const GLuint * textureNames, GLboolean  isFrameStructure);
extern void (GLAPIENTRY *glVDPAUSurfaceAccessNV) (GLvdpauSurfaceNV  surface, GLenum  access);
extern void (GLAPIENTRY *glVDPAUUnmapSurfacesNV) (GLsizei  numSurface, const GLvdpauSurfaceNV * surfaces);
extern void (GLAPIENTRY *glVDPAUUnregisterSurfaceNV) (GLvdpauSurfaceNV  surface);
extern void (GLAPIENTRY *glValidateProgram) (GLuint  program);
extern void (GLAPIENTRY *glValidateProgramARB) (GLhandleARB  programObj);
extern void (GLAPIENTRY *glValidateProgramPipeline) (GLuint  pipeline);
extern void (GLAPIENTRY *glValidateProgramPipelineEXT) (GLuint  pipeline);
extern void (GLAPIENTRY *glVariantPointerEXT) (GLuint  id, GLenum  type, GLuint  stride, const void * addr);
extern void (GLAPIENTRY *glVariantbvEXT) (GLuint  id, const GLbyte * addr);
extern void (GLAPIENTRY *glVariantdvEXT) (GLuint  id, const GLdouble * addr);
extern void (GLAPIENTRY *glVariantfvEXT) (GLuint  id, const GLfloat * addr);
extern void (GLAPIENTRY *glVariantivEXT) (GLuint  id, const GLint * addr);
extern void (GLAPIENTRY *glVariantsvEXT) (GLuint  id, const GLshort * addr);
extern void (GLAPIENTRY *glVariantubvEXT) (GLuint  id, const GLubyte * addr);
extern void (GLAPIENTRY *glVariantuivEXT) (GLuint  id, const GLuint * addr);
extern void (GLAPIENTRY *glVariantusvEXT) (GLuint  id, const GLushort * addr);
extern void (GLAPIENTRY *glVertex2d) (GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glVertex2dv) (const GLdouble * v);
extern void (GLAPIENTRY *glVertex2f) (GLfloat  x, GLfloat  y);
extern void (GLAPIENTRY *glVertex2fv) (const GLfloat * v);
extern void (GLAPIENTRY *glVertex2hNV) (GLhalfNV  x, GLhalfNV  y);
extern void (GLAPIENTRY *glVertex2hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glVertex2i) (GLint  x, GLint  y);
extern void (GLAPIENTRY *glVertex2iv) (const GLint * v);
extern void (GLAPIENTRY *glVertex2s) (GLshort  x, GLshort  y);
extern void (GLAPIENTRY *glVertex2sv) (const GLshort * v);
extern void (GLAPIENTRY *glVertex3d) (GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glVertex3dv) (const GLdouble * v);
extern void (GLAPIENTRY *glVertex3f) (GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glVertex3fv) (const GLfloat * v);
extern void (GLAPIENTRY *glVertex3hNV) (GLhalfNV  x, GLhalfNV  y, GLhalfNV  z);
extern void (GLAPIENTRY *glVertex3hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glVertex3i) (GLint  x, GLint  y, GLint  z);
extern void (GLAPIENTRY *glVertex3iv) (const GLint * v);
extern void (GLAPIENTRY *glVertex3s) (GLshort  x, GLshort  y, GLshort  z);
extern void (GLAPIENTRY *glVertex3sv) (const GLshort * v);
extern void (GLAPIENTRY *glVertex4d) (GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glVertex4dv) (const GLdouble * v);
extern void (GLAPIENTRY *glVertex4f) (GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glVertex4fv) (const GLfloat * v);
extern void (GLAPIENTRY *glVertex4hNV) (GLhalfNV  x, GLhalfNV  y, GLhalfNV  z, GLhalfNV  w);
extern void (GLAPIENTRY *glVertex4hvNV) (const GLhalfNV * v);
extern void (GLAPIENTRY *glVertex4i) (GLint  x, GLint  y, GLint  z, GLint  w);
extern void (GLAPIENTRY *glVertex4iv) (const GLint * v);
extern void (GLAPIENTRY *glVertex4s) (GLshort  x, GLshort  y, GLshort  z, GLshort  w);
extern void (GLAPIENTRY *glVertex4sv) (const GLshort * v);
extern void (GLAPIENTRY *glVertexArrayAttribBinding) (GLuint  vaobj, GLuint  attribindex, GLuint  bindingindex);
extern void (GLAPIENTRY *glVertexArrayAttribFormat) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLboolean  normalized, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexArrayAttribIFormat) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexArrayAttribLFormat) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexArrayBindVertexBufferEXT) (GLuint  vaobj, GLuint  bindingindex, GLuint  buffer, GLintptr  offset, GLsizei  stride);
extern void (GLAPIENTRY *glVertexArrayBindingDivisor) (GLuint  vaobj, GLuint  bindingindex, GLuint  divisor);
extern void (GLAPIENTRY *glVertexArrayColorOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayEdgeFlagOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayElementBuffer) (GLuint  vaobj, GLuint  buffer);
extern void (GLAPIENTRY *glVertexArrayFogCoordOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayIndexOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayMultiTexCoordOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLenum  texunit, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayNormalOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayParameteriAPPLE) (GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glVertexArrayRangeAPPLE) (GLsizei  length, void * pointer);
extern void (GLAPIENTRY *glVertexArrayRangeNV) (GLsizei  length, const void * pointer);
extern void (GLAPIENTRY *glVertexArraySecondaryColorOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayTexCoordOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayVertexAttribBindingEXT) (GLuint  vaobj, GLuint  attribindex, GLuint  bindingindex);
extern void (GLAPIENTRY *glVertexArrayVertexAttribDivisorEXT) (GLuint  vaobj, GLuint  index, GLuint  divisor);
extern void (GLAPIENTRY *glVertexArrayVertexAttribFormatEXT) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLboolean  normalized, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexArrayVertexAttribIFormatEXT) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexArrayVertexAttribIOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLuint  index, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayVertexAttribLFormatEXT) (GLuint  vaobj, GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexArrayVertexAttribLOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLuint  index, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayVertexAttribOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLuint  index, GLint  size, GLenum  type, GLboolean  normalized, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexArrayVertexBindingDivisorEXT) (GLuint  vaobj, GLuint  bindingindex, GLuint  divisor);
extern void (GLAPIENTRY *glVertexArrayVertexBuffer) (GLuint  vaobj, GLuint  bindingindex, GLuint  buffer, GLintptr  offset, GLsizei  stride);
extern void (GLAPIENTRY *glVertexArrayVertexBuffers) (GLuint  vaobj, GLuint  first, GLsizei  count, const GLuint * buffers, const GLintptr * offsets, const GLsizei * strides);
extern void (GLAPIENTRY *glVertexArrayVertexOffsetEXT) (GLuint  vaobj, GLuint  buffer, GLint  size, GLenum  type, GLsizei  stride, GLintptr  offset);
extern void (GLAPIENTRY *glVertexAttrib1d) (GLuint  index, GLdouble  x);
extern void (GLAPIENTRY *glVertexAttrib1dARB) (GLuint  index, GLdouble  x);
extern void (GLAPIENTRY *glVertexAttrib1dNV) (GLuint  index, GLdouble  x);
extern void (GLAPIENTRY *glVertexAttrib1dv) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib1dvARB) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib1dvNV) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib1f) (GLuint  index, GLfloat  x);
extern void (GLAPIENTRY *glVertexAttrib1fARB) (GLuint  index, GLfloat  x);
extern void (GLAPIENTRY *glVertexAttrib1fNV) (GLuint  index, GLfloat  x);
extern void (GLAPIENTRY *glVertexAttrib1fv) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib1fvARB) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib1fvNV) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib1hNV) (GLuint  index, GLhalfNV  x);
extern void (GLAPIENTRY *glVertexAttrib1hvNV) (GLuint  index, const GLhalfNV * v);
extern void (GLAPIENTRY *glVertexAttrib1s) (GLuint  index, GLshort  x);
extern void (GLAPIENTRY *glVertexAttrib1sARB) (GLuint  index, GLshort  x);
extern void (GLAPIENTRY *glVertexAttrib1sNV) (GLuint  index, GLshort  x);
extern void (GLAPIENTRY *glVertexAttrib1sv) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib1svARB) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib1svNV) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib2d) (GLuint  index, GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glVertexAttrib2dARB) (GLuint  index, GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glVertexAttrib2dNV) (GLuint  index, GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glVertexAttrib2dv) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib2dvARB) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib2dvNV) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib2f) (GLuint  index, GLfloat  x, GLfloat  y);
extern void (GLAPIENTRY *glVertexAttrib2fARB) (GLuint  index, GLfloat  x, GLfloat  y);
extern void (GLAPIENTRY *glVertexAttrib2fNV) (GLuint  index, GLfloat  x, GLfloat  y);
extern void (GLAPIENTRY *glVertexAttrib2fv) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib2fvARB) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib2fvNV) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib2hNV) (GLuint  index, GLhalfNV  x, GLhalfNV  y);
extern void (GLAPIENTRY *glVertexAttrib2hvNV) (GLuint  index, const GLhalfNV * v);
extern void (GLAPIENTRY *glVertexAttrib2s) (GLuint  index, GLshort  x, GLshort  y);
extern void (GLAPIENTRY *glVertexAttrib2sARB) (GLuint  index, GLshort  x, GLshort  y);
extern void (GLAPIENTRY *glVertexAttrib2sNV) (GLuint  index, GLshort  x, GLshort  y);
extern void (GLAPIENTRY *glVertexAttrib2sv) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib2svARB) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib2svNV) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib3d) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glVertexAttrib3dARB) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glVertexAttrib3dNV) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glVertexAttrib3dv) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib3dvARB) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib3dvNV) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib3f) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glVertexAttrib3fARB) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glVertexAttrib3fNV) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glVertexAttrib3fv) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib3fvARB) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib3fvNV) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib3hNV) (GLuint  index, GLhalfNV  x, GLhalfNV  y, GLhalfNV  z);
extern void (GLAPIENTRY *glVertexAttrib3hvNV) (GLuint  index, const GLhalfNV * v);
extern void (GLAPIENTRY *glVertexAttrib3s) (GLuint  index, GLshort  x, GLshort  y, GLshort  z);
extern void (GLAPIENTRY *glVertexAttrib3sARB) (GLuint  index, GLshort  x, GLshort  y, GLshort  z);
extern void (GLAPIENTRY *glVertexAttrib3sNV) (GLuint  index, GLshort  x, GLshort  y, GLshort  z);
extern void (GLAPIENTRY *glVertexAttrib3sv) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib3svARB) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib3svNV) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib4Nbv) (GLuint  index, const GLbyte * v);
extern void (GLAPIENTRY *glVertexAttrib4NbvARB) (GLuint  index, const GLbyte * v);
extern void (GLAPIENTRY *glVertexAttrib4Niv) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttrib4NivARB) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttrib4Nsv) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib4NsvARB) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib4Nub) (GLuint  index, GLubyte  x, GLubyte  y, GLubyte  z, GLubyte  w);
extern void (GLAPIENTRY *glVertexAttrib4NubARB) (GLuint  index, GLubyte  x, GLubyte  y, GLubyte  z, GLubyte  w);
extern void (GLAPIENTRY *glVertexAttrib4Nubv) (GLuint  index, const GLubyte * v);
extern void (GLAPIENTRY *glVertexAttrib4NubvARB) (GLuint  index, const GLubyte * v);
extern void (GLAPIENTRY *glVertexAttrib4Nuiv) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttrib4NuivARB) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttrib4Nusv) (GLuint  index, const GLushort * v);
extern void (GLAPIENTRY *glVertexAttrib4NusvARB) (GLuint  index, const GLushort * v);
extern void (GLAPIENTRY *glVertexAttrib4bv) (GLuint  index, const GLbyte * v);
extern void (GLAPIENTRY *glVertexAttrib4bvARB) (GLuint  index, const GLbyte * v);
extern void (GLAPIENTRY *glVertexAttrib4d) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glVertexAttrib4dARB) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glVertexAttrib4dNV) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glVertexAttrib4dv) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib4dvARB) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib4dvNV) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttrib4f) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glVertexAttrib4fARB) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glVertexAttrib4fNV) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  z, GLfloat  w);
extern void (GLAPIENTRY *glVertexAttrib4fv) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib4fvARB) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib4fvNV) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttrib4hNV) (GLuint  index, GLhalfNV  x, GLhalfNV  y, GLhalfNV  z, GLhalfNV  w);
extern void (GLAPIENTRY *glVertexAttrib4hvNV) (GLuint  index, const GLhalfNV * v);
extern void (GLAPIENTRY *glVertexAttrib4iv) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttrib4ivARB) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttrib4s) (GLuint  index, GLshort  x, GLshort  y, GLshort  z, GLshort  w);
extern void (GLAPIENTRY *glVertexAttrib4sARB) (GLuint  index, GLshort  x, GLshort  y, GLshort  z, GLshort  w);
extern void (GLAPIENTRY *glVertexAttrib4sNV) (GLuint  index, GLshort  x, GLshort  y, GLshort  z, GLshort  w);
extern void (GLAPIENTRY *glVertexAttrib4sv) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib4svARB) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib4svNV) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttrib4ubNV) (GLuint  index, GLubyte  x, GLubyte  y, GLubyte  z, GLubyte  w);
extern void (GLAPIENTRY *glVertexAttrib4ubv) (GLuint  index, const GLubyte * v);
extern void (GLAPIENTRY *glVertexAttrib4ubvARB) (GLuint  index, const GLubyte * v);
extern void (GLAPIENTRY *glVertexAttrib4ubvNV) (GLuint  index, const GLubyte * v);
extern void (GLAPIENTRY *glVertexAttrib4uiv) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttrib4uivARB) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttrib4usv) (GLuint  index, const GLushort * v);
extern void (GLAPIENTRY *glVertexAttrib4usvARB) (GLuint  index, const GLushort * v);
extern void (GLAPIENTRY *glVertexAttribBinding) (GLuint  attribindex, GLuint  bindingindex);
extern void (GLAPIENTRY *glVertexAttribDivisor) (GLuint  index, GLuint  divisor);
extern void (GLAPIENTRY *glVertexAttribDivisorARB) (GLuint  index, GLuint  divisor);
extern void (GLAPIENTRY *glVertexAttribFormat) (GLuint  attribindex, GLint  size, GLenum  type, GLboolean  normalized, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexAttribFormatNV) (GLuint  index, GLint  size, GLenum  type, GLboolean  normalized, GLsizei  stride);
extern void (GLAPIENTRY *glVertexAttribI1i) (GLuint  index, GLint  x);
extern void (GLAPIENTRY *glVertexAttribI1iEXT) (GLuint  index, GLint  x);
extern void (GLAPIENTRY *glVertexAttribI1iv) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttribI1ivEXT) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttribI1ui) (GLuint  index, GLuint  x);
extern void (GLAPIENTRY *glVertexAttribI1uiEXT) (GLuint  index, GLuint  x);
extern void (GLAPIENTRY *glVertexAttribI1uiv) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttribI1uivEXT) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttribI2i) (GLuint  index, GLint  x, GLint  y);
extern void (GLAPIENTRY *glVertexAttribI2iEXT) (GLuint  index, GLint  x, GLint  y);
extern void (GLAPIENTRY *glVertexAttribI2iv) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttribI2ivEXT) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttribI2ui) (GLuint  index, GLuint  x, GLuint  y);
extern void (GLAPIENTRY *glVertexAttribI2uiEXT) (GLuint  index, GLuint  x, GLuint  y);
extern void (GLAPIENTRY *glVertexAttribI2uiv) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttribI2uivEXT) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttribI3i) (GLuint  index, GLint  x, GLint  y, GLint  z);
extern void (GLAPIENTRY *glVertexAttribI3iEXT) (GLuint  index, GLint  x, GLint  y, GLint  z);
extern void (GLAPIENTRY *glVertexAttribI3iv) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttribI3ivEXT) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttribI3ui) (GLuint  index, GLuint  x, GLuint  y, GLuint  z);
extern void (GLAPIENTRY *glVertexAttribI3uiEXT) (GLuint  index, GLuint  x, GLuint  y, GLuint  z);
extern void (GLAPIENTRY *glVertexAttribI3uiv) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttribI3uivEXT) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttribI4bv) (GLuint  index, const GLbyte * v);
extern void (GLAPIENTRY *glVertexAttribI4bvEXT) (GLuint  index, const GLbyte * v);
extern void (GLAPIENTRY *glVertexAttribI4i) (GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
extern void (GLAPIENTRY *glVertexAttribI4iEXT) (GLuint  index, GLint  x, GLint  y, GLint  z, GLint  w);
extern void (GLAPIENTRY *glVertexAttribI4iv) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttribI4ivEXT) (GLuint  index, const GLint * v);
extern void (GLAPIENTRY *glVertexAttribI4sv) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttribI4svEXT) (GLuint  index, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttribI4ubv) (GLuint  index, const GLubyte * v);
extern void (GLAPIENTRY *glVertexAttribI4ubvEXT) (GLuint  index, const GLubyte * v);
extern void (GLAPIENTRY *glVertexAttribI4ui) (GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
extern void (GLAPIENTRY *glVertexAttribI4uiEXT) (GLuint  index, GLuint  x, GLuint  y, GLuint  z, GLuint  w);
extern void (GLAPIENTRY *glVertexAttribI4uiv) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttribI4uivEXT) (GLuint  index, const GLuint * v);
extern void (GLAPIENTRY *glVertexAttribI4usv) (GLuint  index, const GLushort * v);
extern void (GLAPIENTRY *glVertexAttribI4usvEXT) (GLuint  index, const GLushort * v);
extern void (GLAPIENTRY *glVertexAttribIFormat) (GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexAttribIFormatNV) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glVertexAttribIPointer) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexAttribIPointerEXT) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexAttribL1d) (GLuint  index, GLdouble  x);
extern void (GLAPIENTRY *glVertexAttribL1dEXT) (GLuint  index, GLdouble  x);
extern void (GLAPIENTRY *glVertexAttribL1dv) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribL1dvEXT) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribL1i64NV) (GLuint  index, GLint64EXT  x);
extern void (GLAPIENTRY *glVertexAttribL1i64vNV) (GLuint  index, const GLint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribL1ui64ARB) (GLuint  index, GLuint64EXT  x);
extern void (GLAPIENTRY *glVertexAttribL1ui64NV) (GLuint  index, GLuint64EXT  x);
extern void (GLAPIENTRY *glVertexAttribL1ui64vARB) (GLuint  index, const GLuint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribL1ui64vNV) (GLuint  index, const GLuint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribL2d) (GLuint  index, GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glVertexAttribL2dEXT) (GLuint  index, GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glVertexAttribL2dv) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribL2dvEXT) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribL2i64NV) (GLuint  index, GLint64EXT  x, GLint64EXT  y);
extern void (GLAPIENTRY *glVertexAttribL2i64vNV) (GLuint  index, const GLint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribL2ui64NV) (GLuint  index, GLuint64EXT  x, GLuint64EXT  y);
extern void (GLAPIENTRY *glVertexAttribL2ui64vNV) (GLuint  index, const GLuint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribL3d) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glVertexAttribL3dEXT) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glVertexAttribL3dv) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribL3dvEXT) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribL3i64NV) (GLuint  index, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z);
extern void (GLAPIENTRY *glVertexAttribL3i64vNV) (GLuint  index, const GLint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribL3ui64NV) (GLuint  index, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z);
extern void (GLAPIENTRY *glVertexAttribL3ui64vNV) (GLuint  index, const GLuint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribL4d) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glVertexAttribL4dEXT) (GLuint  index, GLdouble  x, GLdouble  y, GLdouble  z, GLdouble  w);
extern void (GLAPIENTRY *glVertexAttribL4dv) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribL4dvEXT) (GLuint  index, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribL4i64NV) (GLuint  index, GLint64EXT  x, GLint64EXT  y, GLint64EXT  z, GLint64EXT  w);
extern void (GLAPIENTRY *glVertexAttribL4i64vNV) (GLuint  index, const GLint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribL4ui64NV) (GLuint  index, GLuint64EXT  x, GLuint64EXT  y, GLuint64EXT  z, GLuint64EXT  w);
extern void (GLAPIENTRY *glVertexAttribL4ui64vNV) (GLuint  index, const GLuint64EXT * v);
extern void (GLAPIENTRY *glVertexAttribLFormat) (GLuint  attribindex, GLint  size, GLenum  type, GLuint  relativeoffset);
extern void (GLAPIENTRY *glVertexAttribLFormatNV) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glVertexAttribLPointer) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexAttribLPointerEXT) (GLuint  index, GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexAttribP1ui) (GLuint  index, GLenum  type, GLboolean  normalized, GLuint  value);
extern void (GLAPIENTRY *glVertexAttribP1uiv) (GLuint  index, GLenum  type, GLboolean  normalized, const GLuint * value);
extern void (GLAPIENTRY *glVertexAttribP2ui) (GLuint  index, GLenum  type, GLboolean  normalized, GLuint  value);
extern void (GLAPIENTRY *glVertexAttribP2uiv) (GLuint  index, GLenum  type, GLboolean  normalized, const GLuint * value);
extern void (GLAPIENTRY *glVertexAttribP3ui) (GLuint  index, GLenum  type, GLboolean  normalized, GLuint  value);
extern void (GLAPIENTRY *glVertexAttribP3uiv) (GLuint  index, GLenum  type, GLboolean  normalized, const GLuint * value);
extern void (GLAPIENTRY *glVertexAttribP4ui) (GLuint  index, GLenum  type, GLboolean  normalized, GLuint  value);
extern void (GLAPIENTRY *glVertexAttribP4uiv) (GLuint  index, GLenum  type, GLboolean  normalized, const GLuint * value);
extern void (GLAPIENTRY *glVertexAttribParameteriAMD) (GLuint  index, GLenum  pname, GLint  param);
extern void (GLAPIENTRY *glVertexAttribPointer) (GLuint  index, GLint  size, GLenum  type, GLboolean  normalized, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexAttribPointerARB) (GLuint  index, GLint  size, GLenum  type, GLboolean  normalized, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexAttribPointerNV) (GLuint  index, GLint  fsize, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexAttribs1dvNV) (GLuint  index, GLsizei  count, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribs1fvNV) (GLuint  index, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttribs1hvNV) (GLuint  index, GLsizei  n, const GLhalfNV * v);
extern void (GLAPIENTRY *glVertexAttribs1svNV) (GLuint  index, GLsizei  count, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttribs2dvNV) (GLuint  index, GLsizei  count, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribs2fvNV) (GLuint  index, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttribs2hvNV) (GLuint  index, GLsizei  n, const GLhalfNV * v);
extern void (GLAPIENTRY *glVertexAttribs2svNV) (GLuint  index, GLsizei  count, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttribs3dvNV) (GLuint  index, GLsizei  count, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribs3fvNV) (GLuint  index, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttribs3hvNV) (GLuint  index, GLsizei  n, const GLhalfNV * v);
extern void (GLAPIENTRY *glVertexAttribs3svNV) (GLuint  index, GLsizei  count, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttribs4dvNV) (GLuint  index, GLsizei  count, const GLdouble * v);
extern void (GLAPIENTRY *glVertexAttribs4fvNV) (GLuint  index, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glVertexAttribs4hvNV) (GLuint  index, GLsizei  n, const GLhalfNV * v);
extern void (GLAPIENTRY *glVertexAttribs4svNV) (GLuint  index, GLsizei  count, const GLshort * v);
extern void (GLAPIENTRY *glVertexAttribs4ubvNV) (GLuint  index, GLsizei  count, const GLubyte * v);
extern void (GLAPIENTRY *glVertexBindingDivisor) (GLuint  bindingindex, GLuint  divisor);
extern void (GLAPIENTRY *glVertexBlendARB) (GLint  count);
extern void (GLAPIENTRY *glVertexFormatNV) (GLint  size, GLenum  type, GLsizei  stride);
extern void (GLAPIENTRY *glVertexP2ui) (GLenum  type, GLuint  value);
extern void (GLAPIENTRY *glVertexP2uiv) (GLenum  type, const GLuint * value);
extern void (GLAPIENTRY *glVertexP3ui) (GLenum  type, GLuint  value);
extern void (GLAPIENTRY *glVertexP3uiv) (GLenum  type, const GLuint * value);
extern void (GLAPIENTRY *glVertexP4ui) (GLenum  type, GLuint  value);
extern void (GLAPIENTRY *glVertexP4uiv) (GLenum  type, const GLuint * value);
extern void (GLAPIENTRY *glVertexPointer) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexPointerEXT) (GLint  size, GLenum  type, GLsizei  stride, GLsizei  count, const void * pointer);
extern void (GLAPIENTRY *glVertexPointervINTEL) (GLint  size, GLenum  type, const void ** pointer);
extern void (GLAPIENTRY *glVertexWeightPointerEXT) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glVertexWeightfEXT) (GLfloat  weight);
extern void (GLAPIENTRY *glVertexWeightfvEXT) (const GLfloat * weight);
extern void (GLAPIENTRY *glVertexWeighthNV) (GLhalfNV  weight);
extern void (GLAPIENTRY *glVertexWeighthvNV) (const GLhalfNV * weight);
extern GLenum (GLAPIENTRY *glVideoCaptureNV) (GLuint  video_capture_slot, GLuint * sequence_num, GLuint64EXT * capture_time);
extern void (GLAPIENTRY *glVideoCaptureStreamParameterdvNV) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, const GLdouble * params);
extern void (GLAPIENTRY *glVideoCaptureStreamParameterfvNV) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, const GLfloat * params);
extern void (GLAPIENTRY *glVideoCaptureStreamParameterivNV) (GLuint  video_capture_slot, GLuint  stream, GLenum  pname, const GLint * params);
extern void (GLAPIENTRY *glViewport) (GLint  x, GLint  y, GLsizei  width, GLsizei  height);
extern void (GLAPIENTRY *glViewportArrayv) (GLuint  first, GLsizei  count, const GLfloat * v);
extern void (GLAPIENTRY *glViewportIndexedf) (GLuint  index, GLfloat  x, GLfloat  y, GLfloat  w, GLfloat  h);
extern void (GLAPIENTRY *glViewportIndexedfv) (GLuint  index, const GLfloat * v);
extern void (GLAPIENTRY *glViewportPositionWScaleNV) (GLuint  index, GLfloat  xcoeff, GLfloat  ycoeff);
extern void (GLAPIENTRY *glViewportSwizzleNV) (GLuint  index, GLenum  swizzlex, GLenum  swizzley, GLenum  swizzlez, GLenum  swizzlew);
extern void (GLAPIENTRY *glWaitSemaphoreEXT) (GLuint  semaphore, GLuint  numBufferBarriers, const GLuint * buffers, GLuint  numTextureBarriers, const GLuint * textures, const GLenum * srcLayouts);
extern void (GLAPIENTRY *glWaitSync) (GLsync  sync, GLbitfield  flags, GLuint64  timeout);
extern void (GLAPIENTRY *glWeightPathsNV) (GLuint  resultPath, GLsizei  numPaths, const GLuint * paths, const GLfloat * weights);
extern void (GLAPIENTRY *glWeightPointerARB) (GLint  size, GLenum  type, GLsizei  stride, const void * pointer);
extern void (GLAPIENTRY *glWeightbvARB) (GLint  size, const GLbyte * weights);
extern void (GLAPIENTRY *glWeightdvARB) (GLint  size, const GLdouble * weights);
extern void (GLAPIENTRY *glWeightfvARB) (GLint  size, const GLfloat * weights);
extern void (GLAPIENTRY *glWeightivARB) (GLint  size, const GLint * weights);
extern void (GLAPIENTRY *glWeightsvARB) (GLint  size, const GLshort * weights);
extern void (GLAPIENTRY *glWeightubvARB) (GLint  size, const GLubyte * weights);
extern void (GLAPIENTRY *glWeightuivARB) (GLint  size, const GLuint * weights);
extern void (GLAPIENTRY *glWeightusvARB) (GLint  size, const GLushort * weights);
extern void (GLAPIENTRY *glWindowPos2d) (GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glWindowPos2dARB) (GLdouble  x, GLdouble  y);
extern void (GLAPIENTRY *glWindowPos2dv) (const GLdouble * v);
extern void (GLAPIENTRY *glWindowPos2dvARB) (const GLdouble * v);
extern void (GLAPIENTRY *glWindowPos2f) (GLfloat  x, GLfloat  y);
extern void (GLAPIENTRY *glWindowPos2fARB) (GLfloat  x, GLfloat  y);
extern void (GLAPIENTRY *glWindowPos2fv) (const GLfloat * v);
extern void (GLAPIENTRY *glWindowPos2fvARB) (const GLfloat * v);
extern void (GLAPIENTRY *glWindowPos2i) (GLint  x, GLint  y);
extern void (GLAPIENTRY *glWindowPos2iARB) (GLint  x, GLint  y);
extern void (GLAPIENTRY *glWindowPos2iv) (const GLint * v);
extern void (GLAPIENTRY *glWindowPos2ivARB) (const GLint * v);
extern void (GLAPIENTRY *glWindowPos2s) (GLshort  x, GLshort  y);
extern void (GLAPIENTRY *glWindowPos2sARB) (GLshort  x, GLshort  y);
extern void (GLAPIENTRY *glWindowPos2sv) (const GLshort * v);
extern void (GLAPIENTRY *glWindowPos2svARB) (const GLshort * v);
extern void (GLAPIENTRY *glWindowPos3d) (GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glWindowPos3dARB) (GLdouble  x, GLdouble  y, GLdouble  z);
extern void (GLAPIENTRY *glWindowPos3dv) (const GLdouble * v);
extern void (GLAPIENTRY *glWindowPos3dvARB) (const GLdouble * v);
extern void (GLAPIENTRY *glWindowPos3f) (GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glWindowPos3fARB) (GLfloat  x, GLfloat  y, GLfloat  z);
extern void (GLAPIENTRY *glWindowPos3fv) (const GLfloat * v);
extern void (GLAPIENTRY *glWindowPos3fvARB) (const GLfloat * v);
extern void (GLAPIENTRY *glWindowPos3i) (GLint  x, GLint  y, GLint  z);
extern void (GLAPIENTRY *glWindowPos3iARB) (GLint  x, GLint  y, GLint  z);
extern void (GLAPIENTRY *glWindowPos3iv) (const GLint * v);
extern void (GLAPIENTRY *glWindowPos3ivARB) (const GLint * v);
extern void (GLAPIENTRY *glWindowPos3s) (GLshort  x, GLshort  y, GLshort  z);
extern void (GLAPIENTRY *glWindowPos3sARB) (GLshort  x, GLshort  y, GLshort  z);
extern void (GLAPIENTRY *glWindowPos3sv) (const GLshort * v);
extern void (GLAPIENTRY *glWindowPos3svARB) (const GLshort * v);
extern void (GLAPIENTRY *glWindowRectanglesEXT) (GLenum  mode, GLsizei  count, const GLint * box);
extern void (GLAPIENTRY *glWriteMaskEXT) (GLuint  res, GLuint  in, GLenum  outX, GLenum  outY, GLenum  outZ, GLenum  outW);
extern void (GLAPIENTRY *glDrawVkImageNV) (GLuint64  vkImage, GLuint  sampler, GLfloat  x0, GLfloat  y0, GLfloat  x1, GLfloat  y1, GLfloat  z, GLfloat  s0, GLfloat  t0, GLfloat  s1, GLfloat  t1);
extern GLVULKANPROCNV (GLAPIENTRY *glGetVkProcAddrNV) (const GLchar * name);
extern void (GLAPIENTRY *glWaitVkSemaphoreNV) (GLuint64  vkSemaphore);
extern void (GLAPIENTRY *glSignalVkSemaphoreNV) (GLuint64  vkSemaphore);
extern void (GLAPIENTRY *glSignalVkFenceNV) (GLuint64  vkFence);


bool glApiLoad();
void glApiUnload();


}  // namespace GLApi
}  // namespace internal
}  // namespace OpenSubdiv

#undef GLAPIENTRY

#endif
