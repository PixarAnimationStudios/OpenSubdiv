//
//   Copyright 2015 Pixar
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
#ifndef OPENSUBDIV_REGRESSION_SHAPES_ALL_H
#define OPENSUBDIV_REGRESSION_SHAPES_ALL_H

#include "catmark_bishop.h"
#include "catmark_car.h"
#include "catmark_chaikin0.h"
#include "catmark_chaikin1.h"
#include "catmark_chaikin2.h"
#include "catmark_cube_corner0.h"
#include "catmark_cube_corner1.h"
#include "catmark_cube_corner2.h"
#include "catmark_cube_corner3.h"
#include "catmark_cube_corner4.h"
#include "catmark_cube_creases0.h"
#include "catmark_cube_creases1.h"
#include "catmark_cube_creases2.h"
#include "catmark_cube.h"
#include "catmark_cubes_infsharp.h"
#include "catmark_cubes_semisharp.h"
#include "catmark_dart_edgecorner.h"
#include "catmark_dart_edgeonly.h"
#include "catmark_edgecorner.h"
#include "catmark_edgenone.h"
#include "catmark_edgeonly.h"
#include "catmark_fan.h"
#include "catmark_flap.h"
#include "catmark_flap2.h"
#include "catmark_fvar_bound0.h"
#include "catmark_fvar_bound1.h"
#include "catmark_fvar_bound2.h"
#include "catmark_fvar_bound3.h"
#include "catmark_fvar_bound4.h"
#include "catmark_fvar_project0.h"
#include "catmark_gregory_test0.h"
#include "catmark_gregory_test1.h"
#include "catmark_gregory_test2.h"
#include "catmark_gregory_test3.h"
#include "catmark_gregory_test4.h"
#include "catmark_gregory_test5.h"
#include "catmark_gregory_test6.h"
#include "catmark_gregory_test7.h"
#include "catmark_gregory_test8.h"
#include "catmark_helmet.h"
#include "catmark_hole_test1.h"
#include "catmark_hole_test2.h"
#include "catmark_hole_test3.h"
#include "catmark_hole_test4.h"
#include "catmark_lefthanded.h"
#include "catmark_righthanded.h"
#include "catmark_pole8.h"
#include "catmark_pole64.h"
#include "catmark_pole360.h"
#include "catmark_nonman_edges.h"
#include "catmark_nonman_edge100.h"
#include "catmark_nonman_verts.h"
#include "catmark_nonman_quadpole8.h"
#include "catmark_nonman_quadpole64.h"
#include "catmark_nonman_quadpole360.h"
#include "catmark_nonman_bareverts.h"
#include "catmark_nonquads.h"
#include "catmark_pawn.h"
#include "catmark_pyramid_creases0.h"
#include "catmark_pyramid_creases1.h"
#include "catmark_pyramid_creases2.h"
#include "catmark_pyramid.h"
#include "catmark_quadstrips.h"
#include "catmark_rook.h"
#include "catmark_single_crease.h"
#include "catmark_inf_crease0.h"
#include "catmark_inf_crease1.h"
#include "catmark_smoothtris0.h"
#include "catmark_smoothtris1.h"
#include "catmark_square_hedit0.h"
#include "catmark_square_hedit1.h"
#include "catmark_square_hedit2.h"
#include "catmark_square_hedit3.h"
#include "catmark_square_hedit4.h"
#include "catmark_tent_creases0.h"
#include "catmark_tent_creases1.h"
#include "catmark_tent.h"
#include "catmark_toroidal_tet.h"
#include "catmark_torus.h"
#include "catmark_torus_creases0.h"
#include "catmark_torus_creases1.h"
#include "catmark_val2_interior.h"
#include "catmark_val2_back2back.h"
#include "catmark_val2_foldover.h"
#include "catmark_val2_nonman.h"
#include "catmark_xord_interior.h"
#include "catmark_xord_boundary.h"

#include "bilinear_cube.h"
#include "bilinear_nonplanar.h"
#include "bilinear_nonquads0.h"
#include "bilinear_nonquads1.h"

#include "loop_chaikin0.h"
#include "loop_chaikin1.h"
#include "loop_cube.h"
#include "loop_cube_asymmetric.h"
#include "loop_cube_creases0.h"
#include "loop_cube_creases1.h"
#include "loop_cubes_infsharp.h"
#include "loop_cubes_semisharp.h"
#include "loop_fvar_bound0.h"
#include "loop_fvar_bound1.h"
#include "loop_fvar_bound2.h"
#include "loop_fvar_bound3.h"
#include "loop_icosahedron.h"
#include "loop_icos_infsharp.h"
#include "loop_icos_semisharp.h"
#include "loop_nonman_edges.h"
#include "loop_nonman_edge100.h"
#include "loop_nonman_verts.h"
#include "loop_pole8.h"
#include "loop_pole64.h"
#include "loop_pole360.h"
#include "loop_saddle_edgecorner.h"
#include "loop_saddle_edgeonly.h"
#include "loop_tetrahedron.h"
#include "loop_toroidal_tet.h"
#include "loop_triangle_edgecorner.h"
#include "loop_triangle_edgenone.h"
#include "loop_triangle_edgeonly.h"
#include "loop_xord_boundary.h"
#include "loop_xord_interior.h"
#include "loop_val2_interior.h"

#endif   // OPENSUBDIV_REGRESSION_SHAPES_ALL_H
