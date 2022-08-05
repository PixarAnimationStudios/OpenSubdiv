//
//   Copyright 2021 Pixar
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

#include <regression/common/shape_utils.h>
#include <regression/shapes/all.h>

static void
initShapesAll(std::vector<ShapeDesc> & shapes) {

    shapes.push_back(ShapeDesc("catmark_cube", catmark_cube, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube_corner0", catmark_cube_corner0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube_corner1", catmark_cube_corner1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube_corner2", catmark_cube_corner2, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube_corner3", catmark_cube_corner3, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube_corner4", catmark_cube_corner4, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube_creases0", catmark_cube_creases0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube_creases1", catmark_cube_creases1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube_creases2", catmark_cube_creases2, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cubes_infsharp", catmark_cubes_infsharp, kCatmark));
    shapes.push_back(ShapeDesc("catmark_cubes_semisharp", catmark_cubes_semisharp, kCatmark));
    shapes.push_back(ShapeDesc("catmark_chaikin0", catmark_chaikin0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_chaikin1", catmark_chaikin1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_chaikin2", catmark_chaikin2, kCatmark));
    shapes.push_back(ShapeDesc("catmark_dart_edgecorner", catmark_dart_edgecorner, kCatmark));
    shapes.push_back(ShapeDesc("catmark_dart_edgeonly", catmark_dart_edgeonly, kCatmark));
    shapes.push_back(ShapeDesc("catmark_edgecorner", catmark_edgecorner, kCatmark));
    shapes.push_back(ShapeDesc("catmark_edgeonly", catmark_edgeonly, kCatmark));
    shapes.push_back(ShapeDesc("catmark_edgenone", catmark_edgenone, kCatmark));
    shapes.push_back(ShapeDesc("catmark_fan", catmark_fan, kCatmark));
    shapes.push_back(ShapeDesc("catmark_flap", catmark_flap, kCatmark));
    shapes.push_back(ShapeDesc("catmark_flap2", catmark_flap2, kCatmark));
    shapes.push_back(ShapeDesc("catmark_fvar_bound0", catmark_fvar_bound0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_fvar_bound1", catmark_fvar_bound1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_fvar_bound2", catmark_fvar_bound2, kCatmark));
    shapes.push_back(ShapeDesc("catmark_fvar_bound3", catmark_fvar_bound3, kCatmark));
    shapes.push_back(ShapeDesc("catmark_fvar_bound4", catmark_fvar_bound4, kCatmark));
    shapes.push_back(ShapeDesc("catmark_fvar_project0", catmark_fvar_project0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test0", catmark_gregory_test0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test1", catmark_gregory_test1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test2", catmark_gregory_test2, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test3", catmark_gregory_test3, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test4", catmark_gregory_test4, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test5", catmark_gregory_test5, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test6", catmark_gregory_test6, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test7", catmark_gregory_test7, kCatmark));
    shapes.push_back(ShapeDesc("catmark_gregory_test8", catmark_gregory_test8, kCatmark));
    shapes.push_back(ShapeDesc("catmark_helmet", catmark_helmet, kCatmark));
    shapes.push_back(ShapeDesc("catmark_hole_test1", catmark_hole_test1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_hole_test2", catmark_hole_test2, kCatmark));
    shapes.push_back(ShapeDesc("catmark_hole_test3", catmark_hole_test3, kCatmark));
    shapes.push_back(ShapeDesc("catmark_hole_test4", catmark_hole_test4, kCatmark));
    shapes.push_back(ShapeDesc("catmark_lefthanded", catmark_lefthanded, kCatmark));
    shapes.push_back(ShapeDesc("catmark_righthanded", catmark_righthanded, kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_edges", catmark_nonman_edges, kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_edge100", catmark_nonman_edge100, kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_verts", catmark_nonman_verts, kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_quadpole8", catmark_nonman_quadpole8, kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_quadpole64", catmark_nonman_quadpole64, kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_quadpole360", catmark_nonman_quadpole360, kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_bareverts", catmark_nonman_bareverts, kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonquads", catmark_nonquads, kCatmark));
    shapes.push_back(ShapeDesc("catmark_pawn", catmark_pawn, kCatmark));
    shapes.push_back(ShapeDesc("catmark_pole8", catmark_pole8, kCatmark));
    shapes.push_back(ShapeDesc("catmark_pole64", catmark_pole64, kCatmark));
    shapes.push_back(ShapeDesc("catmark_pyramid_creases0", catmark_pyramid_creases0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_pyramid_creases1", catmark_pyramid_creases1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_pyramid_creases2", catmark_pyramid_creases2, kCatmark));
    shapes.push_back(ShapeDesc("catmark_pyramid", catmark_pyramid, kCatmark));
    shapes.push_back(ShapeDesc("catmark_quadstrips", catmark_quadstrips, kCatmark));
    shapes.push_back(ShapeDesc("catmark_single_crease", catmark_single_crease, kCatmark));
    shapes.push_back(ShapeDesc("catmark_inf_crease0", catmark_inf_crease0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_inf_crease1", catmark_inf_crease1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_smoothtris0", catmark_smoothtris0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_smoothtris1", catmark_smoothtris1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_tent_creases0", catmark_tent_creases0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_tent_creases1", catmark_tent_creases1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_tent", catmark_tent, kCatmark));
    shapes.push_back(ShapeDesc("catmark_toroidal_tet", catmark_toroidal_tet, kCatmark));
    shapes.push_back(ShapeDesc("catmark_torus", catmark_torus, kCatmark));
    shapes.push_back(ShapeDesc("catmark_torus_creases0", catmark_torus_creases0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_torus_creases1", catmark_torus_creases1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_val2_interior", catmark_val2_interior, kCatmark));
    shapes.push_back(ShapeDesc("catmark_val2_back2back", catmark_val2_back2back, kCatmark));
    shapes.push_back(ShapeDesc("catmark_val2_foldover", catmark_val2_foldover, kCatmark));
    shapes.push_back(ShapeDesc("catmark_val2_nonman", catmark_val2_nonman, kCatmark));
    shapes.push_back(ShapeDesc("catmark_xord_interior", catmark_xord_interior, kCatmark));
    shapes.push_back(ShapeDesc("catmark_xord_boundary", catmark_xord_boundary, kCatmark));
    shapes.push_back(ShapeDesc("bilinear_cube", bilinear_cube, kBilinear));
    shapes.push_back(ShapeDesc("bilinear_nonplanar", bilinear_nonplanar, kBilinear));
    shapes.push_back(ShapeDesc("bilinear_nonquads0", bilinear_nonquads0, kBilinear));
    shapes.push_back(ShapeDesc("bilinear_nonquads1", bilinear_nonquads1, kBilinear));
    shapes.push_back(ShapeDesc("loop_chaikin0", loop_chaikin0, kLoop));
    shapes.push_back(ShapeDesc("loop_chaikin1", loop_chaikin1, kLoop));
    shapes.push_back(ShapeDesc("loop_cube", loop_cube, kLoop));
    shapes.push_back(ShapeDesc("loop_cube_asymmetric", loop_cube_asymmetric, kLoop));
    shapes.push_back(ShapeDesc("loop_cube_creases0", loop_cube_creases0, kLoop));
    shapes.push_back(ShapeDesc("loop_cube_creases1", loop_cube_creases1, kLoop));
    shapes.push_back(ShapeDesc("loop_cubes_infsharp", loop_cubes_infsharp, kLoop));
    shapes.push_back(ShapeDesc("loop_cubes_semisharp", loop_cubes_semisharp, kLoop));
    shapes.push_back(ShapeDesc("loop_fvar_bound0", loop_fvar_bound0, kLoop));
    shapes.push_back(ShapeDesc("loop_fvar_bound1", loop_fvar_bound1, kLoop));
    shapes.push_back(ShapeDesc("loop_fvar_bound2", loop_fvar_bound2, kLoop));
    shapes.push_back(ShapeDesc("loop_fvar_bound3", loop_fvar_bound3, kLoop));
    shapes.push_back(ShapeDesc("loop_icosahedron", loop_icosahedron, kLoop));
    shapes.push_back(ShapeDesc("loop_icos_infsharp", loop_icos_infsharp, kLoop));
    shapes.push_back(ShapeDesc("loop_icos_semisharp", loop_icos_semisharp, kLoop));
    shapes.push_back(ShapeDesc("loop_nonman_edges", loop_nonman_edges, kLoop));
    shapes.push_back(ShapeDesc("loop_nonman_edge100", loop_nonman_edge100, kLoop));
    shapes.push_back(ShapeDesc("loop_nonman_verts", loop_nonman_verts, kLoop));
    shapes.push_back(ShapeDesc("loop_pole8", loop_pole8, kLoop));
    shapes.push_back(ShapeDesc("loop_pole64", loop_pole64, kLoop));
    shapes.push_back(ShapeDesc("loop_saddle_edgecorner", loop_saddle_edgecorner, kLoop));
    shapes.push_back(ShapeDesc("loop_saddle_edgeonly", loop_saddle_edgeonly, kLoop));
    shapes.push_back(ShapeDesc("loop_tetrahedron", loop_tetrahedron, kLoop));
    shapes.push_back(ShapeDesc("loop_toroidal_tet", loop_toroidal_tet, kLoop));
    shapes.push_back(ShapeDesc("loop_triangle_edgecorner", loop_triangle_edgecorner, kLoop));
    shapes.push_back(ShapeDesc("loop_triangle_edgeonly", loop_triangle_edgeonly, kLoop));
    shapes.push_back(ShapeDesc("loop_xord_boundary", loop_xord_boundary, kLoop));
    shapes.push_back(ShapeDesc("loop_xord_interior", loop_xord_interior, kLoop));
    shapes.push_back(ShapeDesc("loop_val2_interior", loop_val2_interior, kLoop));

    //  More complicated shapes with longer execution times:
    shapes.push_back(ShapeDesc("catmark_car", catmark_car, kCatmark));
    shapes.push_back(ShapeDesc("catmark_rook", catmark_rook, kCatmark));
    shapes.push_back(ShapeDesc("catmark_bishop", catmark_bishop, kCatmark));
//  shapes.push_back(ShapeDesc("catmark_pole360", catmark_pole360, kCatmark));
//  shapes.push_back(ShapeDesc("loop_pole360", loop_pole360, kLoop));
}
