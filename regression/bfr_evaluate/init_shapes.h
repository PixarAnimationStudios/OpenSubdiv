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


//------------------------------------------------------------------------------
static void
initShapes(std::vector<ShapeDesc> & shapes) {

    shapes.push_back(ShapeDesc("catmark_toroidal_tet",     catmark_toroidal_tet,     kCatmark));
    shapes.push_back(ShapeDesc("catmark_cube",             catmark_cube,             kCatmark));
    shapes.push_back(ShapeDesc("catmark_cubes_semisharp",  catmark_cubes_semisharp,  kCatmark));
    shapes.push_back(ShapeDesc("catmark_cubes_infsharp",   catmark_cubes_infsharp,   kCatmark));
    shapes.push_back(ShapeDesc("catmark_pyramid",          catmark_pyramid,          kCatmark));
    shapes.push_back(ShapeDesc("catmark_pyramid_creases0", catmark_pyramid_creases0, kCatmark));
    shapes.push_back(ShapeDesc("catmark_pyramid_creases1", catmark_pyramid_creases1, kCatmark));
    shapes.push_back(ShapeDesc("catmark_torus",            catmark_torus,            kCatmark));
    shapes.push_back(ShapeDesc("catmark_torus_creases0",   catmark_torus_creases0,   kCatmark));
    shapes.push_back(ShapeDesc("catmark_torus_creases1",   catmark_torus_creases1,   kCatmark));
    shapes.push_back(ShapeDesc("catmark_edgecorner",       catmark_edgecorner,       kCatmark));
    shapes.push_back(ShapeDesc("catmark_edgeonly",         catmark_edgeonly,         kCatmark));
    shapes.push_back(ShapeDesc("catmark_edgenone",         catmark_edgenone,         kCatmark));
    shapes.push_back(ShapeDesc("catmark_quadstrips",       catmark_quadstrips,       kCatmark));
    shapes.push_back(ShapeDesc("catmark_xord_interior",    catmark_xord_interior,    kCatmark));
    shapes.push_back(ShapeDesc("catmark_xord_boundary",    catmark_xord_boundary,    kCatmark));
    shapes.push_back(ShapeDesc("catmark_val2_interior",    catmark_val2_interior,    kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonquads",         catmark_nonquads,         kCatmark));
    shapes.push_back(ShapeDesc("catmark_single_crease",    catmark_single_crease,    kCatmark));
    shapes.push_back(ShapeDesc("catmark_inf_crease0",      catmark_inf_crease0,      kCatmark));
    shapes.push_back(ShapeDesc("catmark_inf_crease1",      catmark_inf_crease1,      kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_verts",     catmark_nonman_verts,     kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_edges",     catmark_nonman_edges,     kCatmark));
    shapes.push_back(ShapeDesc("catmark_hole_test1",       catmark_hole_test1,       kCatmark));
    shapes.push_back(ShapeDesc("catmark_hole_test2",       catmark_hole_test2,       kCatmark));
    shapes.push_back(ShapeDesc("catmark_hole_test3",       catmark_hole_test3,       kCatmark));
    shapes.push_back(ShapeDesc("catmark_hole_test4",       catmark_hole_test4,       kCatmark));
    shapes.push_back(ShapeDesc("catmark_chaikin0",         catmark_chaikin0,         kCatmark));
    shapes.push_back(ShapeDesc("catmark_chaikin2",         catmark_chaikin2,         kCatmark));
    shapes.push_back(ShapeDesc("catmark_smoothtris0",      catmark_smoothtris0,      kCatmark));
    shapes.push_back(ShapeDesc("catmark_smoothtris1",      catmark_smoothtris1,      kCatmark));
    shapes.push_back(ShapeDesc("catmark_pole8",            catmark_pole8,            kCatmark));
//  shapes.push_back(ShapeDesc("catmark_pole64",           catmark_pole64,           kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_quadpole64",catmark_nonman_quadpole64,kCatmark));
    shapes.push_back(ShapeDesc("catmark_nonman_edge100",   catmark_nonman_edge100,   kCatmark));
    shapes.push_back(ShapeDesc("catmark_helmet",           catmark_helmet,           kCatmark));
    shapes.push_back(ShapeDesc("catmark_bishop",           catmark_bishop,           kCatmark));
    shapes.push_back(ShapeDesc("catmark_pawn",             catmark_pawn,             kCatmark));
//  shapes.push_back(ShapeDesc("catmark_rook",             catmark_rook,             kCatmark));
    shapes.push_back(ShapeDesc("catmark_car",              catmark_car,              kCatmark));

    shapes.push_back(ShapeDesc("loop_toroidal_tet",        loop_toroidal_tet,        kLoop));
    shapes.push_back(ShapeDesc("loop_tetrahedron",         loop_tetrahedron,         kLoop));
    shapes.push_back(ShapeDesc("loop_cube",                loop_cube,                kLoop));
    shapes.push_back(ShapeDesc("loop_cubes_semisharp",     loop_cubes_semisharp,     kLoop));
    shapes.push_back(ShapeDesc("loop_cubes_infsharp",      loop_cubes_infsharp,      kLoop));
    shapes.push_back(ShapeDesc("loop_cube_asymmetric",     loop_cube_asymmetric,     kLoop));
    shapes.push_back(ShapeDesc("loop_icosahedron",         loop_icosahedron,         kLoop));
    shapes.push_back(ShapeDesc("loop_icos_semisharp",      loop_icos_semisharp,      kLoop));
    shapes.push_back(ShapeDesc("loop_icos_infsharp",       loop_icos_infsharp,       kLoop));
    shapes.push_back(ShapeDesc("loop_triangle_edgecorner", loop_triangle_edgecorner, kLoop));
    shapes.push_back(ShapeDesc("loop_triangle_edgeonly",   loop_triangle_edgeonly,   kLoop));
    shapes.push_back(ShapeDesc("loop_triangle_edgenone",   loop_triangle_edgenone,   kLoop));
    shapes.push_back(ShapeDesc("loop_xord_interior",       loop_xord_interior,       kLoop));
    shapes.push_back(ShapeDesc("loop_xord_boundary",       loop_xord_boundary,       kLoop));
    shapes.push_back(ShapeDesc("loop_nonman_verts",        loop_nonman_verts,        kLoop));
    shapes.push_back(ShapeDesc("loop_nonman_edges",        loop_nonman_edges,        kLoop));
//  shapes.push_back(ShapeDesc("loop_pole64",              loop_pole64,              kLoop));
    shapes.push_back(ShapeDesc("loop_nonman_edge100",      loop_nonman_edge100,      kLoop));

    shapes.push_back(ShapeDesc("bilinear_cube",            bilinear_cube,            kBilinear));
    shapes.push_back(ShapeDesc("bilinear_nonplanar",       bilinear_nonplanar,       kBilinear));
    shapes.push_back(ShapeDesc("bilinear_nonquads0",       bilinear_nonquads0,       kBilinear));
    shapes.push_back(ShapeDesc("bilinear_nonquads1",       bilinear_nonquads1,       kBilinear));
}
//------------------------------------------------------------------------------
