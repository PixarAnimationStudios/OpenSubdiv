//
//   Copyright 2013 Pixar
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

#include "../../regression/common/shape_utils.h"
#include "../../regression/shapes/all.h"


static std::vector<ShapeDesc> g_defaultShapes;

//------------------------------------------------------------------------------
static void initShapes() {

    //
    //  Note that any shapes added here must have UVs -- loading a shape without UVs is a fatal
    //  error and will result in termination when it is selected.
    //
    g_defaultShapes.push_back(ShapeDesc("catmark_toroidal_tet",     catmark_toroidal_tet,     kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_cube",             catmark_cube,             kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_cubes_semisharp",  catmark_cubes_semisharp,  kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_cubes_infsharp",   catmark_cubes_infsharp,   kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_pyramid",          catmark_pyramid,          kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_torus",            catmark_torus,            kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_torus_creases0",   catmark_torus_creases0,   kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_fvar_bound0",      catmark_fvar_bound0,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_fvar_bound1",      catmark_fvar_bound1,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_fvar_bound2",      catmark_fvar_bound2,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_fvar_bound3",      catmark_fvar_bound3,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_fvar_bound4",      catmark_fvar_bound4,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_fvar_project0",    catmark_fvar_project0,    kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_edgecorner",       catmark_edgecorner,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_edgeonly",         catmark_edgeonly,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_edgenone",         catmark_edgenone,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_quadstrips",       catmark_quadstrips,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_xord_interior",    catmark_xord_interior,    kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_xord_boundary",    catmark_xord_boundary,    kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_nonquads",         catmark_nonquads,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_nonman_verts",     catmark_nonman_verts,     kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_nonman_edges",     catmark_nonman_edges,     kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_hole_test1",       catmark_hole_test1,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_hole_test2",       catmark_hole_test2,       kCatmark));

    g_defaultShapes.push_back(ShapeDesc("loop_toroidal_tet",        loop_toroidal_tet,        kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_tetrahedron",         loop_tetrahedron,         kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_cube",                loop_cube,                kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_cubes_semisharp",     loop_cubes_semisharp,     kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_cubes_infsharp",      loop_cubes_infsharp,      kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_cube_asymmetric",     loop_cube_asymmetric,     kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_icosahedron",         loop_icosahedron,         kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_icos_semisharp",      loop_icos_semisharp,      kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_icos_infsharp",       loop_icos_infsharp,       kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_fvar_bound0",         loop_fvar_bound0,         kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_fvar_bound1",         loop_fvar_bound1,         kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_fvar_bound2",         loop_fvar_bound2,         kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_fvar_bound3",         loop_fvar_bound3,         kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_triangle_edgecorner", loop_triangle_edgecorner, kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_triangle_edgeonly",   loop_triangle_edgeonly,   kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_triangle_edgenone",   loop_triangle_edgenone,   kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_xord_interior",       loop_xord_interior,       kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_xord_boundary",       loop_xord_boundary,       kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_nonman_verts",        loop_nonman_verts,        kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_nonman_edges",        loop_nonman_edges,        kLoop));

    g_defaultShapes.push_back(ShapeDesc("bilinear_cube",            bilinear_cube,            kBilinear));
    g_defaultShapes.push_back(ShapeDesc("bilinear_nonplanar",       bilinear_nonplanar,       kBilinear));
    g_defaultShapes.push_back(ShapeDesc("bilinear_nonquads0",       bilinear_nonquads0,       kBilinear));
    g_defaultShapes.push_back(ShapeDesc("bilinear_nonquads1",       bilinear_nonquads1,       kBilinear));
}
//------------------------------------------------------------------------------
