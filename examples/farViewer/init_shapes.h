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


static std::vector<ShapeDesc> g_shapes;

//------------------------------------------------------------------------------
static void initShapes() {
//    g_shapes.push_back( ShapeDesc("bilinear_cube",            bilinear_cube,            kBilinear) );

    g_shapes.push_back( ShapeDesc("catmark_cube_corner0",     catmark_cube_corner0,     kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_cube_corner1",     catmark_cube_corner1,     kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_cube_corner2",     catmark_cube_corner2,     kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_cube_corner3",     catmark_cube_corner3,     kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_cube_corner4",     catmark_cube_corner4,     kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_cube_creases0",    catmark_cube_creases0,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_cube_creases1",    catmark_cube_creases1,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_cube",             catmark_cube,             kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_dart_edgecorner",  catmark_dart_edgecorner,  kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_dart_edgeonly",    catmark_dart_edgeonly,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_chaikin0",         catmark_chaikin0,         kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_chaikin1",         catmark_chaikin1,         kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_chaikin2",         catmark_chaikin2,         kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_edgecorner",       catmark_edgecorner,       kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_edgeonly",         catmark_edgeonly,         kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_edgenone",         catmark_edgenone,         kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_fan",              catmark_fan,              kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_flap",             catmark_flap,             kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_flap2",            catmark_flap2,            kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_fvar_bound0",      catmark_fvar_bound0,      kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_fvar_bound1",      catmark_fvar_bound1,      kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_fvar_bound2",      catmark_fvar_bound2,      kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_gregory_test0",    catmark_gregory_test0,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_gregory_test1",    catmark_gregory_test1,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_gregory_test2",    catmark_gregory_test2,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_gregory_test3",    catmark_gregory_test3,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_gregory_test4",    catmark_gregory_test4,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_helmet",           catmark_helmet,           kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_pyramid_creases0", catmark_pyramid_creases0, kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_pyramid_creases1", catmark_pyramid_creases1, kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_pyramid",          catmark_pyramid,          kCatmark ) );
//    g_shapes.push_back( ShapeDesc("catmark_square_hedit0",    catmark_square_hedit0,    kCatmark ) );
//    g_shapes.push_back( ShapeDesc("catmark_square_hedit1",    catmark_square_hedit1,    kCatmark ) );
//    g_shapes.push_back( ShapeDesc("catmark_square_hedit2",    catmark_square_hedit2,    kCatmark ) );
//    g_shapes.push_back( ShapeDesc("catmark_square_hedit3",    catmark_square_hedit3,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_tent_creases0",    catmark_tent_creases0,    kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_tent_creases1",    catmark_tent_creases1 ,   kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_tent",             catmark_tent,             kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_tent",             catmark_tent,             kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_torus",            catmark_torus,            kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_torus_creases0",   catmark_torus_creases0,   kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_smoothtris0",      catmark_smoothtris0,      kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_smoothtris1",      catmark_smoothtris1,      kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_car",              catmark_car,              kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_bishop",           catmark_bishop,           kCatmark ) );

//    g_shapes.push_back( ShapeDesc("loop_cube_creases0",       loop_cube_creases0,       kLoop ) );
//    g_shapes.push_back( ShapeDesc("loop_cube_creases1",       loop_cube_creases1,       kLoop ) );
//    g_shapes.push_back( ShapeDesc("loop_cube",                loop_cube,                kLoop ) );
//    g_shapes.push_back( ShapeDesc("loop_icosahedron",         loop_icosahedron,         kLoop ) );
//    g_shapes.push_back( ShapeDesc("loop_triangle_edgecorner", loop_triangle_edgecorner, kLoop ) );
//    g_shapes.push_back( ShapeDesc("loop_triangle_edgeonly",   loop_triangle_edgeonly,   kLoop ) );
//    g_shapes.push_back( ShapeDesc("loop_triangle_edgenone",   loop_triangle_edgenone,   kLoop ) );
//    g_shapes.push_back( ShapeDesc("loop_chaikin0",            loop_chaikin0,            kLoop ) );
//    g_shapes.push_back( ShapeDesc("loop_chaikin1",            loop_chaikin1,            kLoop ) );
}
//------------------------------------------------------------------------------
