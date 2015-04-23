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

#include "../common/shape_utils.h"

struct shaperec {

    shaperec(char const * iname, std::string const & idata, Scheme ischeme) :
        name(iname), data(idata), scheme(ischeme) { }

    std::string name,
                data;
    Scheme      scheme;
};

static std::vector<shaperec> g_shapes;

#include "../shapes/bilinear_cube.h"
#include "../shapes/catmark_chaikin0.h"
#include "../shapes/catmark_chaikin1.h"
#include "../shapes/catmark_cube_corner0.h"
#include "../shapes/catmark_cube_corner1.h"
#include "../shapes/catmark_cube_corner2.h"
#include "../shapes/catmark_cube_corner3.h"
#include "../shapes/catmark_cube_corner4.h"
#include "../shapes/catmark_cube_creases0.h"
#include "../shapes/catmark_cube_creases1.h"
#include "../shapes/catmark_cube.h"
#include "../shapes/catmark_dart_edgecorner.h"
#include "../shapes/catmark_dart_edgeonly.h"
#include "../shapes/catmark_edgecorner.h"
#include "../shapes/catmark_edgeonly.h"
#include "../shapes/catmark_flap.h"
#include "../shapes/catmark_flap2.h"
#include "../shapes/catmark_pyramid_creases0.h"
#include "../shapes/catmark_pyramid_creases1.h"
#include "../shapes/catmark_pyramid.h"
#include "../shapes/catmark_square_hedit0.h"
#include "../shapes/catmark_square_hedit1.h"
#include "../shapes/catmark_square_hedit2.h"
#include "../shapes/catmark_square_hedit3.h"
#include "../shapes/catmark_tent_creases0.h"
#include "../shapes/catmark_tent_creases1.h"
#include "../shapes/catmark_tent.h"
#include "../shapes/loop_cube_creases0.h"
#include "../shapes/loop_cube_creases1.h"
#include "../shapes/loop_cube.h"
#include "../shapes/loop_icosahedron.h"
#include "../shapes/loop_saddle_edgecorner.h"
#include "../shapes/loop_saddle_edgeonly.h"
#include "../shapes/loop_triangle_edgecorner.h"
#include "../shapes/loop_triangle_edgeonly.h"
#include "../shapes/loop_chaikin0.h"
#include "../shapes/loop_chaikin1.h"

//------------------------------------------------------------------------------
static void initShapes() {
    g_shapes.push_back( shaperec("bilinear_cube",            bilinear_cube,            kBilinear) );
    g_shapes.push_back( shaperec("catmark_chaikin0",         catmark_chaikin0,         kCatmark ) );
    g_shapes.push_back( shaperec("catmark_chaikin1",         catmark_chaikin1,         kCatmark ) );
    g_shapes.push_back( shaperec("catmark_cube_corner0",     catmark_cube_corner0,     kCatmark ) );
    g_shapes.push_back( shaperec("catmark_cube_corner1",     catmark_cube_corner1,     kCatmark ) );
    g_shapes.push_back( shaperec("catmark_cube_corner2",     catmark_cube_corner2,     kCatmark ) );
    g_shapes.push_back( shaperec("catmark_cube_corner3",     catmark_cube_corner3,     kCatmark ) );
    g_shapes.push_back( shaperec("catmark_cube_corner4",     catmark_cube_corner4,     kCatmark ) );
    g_shapes.push_back( shaperec("catmark_cube_creases0",    catmark_cube_creases0,    kCatmark ) );
    g_shapes.push_back( shaperec("catmark_cube_creases1",    catmark_cube_creases1,    kCatmark ) );
    g_shapes.push_back( shaperec("catmark_cube",             catmark_cube,             kCatmark ) );
    g_shapes.push_back( shaperec("catmark_dart_edgecorner",  catmark_dart_edgecorner,  kCatmark ) );
    g_shapes.push_back( shaperec("catmark_dart_edgeonly",    catmark_dart_edgeonly,    kCatmark ) );
    g_shapes.push_back( shaperec("catmark_edgecorner",       catmark_edgecorner,       kCatmark ) );
    g_shapes.push_back( shaperec("catmark_edgeonly",         catmark_edgeonly,         kCatmark ) );
    g_shapes.push_back( shaperec("catmark_flap",             catmark_flap,             kCatmark ) );
    g_shapes.push_back( shaperec("catmark_flap2",            catmark_flap2,            kCatmark ) );
    g_shapes.push_back( shaperec("catmark_pyramid_creases0", catmark_pyramid_creases0, kCatmark ) );
    g_shapes.push_back( shaperec("catmark_pyramid_creases1", catmark_pyramid_creases1, kCatmark ) );
    g_shapes.push_back( shaperec("catmark_pyramid",          catmark_pyramid,          kCatmark ) );
    g_shapes.push_back( shaperec("catmark_square_hedit0",    catmark_square_hedit0,    kCatmark ) );
    g_shapes.push_back( shaperec("catmark_square_hedit1",    catmark_square_hedit1,    kCatmark ) );
    g_shapes.push_back( shaperec("catmark_square_hedit2",    catmark_square_hedit2,    kCatmark ) );
    g_shapes.push_back( shaperec("catmark_square_hedit3",    catmark_square_hedit3,    kCatmark ) );
    g_shapes.push_back( shaperec("catmark_tent_creases0",    catmark_tent_creases0,    kCatmark ) );
    g_shapes.push_back( shaperec("catmark_tent_creases1",    catmark_tent_creases1 ,   kCatmark ) );
    g_shapes.push_back( shaperec("catmark_tent",             catmark_tent,             kCatmark ) );
    g_shapes.push_back( shaperec("loop_cube_creases0",       loop_cube_creases0,       kLoop ) );
    g_shapes.push_back( shaperec("loop_cube_creases1",       loop_cube_creases1,       kLoop ) );
    g_shapes.push_back( shaperec("loop_cube",                loop_cube,                kLoop ) );
    g_shapes.push_back( shaperec("loop_icosahedron",         loop_icosahedron,         kLoop ) );
    g_shapes.push_back( shaperec("loop_saddle_edgecorner",   loop_saddle_edgecorner,   kLoop ) );
    g_shapes.push_back( shaperec("loop_saddle_edgeonly",     loop_saddle_edgeonly,     kLoop ) );
    g_shapes.push_back( shaperec("loop_triangle_edgecorner", loop_triangle_edgecorner, kLoop ) );
    g_shapes.push_back( shaperec("loop_triangle_edgeonly",   loop_triangle_edgeonly,   kLoop ) );
    g_shapes.push_back( shaperec("loop_chaikin0",            loop_chaikin0,            kLoop ) );
    g_shapes.push_back( shaperec("loop_chaikin1",            loop_chaikin1,            kLoop ) );
}
//------------------------------------------------------------------------------
