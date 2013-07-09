//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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

//------------------------------------------------------------------------------
static void initShapes() {
    g_shapes.push_back( shaperec("bilinear_cube",            bilinear_cube,            kBilinear) );
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
}
//------------------------------------------------------------------------------
