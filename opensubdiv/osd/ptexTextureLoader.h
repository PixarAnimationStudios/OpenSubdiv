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
#ifndef OSD_PTEX_TEXTURE_LOADER_H
#define OSD_PTEX_TEXTURE_LOADER_H

#include "../version.h"

#include <vector>

class PtexTexture;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// Ptex reader helper - manages up/down sizing and texel packing of blocks into
// texel pages and generate the GL texture buffers for rendering :
//
// Pages table : maps the face (quad) to a page based on gl_PrimitiveID
//
//                      face idx = 1
//                           V
//               0          1           2      ...
//         |----------|----------|----------|--------
//         | page idx | page idx | page idx | ...
//         |----------|----------|----------|--------
//
// Layout table : coordinates of the gprim in the page
//
//         - layout coords = vec4 normalized(top left (u,v), ures, vres))
//
//                   face idx = 1
//                       V
//              0        1        2      ...
//         |--------|--------|--------|--------
//         | layout | layout | layout | ...
//         |--------|--------|--------|--------
//
// Texels buffer : the packed texels
//
//             page 0                     page 1
//  |------------|-------------||------------|-------------||------
//  |............|.............||............|.............||
//  |............|.............||............|.............||
//  |............|.............||............|..... ( X ) .||
//  |.... B 0 ...|.... B 1 ....||.... B 3 ...|.............||
//  |............|.............||............|.............||
//  |............|.............||............|.............||
//  |............|.............||............|.............||
//  |------------|-------------||------------|.... B 5 ....||
//  |..........................||............|.............||
//  |..........................||............|.............||
//  |..........................||............|.............||
//  |.......... B 2 ...........||.... B 4 ...|.............||
//  |..........................||............|.............||
//  |..........................||............|.............||
//  |..........................||............|.............||
//  |--------------------------||--------------------------||-------
//
// GLSL shader computes texel coordinates with :
//   * vec3 ( X ) = ( layout.u + X, layout.v + Y, page idx )
//

class OsdPtexTextureLoader {
public:
    struct block;
    struct page;

    OsdPtexTextureLoader( PtexTexture *ptex, int gutterWidth, int pageMargin );

    ~OsdPtexTextureLoader();

    const unsigned short GetPageSize( ) const {
        return _pagesize;
    }

    const unsigned long int GetNumBlocks( ) const;

    const unsigned long int GetNumPages( ) const;

    const unsigned int * GetIndexBuffer( ) const {
        return _indexBuffer;
    }

    const float * GetLayoutBuffer( ) const {
        return _layoutBuffer;
    }

    const unsigned char * GetTexelBuffer( ) const {
        return _texelBuffer;
    }

    unsigned long int GetUncompressedSize() const {
        return _txc * _bpp;
    }

    unsigned long int GetNativeUncompressedSize() const {
        return _txn * _bpp;
    }

    int GetGutterWidth() const { return _gutterWidth; }
    
    int GetPageMargin() const { return _pageMargin; }

    void OptimizeResolution( unsigned long int memrec );

    void OptimizePacking( int maxnumpages );

    bool GenerateBuffers( );

    float EvaluateWaste( ) const;

    void ClearPages( );

    void ClearBuffers();

    void PrintBlocks() const;

    void PrintPages() const;

protected:

    friend struct block;

    PtexTexture * _ptex;

private:

    int _bpp;           // bits per pixel

    unsigned long int _txc,        // texel count for current resolution
                      _txn;        // texel count for native resolution

    std::vector<block> _blocks;

    std::vector<page *> _pages;
    unsigned short      _pagesize;

    unsigned int *  _indexBuffer;
    float *         _layoutBuffer;
    unsigned char * _texelBuffer;

    int _gutterWidth, _pageMargin;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_PTEX_TEXTURE_LOADER_H
