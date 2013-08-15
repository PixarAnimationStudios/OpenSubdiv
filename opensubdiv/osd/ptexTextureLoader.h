//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
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

    unsigned short GetPageSize( ) const {
        return _pagesize;
    }

    unsigned long int GetNumBlocks( ) const;

    unsigned long int GetNumPages( ) const;

    unsigned int * GetIndexBuffer( ) const {
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
