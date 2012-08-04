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
#ifndef OSD_PTEXTURE_H
#define OSD_PTEXTURE_H

#if not defined(__APPLE__)
    #if defined(_WIN32)
        #include <windows.h>
    #endif
    #include <GL/gl.h>
#else
    #include <OpenGL/gl3.h>
#endif

#include "../version.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

class PtexTexture;
class OsdMesh;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// OsdPTexture : implements simple support for ptex textures
//
// The current implementation declares _texels as a GL_TEXTURE_2D_ARRAY of
// n pages of a resolution that matches that of the largest face in the PTex file.
//
// Two GL_TEXTURE_BUFFER constructs are used
// as lookup tables :
// * _pages stores the array index in which a given face is located
// * _layout stores 4 float coordinates : top-left corner and width/height for each face
//
// GLSL fragments use gl_PrimitiveID and gl_TessCoords to access the _pages and _layout
// indirection tables, which provide then texture coordinates for the texels stored in
// the _texels texture array.
//
// Hbr provides per-face support for a ptex face indexing scheme. This
// class provides a container that can be initialized by an OsdMesh and
// instantiated in GPU memory as a texture buffer object that can be
// accessed by GLSL shaders.
//

class OsdPTexture {
public:

    static OsdPTexture * Create( PtexTexture * reader, unsigned long int targetMemory );

    // Returns the texture buffer containing the lookup table associate each ptex
    // face index with its 3D texture page in the texels texture array.
    GLuint GetPagesTextureBuffer() const { return _pages; }

    // Returns the texture buffer containing the layout of the ptex faces in the
    // texels texture array.
    GLuint GetLayoutTextureBuffer() const { return _layout; }

    // Returns the texels texture array.
    GLuint GetTexelsTexture() const { return _texels; }

   ~OsdPTexture( );

    // get/set guttering control variables
    static int GetGutterWidth() { return _gutterWidth; }

    static int GetPageMargin() { return _pageMargin; }

    static int GetGutterDebug() { return _gutterDebug; }

    static void SetGutterWidth(int width) { _gutterWidth = width; }

    static void SetPageMargin(int margin) { _pageMargin = margin; }

    static void SetGutterDebug(int debug) { _gutterDebug = debug; }

private:
    OsdPTexture();

    // Non-copyable, so these are not implemented:
    OsdPTexture(OsdPTexture const &);
    OsdPTexture & operator=(OsdPTexture const &);


    GLsizei _width,   // widht / height / depth of the 3D texel buffer
            _height,
            _depth;

    GLint   _format;  // texel color format

    GLuint _pages,    // per-face page indices into the texel array
           _layout,   // per-face lookup table (vec4 : top-left corner & width / height)
           _texels;   // texel data

    static int _gutterWidth, _pageMargin, _gutterDebug;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_PTEXTURE_H
