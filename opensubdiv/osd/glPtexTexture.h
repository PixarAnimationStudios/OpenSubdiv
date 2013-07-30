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
#ifndef OSD_GL_PTEX_TEXTURE_H
#define OSD_GL_PTEX_TEXTURE_H

#include "../version.h"

#include "../osd/nonCopyable.h"

#include "../osd/opengl.h"

class PtexTexture;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// OsdGLPTexture : implements simple support for ptex textures
///
/// The current implementation declares _texels as a GL_TEXTURE_2D_ARRAY of
/// n pages of a resolution that matches that of the largest face in the PTex file.
///
/// Two GL_TEXTURE_BUFFER constructs are used
/// as lookup tables :
/// * _pages stores the array index in which a given face is located
/// * _layout stores 4 float coordinates : top-left corner and width/height for each face
///
/// GLSL fragments use gl_PrimitiveID and gl_TessCoords to access the _pages and _layout
/// indirection tables, which provide then texture coordinates for the texels stored in
/// the _texels texture array.
///
/// Hbr provides per-face support for a ptex face indexing scheme. OsdGLDrawContext
/// class provides ptex face index lookup table as a texture buffer object that
/// can be accessed by GLSL shaders.
///
class OsdGLPtexTexture : OsdNonCopyable<OsdGLPtexTexture> {
public:
    static OsdGLPtexTexture * Create(PtexTexture * reader,
                                  unsigned long int targetMemory = 0,
                                  int gutterWidth = 0,
                                  int pageMargin = 0);

    /// Returns the texture buffer containing the lookup table associate each ptex
    /// face index with its 3D texture page in the texels texture array.
    GLuint GetPagesTextureBuffer() const { return _pages; }

    /// Returns the texture buffer containing the layout of the ptex faces in the
    /// texels texture array.
    GLuint GetLayoutTextureBuffer() const { return _layout; }

    /// Returns the texels texture array.
    GLuint GetTexelsTexture() const { return _texels; }

    ~OsdGLPtexTexture();

private:
    OsdGLPtexTexture();

    GLsizei _width,   // widht / height / depth of the 3D texel buffer
            _height,
            _depth;

    GLint   _format;  // texel color format

    GLuint _pages,    // per-face page indices into the texel array
           _layout,   // per-face lookup table (vec4 : top-left corner & width / height)
           _texels;   // texel data
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GL_PTEX_TEXTURE_H
