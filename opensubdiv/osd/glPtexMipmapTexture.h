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
#ifndef OSD_GL_PTEX_MIPMAP_TEXTURE_H
#define OSD_GL_PTEX_MIPMAP_TEXTURE_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/opengl.h"

class PtexTexture;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGLPtexMipmapTexture : OsdNonCopyable<OsdGLPtexMipmapTexture> {
public:
    static OsdGLPtexMipmapTexture * Create(PtexTexture * reader,
                                           int maxLevels=10);

    /// Returns the texture buffer containing the layout of the ptex faces
    /// in the texels texture array.
    GLuint GetLayoutTextureBuffer() const { return _layout; }

    /// Returns the texels texture array.
    GLuint GetTexelsTexture() const { return _texels; }

    ~OsdGLPtexMipmapTexture();

private:
    OsdGLPtexMipmapTexture();

    GLsizei _width,   // widht / height / depth of the 3D texel buffer
            _height,
            _depth;

    GLint   _format;  // texel color format

    GLuint _layout,   // per-face lookup table
           _texels;   // texel data
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GL_PTEX_MIPMAP_TEXTURE_H
