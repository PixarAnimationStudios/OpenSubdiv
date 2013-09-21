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
#ifndef OSD_PTEX_MIPMAP_TEXTURE_LOADER_H
#define OSD_PTEX_MIPMAP_TEXTURE_LOADER_H

#include "../version.h"

#include <vector>

class PtexTexture;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdPtexMipmapTextureLoader {
public:
    OsdPtexMipmapTextureLoader(PtexTexture *ptex,
                               int maxNumPages, int maxLevels=10);
    ~OsdPtexMipmapTextureLoader();

    const unsigned char * GetLayoutBuffer() const {
        return _layoutBuffer;
    }
    const unsigned char * GetTexelBuffer() const {
        return _texelBuffer;
    }
    int GetNumFaces() const {
        return (int)_blocks.size();
    }
    int GetNumPages() const {
        return (int)_pages.size();
    }
    int GetPageWidth() const {
        return _pageWidth;
    }
    int GetPageHeight() const {
        return _pageHeight;
    }

/*
  block : atomic texture unit
  XXX: face of 128x128 or more (64kb~) texels should be considered separately
       using ARB_sparse_texture...?

  . : per-face texels for each mipmap level
  x : guttering pixel

  xxxxxxxxxxxxxx
  x........xx..x 2x2
  x........xx..x
  x........xxxxx
  x..8x8...xxxxxxx
  x........xx....x
  x........xx....x 4x4
  x........xx....x
  x........xx....x
  xxxxxxxxxxxxxxxx

  For each face (w*h), texels with guttering and mipmap is stored into
  (w+2+w/2+2)*(h+2) area as above.

 */

/*
  Ptex loader

  Texels buffer : the packed texels

 */

private:
    struct Block {
        int index;                       // ptex index
        int nMipmaps;
        unsigned short u, v;             // top-left texel offset
        unsigned short width, height;    // texel dimension (includes mipmap)
        unsigned short texWidth, texHeight;  // texel dimension (original tile)

        void Generate(PtexTexture *ptex, unsigned char *destination,
                      int bpp, int width, int maxLevels);

        void guttering(PtexTexture *ptex, int level, int width, int height,
                       unsigned char *pptr, int bpp, int stride);

        static bool sort(const Block *a, const Block *b) {
            return (a->height > b->height) or
                ((a->height == b->height) and (a->width > b->width));
        }
    };

    struct Page;

    void generateBuffers();
    void optimizePacking(int maxNumPages);

    std::vector<Block> _blocks;
    std::vector<Page *> _pages;

    PtexTexture *_ptex;
    int _maxLevels;
    int _bpp;
    int _pageWidth, _pageHeight;

    unsigned char *_texelBuffer;
    unsigned char *_layoutBuffer;
};


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_PTEX_MIPMAP_TEXTURE_LOADER_H
