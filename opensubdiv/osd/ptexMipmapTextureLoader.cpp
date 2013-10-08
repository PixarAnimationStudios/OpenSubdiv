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

#include "../osd/ptexMipmapTextureLoader.h"
#include "../osd/error.h"

#include <Ptexture.h>
#include <vector>
#include <list>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// resample border texels for guttering
//
static int
resampleBorder(PtexTexture * ptex, int face, int edgeId, unsigned char *result,
               int dstLength, int bpp,
               float srcStart = 0.0f, float srcEnd = 1.0f)
{
    const Ptex::FaceInfo & pf = ptex->getFaceInfo(face);

    int edgeLength = (edgeId == 0 || edgeId == 2) ? pf.res.u() : pf.res.v();
    int srcOffset = (int)(srcStart*edgeLength);
    int srcLength = (int)((srcEnd-srcStart)*edgeLength);

    // if dstLength < 0, returns as original resolution without scaling
    if (dstLength < 0) dstLength = srcLength;

    if (dstLength >= srcLength) {
        PtexFaceData * data = ptex->getData(face);
        unsigned char *border = new unsigned char[bpp*srcLength];

        // order of the result will be flipped to match adjacent pixel order
        for (int i = 0; i < srcLength; ++i) {
            int u = 0, v = 0;
            if (edgeId == Ptex::e_bottom) {
                u = edgeLength-1-(i+srcOffset);
                v = 0;
            } else if (edgeId == Ptex::e_right) {
                u = pf.res.u()-1;
                v = edgeLength-1-(i+srcOffset);
            } else if (edgeId == Ptex::e_top) {
                u = i+srcOffset;
                v = pf.res.v()-1;
            } else if (edgeId == Ptex::e_left) {
                u = 0;
                v = i+srcOffset;
            }
            data->getPixel(u, v, &border[i*bpp]);
        }

        // nearest resample to fit dstLength
        for (int i = 0; i < dstLength; ++i) {
            for (int j = 0; j < bpp; j++) {
                result[i*bpp+j] = border[(i*srcLength/dstLength)*bpp+j];
            }
        }
        delete[] border;
    } else {
        Ptex::Res res = pf.res;
        while (srcLength > dstLength && res.ulog2 && res.vlog2) {
            --res.ulog2;
            --res.vlog2;
            srcLength /= 2;
        }

        PtexFaceData * data = ptex->getData(face, res);
        unsigned char *border = new unsigned char[bpp*srcLength];
        edgeLength = (edgeId == 0 || edgeId == 2) ? res.u() : res.v();
        srcOffset = (int)(srcStart*edgeLength);

        for (int i = 0; i < dstLength; ++i) {
            int u = 0, v = 0;
            if (edgeId == Ptex::e_bottom) {
                u = edgeLength-1-(i+srcOffset);
                v = 0;
            } else if (edgeId == Ptex::e_right) {
                u = res.u() - 1;
                v = edgeLength-1-(i+srcOffset);
            } else if (edgeId == Ptex::e_top) {
                u = i+srcOffset;
                v = res.v() - 1;
            } else if (edgeId == Ptex::e_left) {
                u = 0;
                v = i+srcOffset;
            }
            data->getPixel(u, v, &border[i*bpp]);

            for (int j = 0; j < bpp; ++j) {
                result[i*bpp+j] = border[i*bpp+j];
            }
        }
        delete[] border;
    }


    return srcLength;
}

// flip order of pixel buffer
static void
flipBuffer(unsigned char *buffer, int length, int bpp)
{
    for (int i = 0; i < length/2; ++i) {
        for (int j = 0; j < bpp; j++) {
            std::swap(buffer[i*bpp+j], buffer[(length-1-i)*bpp+j]);
        }
    }
}

// sample neighbor face's edge
static void
sampleNeighbor(PtexTexture * ptex, unsigned char *border,
               int face, int edge, int length, int bpp)
{
    const Ptex::FaceInfo &fi = ptex->getFaceInfo(face);

    // copy adjacent borders
    int adjface = fi.adjface(edge);
    if (adjface != -1) {
        int ae = fi.adjedge(edge);
        if (!fi.isSubface() && ptex->getFaceInfo(adjface).isSubface()) {
            /* nonsubface -> subface (1:0.5)  see http://ptex.us/adjdata.html for more detail
              +------------------+
              |       face       |
              +--------edge------+
              | adj face |       |
              +----------+-------+
            */
            resampleBorder(ptex, adjface, ae, border, length/2, bpp);
            const Ptex::FaceInfo &sfi1 = ptex->getFaceInfo(adjface);
            adjface = sfi1.adjface((ae+3)%4);
            ae = (sfi1.adjedge((ae+3)%4)+3)%4;
            resampleBorder(ptex, adjface, ae, border+(length/2*bpp),
                           length/2, bpp);

        } else if (fi.isSubface() && !ptex->getFaceInfo(adjface).isSubface()) {
            /* subface -> nonsubface (0.5:1).   two possible configuration
                     case 1                    case 2
              +----------+----------+  +----------+----------+--------+
              |   face   |    B     |  |          |  face    |   B    |
              +---edge---+----------+  +----------+--edge----+--------+
              |0.0      0.5      1.0|  |0.0      0.5      1.0|
              |       adj face      |  |       adj face      |
              +---------------------+  +---------------------+
            */
            int Bf = fi.adjface((edge+1)%4);
            int Be = fi.adjedge((edge+1)%4);
            int f = ptex->getFaceInfo(Bf).adjface((Be+1)%4);
            int e = ptex->getFaceInfo(Bf).adjedge((Be+1)%4);
            if (f == adjface && e == ae)  // case 1
                resampleBorder(ptex, adjface, ae, border,
                               length, bpp, 0.0, 0.5);
            else  // case 2
                resampleBorder(ptex, adjface, ae, border,
                               length, bpp, 0.5, 1.0);

        } else {
            /*  ordinary case (1:1 match)
                +------------------+
                |       face       |
                +--------edge------+
                |    adj face      |
                +----------+-------+
            */
            resampleBorder(ptex, adjface, ae, border, length, bpp);
        }
    } else {
        /* border edge. duplicate itself
           +-----------------+
           |       face      |
           +-------edge------+
        */
        resampleBorder(ptex, face, edge, border, length, bpp);
        flipBuffer(border, length, bpp);
    }
}

// get corner pixel by traversing all adjacent faces around vertex
//
static bool
getCornerPixel(PtexTexture *ptex, float *resultPixel, int numchannels,
               int face, int edge, int bpp, int level, unsigned char *lineBuffer)
{
    const Ptex::FaceInfo &fi = ptex->getFaceInfo(face);

    /*
       see http://ptex.us/adjdata.html Figure 2 for the reason of conditions edge==1 and 3
    */

    if (fi.isSubface() && edge == 3) {
        /*
          in T-vertex case, this function sets 'D' pixel value to *resultPixel and returns false
                gutter line
                |
          +------+-------+
          |      |       |
          |     D|C      |<-- gutter line
          |      *-------+
          |     B|A [2]  |
          |      |[3] [1]|
          |      |  [0]  |
          +------+-------+
        */
        int adjface = fi.adjface(edge);
        if (adjface != -1 and !ptex->getFaceInfo(adjface).isSubface()) {
            int length = resampleBorder(ptex,
                                        adjface,
                                        fi.adjedge(edge),
                                        lineBuffer,
                                        /*dstLength=*/-1,
                                        bpp,
                                        0.0f, 1.0f);
            /* then lineBuffer contains

               |-------DB-------|
               0       ^        length-1
                       length/2-1
             */
            Ptex::ConvertToFloat(resultPixel,
                                 lineBuffer + bpp*(length/2-1),
                                 ptex->dataType(),
                                 numchannels);
            return true;
        }
    }
    if (fi.isSubface() && edge == 1) {
        /*      gutter line
                |
          +------+-------+
          |      |  [3]  |
          |      |[0] [2]|
          |     B|A [1]  |
          |      *-------+
          |     D|C      |<-- gutter line
          |      |       |
          +------+-------+

             note: here we're focusing on vertex A which corresponds to the edge 1,
                   but the edge 0 is an adjacent edge to get D pixel.
        */
        int adjface = fi.adjface(0);
        if (adjface != -1 and !ptex->getFaceInfo(adjface).isSubface()) {
            int length = resampleBorder(ptex,
                                        adjface,
                                        fi.adjedge(0),
                                        lineBuffer,
                                        /*dstLength=*/-1,
                                        bpp,
                                        0.0f, 1.0f);
            /* then lineBuffer contains

               |-------BD-------|
               0        ^       length-1
                        length/2
             */
            Ptex::ConvertToFloat(resultPixel,
                                 lineBuffer + bpp*(length/2),
                                 ptex->dataType(),
                                 numchannels);
            return true;
        }
    }

    int currentFace = face;
    int currentEdge = edge;
    int uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    float *pixel = (float*)alloca(sizeof(float)*numchannels);
    float *accumPixel = (float*)alloca(sizeof(float)*numchannels);

    // clear accum pixel
    memset(accumPixel, 0, sizeof(float)*numchannels);

    bool clockWise = true;
    int nFace = 0;
    do {
        nFace++;
        if (nFace > 255) {
            OsdWarning("High valence detected in %s : invalid adjacency around "
                       "face %d", ptex->path(), face);
            break;
        }::
        Ptex::FaceInfo info = ptex->getFaceInfo(currentFace);
        int ulog2 = std::max(0, info.res.ulog2 - level);
        int vlog2 = std::max(0, info.res.vlog2 - level);
        Ptex::Res res(ulog2, vlog2);
        ptex->getPixel(currentFace,
                       uv[currentEdge][0] * (res.u()-1),
                       uv[currentEdge][1] * (res.v()-1),
                       pixel, 0, numchannels, res);
        for (int j = 0; j < numchannels; ++j) {
            accumPixel[j] += pixel[j];
            if (nFace == 3) {
                resultPixel[j] = pixel[j];
            }
        }

        // next face
        if (clockWise) {
            currentFace = info.adjface(currentEdge);
            currentEdge = info.adjedge(currentEdge);
            currentEdge = (currentEdge+1)%4;
        } else {
            currentFace = info.adjface((currentEdge+3)%4);
            currentEdge = info.adjedge((currentEdge+3)%4);
        }

        if (currentFace == -1) {
            // border case.
            if (clockWise) {
                // reset position and restart counter clock wise
                Ptex::FaceInfo sinfo = ptex->getFaceInfo(face);
                currentFace = sinfo.adjface((edge+3)%4);
                currentEdge = sinfo.adjedge((edge+3)%4);
                clockWise = false;
            } else {
                // end
                break;
            }
        }
    } while (currentFace != -1 and currentFace != face);

    if (nFace == 4) {
        return true;
    }

    // non-4 valence. let's average and return false;
    for (int j = 0; j < numchannels; ++j) {
        resultPixel[j] = accumPixel[j]/nFace;
    }
    return false;
}

// sample neighbor pixels and populate around blocks
void
OsdPtexMipmapTextureLoader::Block::guttering(PtexTexture *ptex, int level,
                                             int width, int height,
                                             unsigned char *pptr, int bpp,
                                             int stride)
{
    // XXX: fixme
    unsigned char * lineBuffer = new unsigned char[16384 * bpp];

    for (int edge = 0; edge < 4; edge++) {
        int len = (edge == 0 or edge == 2) ? width : height;
        sampleNeighbor(ptex, lineBuffer, this->index, edge, len, bpp);

        unsigned char *s = lineBuffer, *d;
        for (int j = 0; j < len; ++j) {
            d = pptr;
            switch (edge) {
            case Ptex::e_bottom:
                d += bpp * (j + 1);
                break;
            case Ptex::e_right:
                d += stride * (j + 1) + bpp * (width+1);
                break;
            case Ptex::e_top:
                d += stride * (height+1) + bpp*(len-j);
                break;
            case Ptex::e_left:
                d += stride * (len-j);
                break;
            }
            for (int k = 0; k < bpp; k++)
                 *d++ = *s++;
        }
    }

    // fix corner pixels
    int numchannels = ptex->numChannels();
    float *accumPixel = new float[numchannels];
    int uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};


    for (int edge = 0; edge < 4; edge++) {
        int du = uv[edge][0];
        int dv = uv[edge][1];

        /*  There are 3 cases when filling a corner pixel on gutter.

            case 1: Regular 4 valence
                    We already have correct 'B' and 'C' pixels by edge
                    resampling above.
                    so here only one more pixel 'D' is needed,
                    and it will be placed on the gutter corner.
               +-----+-----+
               |     |     |<-current
               |    B|A    |
               +-----*-----+
               |    D|C    |
               |     |     |
               +-----+-----+

            case 2: T-vertex case (note that this doesn't mean 3 valence)
                    If the current face comes from non-quad root face, there
                    could be a T-vertex on its corner. Just like case 1,
                    need to fill border corner with pixel 'D'.
               +-----+-----+
               |     |     |<-current
               |    B|A    |
               |     *-----+
               |    D|C    |
               |     |     |
               +-----+-----+

            case 3: Other than 4 valence case
                        (everything else, including boundary)
                    Since guttering pixels are placed on the border of each
                    ptex faces, it's not possible to store more than 4 pixels
                    at a coner for a reasonable interpolation.
                    In this case, we need to average all corner pixels and
                    overwrite with an averaged value, so that every face
                    vertex picks the same value.
               +---+---+
               |   |   |<-current
               |  B|A  |
               +---*---|
               | D/E\C |
               | /   \ |
               |/     \| 
               +-------+
         */

        if (getCornerPixel(ptex, accumPixel, numchannels,
                           this->index, edge, bpp, level, lineBuffer)) {
            // case 1 and case 2
            if (edge == 1 || edge == 2) du += width;
            if (edge == 2 || edge == 3) dv += height;
            unsigned char *d = pptr + dv*stride + du*bpp;
            Ptex::ConvertFromFloat(d, accumPixel,
                                   ptex->dataType(), numchannels);
        } else {
            // case 3, set accumPixel to the corner 4 pixels
            if (edge == 1 || edge == 2) du += width - 1;
            if (edge == 2 || edge == 3) dv += height - 1;
            for (int u = 0; u < 2; ++u) {
                for (int v = 0; v < 2; ++v) {
                    unsigned char *d = pptr + (dv+u)*stride + (du+v)*bpp;
                    Ptex::ConvertFromFloat(d, accumPixel,
                                           ptex->dataType(), numchannels);
                }
            }
        }
    }
    delete[] lineBuffer;
    delete[] accumPixel;
}

void
OsdPtexMipmapTextureLoader::Block::Generate(PtexTexture *ptex,
                                            unsigned char *destination,
                                            int bpp, int width, int maxLevels)
{
    const Ptex::FaceInfo &faceInfo = ptex->getFaceInfo(index);
    int stride = bpp * width;

    Ptex::Res res = faceInfo.res;
    int ulog2 = this->ulog2;
    int vlog2 = this->vlog2;

    int level = 0;
    int uofs = u, vofs = v;
    while (ulog2 >= 1
           and vlog2 >= 1
           and (maxLevels == -1 or level <= maxLevels)) {
        if (level % 2 == 1)
            uofs += (1<<(ulog2+1))+2;
        if ((level > 0) and (level % 2 == 0))
            vofs += (1<<(vlog2+1)) + 2;

        unsigned char *dst = destination + vofs * stride + uofs * bpp;
        unsigned char *dstData = destination
            + (vofs + 1) * stride
            + (uofs + 1) * bpp;
        ptex->getData(index, dstData, stride, Ptex::Res(ulog2, vlog2));

        guttering(ptex, level, 1<<ulog2, 1<<vlog2, dst, bpp, stride);

        --ulog2;
        --vlog2;
        ++level;
    }
    nMipmaps = level;
}

/*
  page :


*/

struct OsdPtexMipmapTextureLoader::Page
{
    struct Slot
    {
        Slot(unsigned short u, unsigned short v,
             unsigned short w, unsigned short h) :
            u(u), v(v), width(w), height(h) { }

        unsigned short u, v, width, height;

        // returns true if a block can fit in this slot
        bool Fits(const Block *block) {
            return (block->width <= width) and (block->height <= height);
        }
    };

    typedef std::list<Block *> BlockList;

    Page(unsigned short width, unsigned short height) {
        _slots.push_back(Slot(0, 0, width, height));
    }

    bool IsFull() const {
        return _slots.empty();
    }

    // true when the block "b" is successfully added to this  page :
    //
    //  |--------------------------|       |------------|-------------|
    //  |                          |       |............|             |
    //  |                          |       |............|             |
    //  |                          |       |.... B .....| Right Slot  |
    //  |                          |       |............|             |
    //  |                          |       |............|             |
    //  |                          |       |------------|-------------|
    //  |      Original Slot       |  ==>  |                          |
    //  |                          |       |                          |
    //  |                          |       |       Bottom Slot        |
    //  |                          |       |                          |
    //  |                          |       |                          |
    //  |--------------------------|       |--------------------------|
    //
    bool AddBlock(Block *block) {
        for (SlotList::iterator it = _slots.begin(); it != _slots.end(); ++it) {
            if (it->Fits(block)) {
                _blocks.push_back(block);

                block->u = it->u;
                block->v = it->v;

                // add new slot to the right
                if (it->width > block->width) {
                    _slots.push_front(Slot(it->u + block->width,
                                           it->v,
                                           it->width - block->width,
                                           block->height));
                }
                // add new slot to the bottom
                if (it->height > block->height) {
                    _slots.push_back(Slot(it->u,
                                          it->v + block->height,
                                          it->width,
                                          it->height - block->height));
                }
                _slots.erase(it);
                return true;
            }
        }
        return false;
    }

    void Generate(PtexTexture *ptex, unsigned char *destination,
                  int bpp, int width, int maxLevels) {
        for (BlockList::iterator it = _blocks.begin();
             it != _blocks.end(); ++it) {
            (*it)->Generate(ptex, destination, bpp, width, maxLevels);
        }
    }

    const BlockList &GetBlocks() const {
        return _blocks;
    }

    void Dump() const {
        for (BlockList::const_iterator it = _blocks.begin();
             it != _blocks.end(); ++it) {
            printf(" (%d, %d)  %d x %d\n",
                   (*it)->u, (*it)->v, (*it)->width, (*it)->height);
        }
    }

private:
    BlockList _blocks;

    typedef std::list<Slot> SlotList;
    SlotList _slots;
};

OsdPtexMipmapTextureLoader::OsdPtexMipmapTextureLoader(PtexTexture *ptex,
                                                       int maxNumPages,
                                                       int maxLevels,
                                                       size_t targetMemory) :
    _ptex(ptex), _maxLevels(maxLevels), _bpp(0),
    _pageWidth(0), _pageHeight(0), _texelBuffer(NULL), _layoutBuffer(NULL),
    _memoryUsage(0)
{
    // byte per pixel
    _bpp = ptex->numChannels() * Ptex::DataSize(ptex->dataType());

    int numFaces = ptex->numFaces();
    _blocks.resize(numFaces);

    for (int i = 0; i < numFaces; ++i) {
        const Ptex::FaceInfo &faceInfo = ptex->getFaceInfo(i);
        _blocks[i].index = i;
        _blocks[i].SetSize(faceInfo.res.ulog2,
                           faceInfo.res.vlog2,
                           _maxLevels != 0);
    }

    optimizePacking(maxNumPages, targetMemory);
    generateBuffers();
}

OsdPtexMipmapTextureLoader::~OsdPtexMipmapTextureLoader()
{
    delete _texelBuffer;
    delete _layoutBuffer;
}

void
OsdPtexMipmapTextureLoader::optimizePacking(int maxNumPages,
                                            size_t targetMemory)
{
    size_t numTexels = 0;

    // prepare a list of pointers
    typedef std::vector<Block> BlockArray;
    typedef std::list<Block *> BlockPtrList;
    BlockPtrList blocks;
    for (BlockArray::iterator it = _blocks.begin(); it != _blocks.end(); ++it) {
        blocks.push_back(&(*it));
        numTexels += it->GetNumTexels();
    }

    // sort blocks by height-width order
    blocks.sort(Block::sort);

    // try to fit into the target memory size if specified
    if (targetMemory != 0 and _bpp * numTexels > targetMemory) {
        size_t numTargetTexels = targetMemory / _bpp;
        while (numTexels > numTargetTexels) {
            Block *block = blocks.front();

            if (block->ulog2 < 2 or block->vlog2 < 2) break;

            // pick a smaller mipmap
            numTexels -= block->GetNumTexels();
            block->SetSize(block->ulog2-1, block->vlog2-1, _maxLevels != 0);
            numTexels += block->GetNumTexels();

            // move to the last
            blocks.pop_front();
            blocks.push_back(block);
        }
    }

    // compute page size ---------------------------------------------
    {
        // page size is set to the largest edge of the largest block :
        // this is the smallest possible page size, which should minimize
        // the texels wasted on the "last page" when the smallest blocks are
        // being packed.
        int w = 0, h = 0;
        for (BlockPtrList::iterator it = blocks.begin();
             it != blocks.end(); ++it) {
            w = std::max(w, (int)(*it)->width);
            h = std::max(h, (int)(*it)->height);
        }

        // grow the pagesize to make sure the optimization will not exceed
        // the maximum number of pages allowed
        int minPageSize = 512;
        int maxPageSize = 4096;  // XXX:should be configurable.

        // use minPageSize if too small
        if (w < minPageSize) w = w*(minPageSize/w + 1);
        if (h < minPageSize) h = h*(minPageSize/h + 1);

        // rough estimate of num pages
        int estimatedNumPages = (int)numTexels/w/h;

        // if expecting too many pages, increase page size
        int pageLimit = std::max(1, maxNumPages/2);
        if (estimatedNumPages > pageLimit) {
            w = std::min(w*(estimatedNumPages/pageLimit), maxPageSize);
            estimatedNumPages = (int)numTexels/w/h;
        }
        if (estimatedNumPages > pageLimit) {
            h = std::min(h*(estimatedNumPages/pageLimit), maxPageSize);
        }

        _pageWidth = w;
        _pageHeight = h;
    }

    // pack blocks into slots ----------------------------------------
    size_t firstslot = 0;
    for (BlockPtrList::iterator it = blocks.begin();
         it != blocks.end(); ++it) {
        Block *block = *it;

        // traverse existing pages for a suitable slot ---------------
        bool added = false;
        for (size_t p = firstslot; p < _pages.size(); ++p) {
            if ((added = _pages[p]->AddBlock(block)) == true) {
                break;
            }
        }
        // if none of page was found : start new page
        if (!added) {
            Page *page = new Page(_pageWidth, _pageHeight);
            added = page->AddBlock(block);
            assert(added);
            _pages.push_back(page);
        }

        // adjust the page flag to the first page with open slots
        if (_pages.size() > (firstslot+1) and
            _pages[firstslot+1]->IsFull()) ++firstslot;
    }

#if 0
    for (size_t i = 0; i < _pages.size(); ++i) {
        printf("Page %ld : \n", i);
        _pages[i]->Dump();
    }
#endif
}

void
OsdPtexMipmapTextureLoader::generateBuffers()
{
    // ptex layout struct
    // struct Layout {
    //     unsigned short page;
    //     unsigned short nMipmap;
    //     unsigned short u;
    //     unsigned short v;
    //     unsigned char  width log2;
    //     unsigned char  height log2;
    // };

    int numFaces = (int)_blocks.size();
    int numPages = (int)_pages.size();

    // populate the texels
    int pageStride = _bpp * _pageWidth * _pageHeight;

    _texelBuffer = new unsigned char[pageStride * numPages];
    _memoryUsage = pageStride * numPages;
    memset(_texelBuffer, 0, pageStride * numPages);

    for (int i = 0; i < numPages; ++i) {
        _pages[i]->Generate(_ptex, _texelBuffer + pageStride * i,
                            _bpp, _pageWidth, _maxLevels);
    }

    // populate the layout texture buffer
    _layoutBuffer = new unsigned char[numFaces * sizeof(short) * 5];
    _memoryUsage += numFaces * sizeof(short) * 5;
    for (int i = 0; i < numPages; ++i) {
        Page *page = _pages[i];
        for (Page::BlockList::const_iterator it = page->GetBlocks().begin();
             it != page->GetBlocks().end(); ++it) {
            int ptexIndex = (*it)->index;
            unsigned short *p = (unsigned short*)(_layoutBuffer
                                                  + sizeof(short)*5*ptexIndex);
            *p++ = i;  // page
            *p++ = (*it)->nMipmaps-1;
            *p++ = (*it)->u+1;
            *p++ = (*it)->v+1;
            *p++ = ((*it)->ulog2 << 8) | (*it)->vlog2;
        }
    }

#if 0
    // debug
    FILE *fp = fopen("out.ppm", "w");
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", _pageWidth, _pageHeight * numPages);
    fprintf(fp, "255\n");
    unsigned char *p = _texelBuffer;
    for (int i = 0; i < numPages; ++i) {
        for (int y = 0; y < _pageHeight; ++y) {
            for (int x = 0; x < _pageWidth; ++x) {
                fprintf(fp, "%d %d %d ", (int)p[0], (int)p[1], (int)p[2]);
                p += _bpp;
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
#endif
}


} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
