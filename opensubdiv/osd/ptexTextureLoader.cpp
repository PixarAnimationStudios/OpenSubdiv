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

#include "../osd/error.h"
#include "../osd/ptexTextureLoader.h"

#include <Ptexture.h>
#include <algorithm>
#include <iostream>
#include <string.h>
#include <list>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// block : atomic texture unit, points to the texels contained in a face
//
//  |-----------------------|   |-----------------------|
//  | (u,v)                 |   | (u,v)                 |
//  |                       |   |                       |
//  |                       |   |                       |
//  |      Block 0          |   |      Block 1          |
//  |                       |   |                       |
//  |                  vres | + |                  vres | ...
//  |                       |   |                       |
//  |                       |   |                       |
//  |                       |   |                       |
//  |                       |   |                       |
//  |        ures           |   |         ures          |
//  |-----------------------|   |-----------------------|
//
struct OsdPtexTextureLoader::block {

    int idx;                    // PTex face index

    unsigned short u, v;        // location in memory pages

    Ptex::Res current,          // current resolution of the block
              native;           // native resolution of the block

    // comparison operator : true when the current texel area of "b" is greater than "a"
    static bool currentAreaSort(block const * a, block const * b) {
        int darea = a->current.ulog2 * a->current.vlog2  -
                    b->current.ulog2 * b->current.vlog2;
        if (darea==0)
          return a->current.ulog2 < b->current.ulog2;
        else
          return darea < 0;
    }

    // returns a "distance" metric from the native texel resolution
    int8_t distanceFromNative( ) const {
        int8_t udist = native.ulog2-current.ulog2,
               vdist = native.vlog2-current.vlog2;

        return udist * udist + vdist * vdist;
    }

    // desirability predicates for resolution scaling optimizations
    static bool downsizePredicate( block const * b0, block const * b1 ) {
        int8_t d0 = b0->distanceFromNative(),
               d1 = b1->distanceFromNative();

        if (d0==d1)
          return (b0->current.ulog2 * b0->current.vlog2) <
                 (b1->current.ulog2 * b1->current.vlog2);
        else
          return d0 < d1;
    }

    static bool upsizePredicate( block const * b0, block const * b1 ) {
        int8_t d0 = b0->distanceFromNative(),
               d1 = b1->distanceFromNative();

        if (d0==d1)
          return (b0->current.ulog2 * b0->current.vlog2) <
                 (b1->current.ulog2 * b1->current.vlog2);
        else
          return d0 > d1;
    }

    friend std::ostream & operator <<(std::ostream &s, block const & b);
};

// page : a handle on a single page of the GL texture array that contains the
//        packed PTex texels. Pages populate "empty" slots with "blocks" of
//        texels.
// Note : pages are square, because i said so...
//
//  |--------------------------|             |------------|-------------|
//  |                          |             |............|.............|
//  |                          |             |............|.............|
//  |                          |             |............|.............|
//  |                          |             |.... B 0 ...|.... B 1 ..../
//  |                          |             |............|.............|
//  |                          |             |............|.............|
//  |                          |             |............|.............|
//  |        Empty Page        |             |------------|-------------|
//  |                          |  packed =>  |..........................|
//  |                          |             |..........................|
//  |                          |             |..........................|
//  |                          |             |.......... B 2 ...........|
//  |                          |             |..........................|
//  |                          |             |..........................|
//  |                          |             |..........................|
//  |--------------------------|             |--------------------------|
//
struct OsdPtexTextureLoader::page {

    //----------------------------------------------------------------
    // slot : rectangular block of available texels in a page
    struct slot {
        unsigned short u, v, ures, vres;

        slot( unsigned short size ) : u(0), v(0), ures(size), vres(size) { }

        slot( unsigned short iu, unsigned short iv, unsigned short iures, unsigned short ivres ) :
              u(iu), v(iv), ures(iures), vres(ivres) { }

        // true if a block can fit in this slot
        bool fits( block const * b, int gutterWidth ) {
            return ( (b->current.u()+2*gutterWidth)<=ures ) &&
                ((b->current.v()+2*gutterWidth)<=vres);
        }
    };

    //----------------------------------------------------------------
    typedef std::list<block *> blist;
    blist blocks;

    typedef std::list<slot> slist;
    slist slots;

    // construct a page with a single empty slot the size of the page
    page( unsigned short pagesize ) {
      slots.push_back( slot( pagesize) );
    }

    // true if there is no empty texels in the page (ie. no slots left)
    bool isFull( ) const {
        return slots.size()==0;
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
    bool addBlock( block * b, int gutterWidth ) {
        for (slist::iterator i=slots.begin(); i!=slots.end(); ++i) {

            if (i->fits( b, gutterWidth )) {

                blocks.push_back( b );

                int w = gutterWidth,
                    w2 = 2*w;

                b->u=i->u + w;
                b->v=i->v + w;

                // add new slot to the right
                if (i->ures > (b->current.u()+w2)) {
                    slots.push_front( slot( i->u+b->current.u()+w2,
                                            i->v,
                                            i->ures-b->current.u()-w2,
                                            b->current.v()+w2));
                }

                // add new slot to the bottom
                if (i->vres > (b->current.v()+w2)) {
                    slots.push_back( slot( i->u,
                                           i->v+b->current.v()+w2,
                                           i->ures,
                                           i->vres-b->current.v()-w2 ));
                }

                slots.erase( i );
                return true;
            }
        }
        return false;
    }

    friend std::ostream & operator <<(std::ostream &s, const page & p);
};

OsdPtexTextureLoader::OsdPtexTextureLoader( PtexTexture * p,
                                      int gutterWidth, int pageMargin) :
    _ptex(p), _indexBuffer( NULL ), _layoutBuffer( NULL ), _texelBuffer(NULL),
    _gutterWidth(gutterWidth), _pageMargin(pageMargin)
{
    _bpp = p->numChannels() * Ptex::DataSize( p->dataType() );

    _txn = 0;

    int nf = p->numFaces();
    _blocks.clear();
    _blocks.resize( nf );

    for (int i=0; i<nf; ++i) {
        const Ptex::FaceInfo & f = p->getFaceInfo(i);
        _blocks[i].idx=i;
        _blocks[i].current=_blocks[i].native=f.res;
        _txn += f.res.u() * f.res.v();
    }

    _txc = _txn;
}

OsdPtexTextureLoader::~OsdPtexTextureLoader() 
{
    ClearPages();
}

unsigned long int
OsdPtexTextureLoader::GetNumBlocks( ) const {
    return (unsigned long int)_blocks.size();
}

unsigned long int
OsdPtexTextureLoader::GetNumPages( ) const {
    return (unsigned long int)_pages.size();
}

// attempt to re-size per-face resolutions to hit the uncompressed texel
// memory use requirement
void
OsdPtexTextureLoader::OptimizeResolution( unsigned long int memrec )
{
    unsigned long int txrec = memrec / _bpp;

    if (txrec==_txc)
        return;
    else {
        unsigned long int txcur = _txc;

        if (_blocks.size()==0)
            return;

        std::vector<block *> blocks( _blocks.size() );
        for (unsigned long int i=0; i<blocks.size(); ++i)
            blocks[i] = &(_blocks[i]);

        // reducing footprint ----------------------------------------
        if (txrec < _txc) {

            // blocks that have already been resized heavily will be considered last
            std::sort(blocks.begin(), blocks.end(), block::downsizePredicate );

            while ( (txcur>0) && (txcur>txrec) ) {

                unsigned long int txsaved = txcur;

                // start stealing from largest to smallest down
                for (int i=(int)blocks.size()-1; i>=0; --i) {

                    block * b = blocks[i];

                    // we have already hit rock bottom resolution... skip this block
                    if (b->current.ulog2==0 || b->current.vlog2==0)
                         continue;

                    unsigned short ures = (1<<(unsigned)(b->current.ulog2-1)),
                                   vres = (1<<(unsigned)(b->current.vlog2-1));

                    int diff = b->current.size() - ures * vres;

                    // we are about to overshoot the limit with our big blocks :
                    // skip until we find something smaller
                    if ( ((unsigned long int)diff>txcur) || ((txcur-diff)<txrec) )
                        continue;

                    b->current.ulog2--;
                    b->current.vlog2--;
                    txcur-=diff;
                }

                // couldn't scavenge anymore even from smallest faces : time to bail out.
                if (txsaved==txcur)
                    break;
            }
            _txc = txcur;
        } else {

        // increasing footprint --------------------------------------

            // blocks that have already been resized heavily will be considered first
            std::sort(blocks.begin(), blocks.end(), block::upsizePredicate );

            while ( (txcur < _txn) && (txcur < txrec) ) {

                unsigned long int txsaved = txcur;

                // start adding back to the largest faces first
                for (int i=0; i<(int)blocks.size(); ++i) {

                    block * b = blocks[i];

                    // already at native resolution... nothing to be done
                    if (b->current == b->native)
                        continue;

                    unsigned short ures = (1<<(unsigned)(b->current.ulog2+1)),
                                   vres = (1<<(unsigned)(b->current.vlog2+1));

                    int diff = ures * vres - b->current.size();

                    // we are about to overshoot the limit with our big blocks :
                    // skip until we find something smaller
                    if ( (txcur + diff) > txrec )
                        continue;

                    b->current.ulog2++;
                    b->current.vlog2++;
                    txcur+=diff;
                }

                // couldn't scavenge anymore even from smallest faces : time to bail out.
                if (txsaved==txcur)
                    break;
            }
            _txc = txcur;
        }
    }
}

// greedy packing of blocks into pages
void
OsdPtexTextureLoader::OptimizePacking( int maxnumpages )
{
    if (_blocks.size()==0)
        return;

    // generate a vector of pointers to the blocks -------------------
    std::vector<block *> blocks( _blocks.size() );
    for (unsigned long int i=0; i<blocks.size(); ++i)
        blocks[i] = &(_blocks[i]);

    // intro-sort blocks from largest to smallest (helps minimizing waste with
    // greedy packing)
    std::sort(blocks.rbegin(), blocks.rend(), block::currentAreaSort );

    // compute page size ---------------------------------------------
    // page size is set to the largest edge of the largest block : this is the
    // smallest possible page size, which should minimize the texels wasted on
    // the "last page" when the smallest blocks are being packed.
    _pagesize = 0;
    // also, find the max native edge length which will be used to allocate temporary
    // buffers of guttering
    for (unsigned long int i=0; i<blocks.size(); ++i) {
        _pagesize = std::max(_pagesize, (unsigned short)blocks[i]->current.u());
        _pagesize = std::max(_pagesize, (unsigned short)blocks[i]->current.v());
    }

    // note: at least 2*GUTTER_WIDTH of margin required for each page to fit
    _pagesize += (unsigned short)GetPageMargin();

    // grow the pagesize to make sure the optimization will not exceed the maximum
    // number of pages allowed
    for (int npages=_txc/(_pagesize*_pagesize); npages>maxnumpages; _pagesize<<=1)
        npages = _txc/(_pagesize*_pagesize );

    ClearPages( );

    // save some memory allocation time : guess the number of pages from the
    // number of texels
    _pages.reserve( _txc / (_pagesize*_pagesize) + 1 );

    // pack blocks into slots ----------------------------------------
    for (unsigned long int i=0, firstslot=0; i<_blocks.size(); ++i ) {

        block * b = blocks[i];

        // traverse existing pages for a suitable slot ---------------
        bool added=false;
        for( unsigned long int p=firstslot; p<_pages.size(); ++p )
            if( (added=_pages[p]->addBlock( b, GetGutterWidth() )) == true ) {
                break;
            }

        // if none was found : start new page
        if( !added ) {
            page * p = new page( _pagesize );
            p->addBlock(b, GetGutterWidth());
            _pages.push_back( p );
        }

        // adjust the page flag to the first page with open slots
        if( (_pages.size()>(firstslot+1)) &&
            (_pages[firstslot+1]->isFull()) )
            ++firstslot;
    }
}

// resample border texels for guttering
//
static int
resampleBorder(PtexTexture * ptex, int face, int edgeId, unsigned char *result,
               int dstLength, int bpp, float srcStart=0.0f, float srcEnd=1.0f)
{
    const Ptex::FaceInfo & pf = ptex->getFaceInfo(face);
    PtexFaceData * data = ptex->getData(face);

    int edgeLength = (edgeId==0||edgeId==2) ? pf.res.u() : pf.res.v();
    int srcOffset = (int)(srcStart*edgeLength);
    int srcLength = (int)((srcEnd-srcStart)*edgeLength);

    // if dstLength < 0, returns as original resolution without scaling
    if (dstLength < 0) dstLength = srcLength;

    unsigned char *border = new unsigned char[bpp*srcLength];

    // order of the result will be flipped to match adjacent pixel order
    for(int i=0;i<srcLength; ++i) {
        int u = 0, v = 0;
        if(edgeId==Ptex::e_bottom) {
            u = edgeLength-1-(i+srcOffset);
            v = 0;
        } else if(edgeId==Ptex::e_right) {
            u = pf.res.u()-1;
            v = edgeLength-1-(i+srcOffset);
        } else if(edgeId==Ptex::e_top) {
            u = i+srcOffset;
            v = pf.res.v()-1;
        } else if(edgeId==Ptex::e_left) {
            u = 0;
            v = i+srcOffset;
        }
        data->getPixel(u, v, &border[i*bpp]);
    }

    // nearest resample to fit dstLength
    for(int i=0;i<dstLength;++i) {
        for(int j=0; j<bpp; j++) {
            result[i*bpp+j] = border[(i*srcLength/dstLength)*bpp+j];
        }
    }

    delete[] border;

    return srcLength;
}

// flip order of pixel buffer
static void
flipBuffer(unsigned char *buffer, int length, int bpp)
{
    for(int i=0; i<length/2; ++i){
        for(int j=0; j<bpp; j++){
            std::swap(buffer[i*bpp+j], buffer[(length-1-i)*bpp+j]);
        }
    }
}

// sample neighbor face's edge
static void
sampleNeighbor(PtexTexture * ptex, unsigned char *border, int face, int edge, int length, int bpp)
{
    const Ptex::FaceInfo &fi = ptex->getFaceInfo(face);

    // copy adjacent borders
    int adjface = fi.adjface(edge);
    if(adjface != -1) {
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
            resampleBorder(ptex, adjface, ae, border+(length/2*bpp), length/2, bpp);

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
            if(f == adjface && e == ae) // case 1
                resampleBorder(ptex, adjface, ae, border, length, bpp, 0.0, 0.5);
            else  // case 2
                resampleBorder(ptex, adjface, ae, border, length, bpp, 0.5, 1.0);

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
              int face, int edge, int bpp, unsigned char *lineBuffer)
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
    int uv[4][2] = {{0,0}, {1,0}, {1,1}, {0,1}};
    float *pixel = (float*)alloca(sizeof(float)*numchannels);
    float *accumPixel = (float*)alloca(sizeof(float)*numchannels);

    // clear accum pixel
    memset(accumPixel, 0, sizeof(float)*numchannels);

    bool clockWise = true;
    int valence = 0;
    do {
        valence++;
        
        if (valence > 255) {
            OsdWarning("High valence detected in %s : invalid adjacency around "
                       "face %d", ptex->path(), face);
            break;
        }
        
        Ptex::FaceInfo info = ptex->getFaceInfo(currentFace);
        ptex->getPixel(currentFace,
                        uv[currentEdge][0] * (info.res.u()-1),
                        uv[currentEdge][1] * (info.res.v()-1),
                        pixel, 0, numchannels);
        for (int j = 0; j < numchannels; ++j) {
            accumPixel[j] += pixel[j];
            if (valence == 3) {
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
    } while(currentFace != face);

    if (valence == 4) {
        return true;
    }

    // non-4 valence. let's average and return false;
    for (int j = 0; j < numchannels; ++j) {
        resultPixel[j] = accumPixel[j]/valence;
    }
    return false;
}

// sample neighbor pixels and populate around blocks
static void
guttering(PtexTexture *_ptex, OsdPtexTextureLoader::block *b, unsigned char *pptr,
          int bpp, int pagesize, int stride, int gwidth)
{
    unsigned char * lineBuffer = new unsigned char[pagesize * bpp];

    for(int w=0; w<gwidth; ++w) {
        for(int edge=0; edge<4; edge++) {

            int len = (edge==0 or edge==2) ? b->current.u() : b->current.v();
            // XXX: for now, sample same edge regardless of gutter depth
            sampleNeighbor(_ptex, lineBuffer, b->idx, edge, len, bpp);

            unsigned char *s = lineBuffer, *d;
            for(int j=0;j<len;++j) {
                d = pptr;
                switch(edge) {
                case Ptex::e_bottom:
                    d += stride*(b->v-1-w) + bpp*(b->u+j);
                    break;
                case Ptex::e_right:
                    d += stride*(b->v+j) + bpp*(b->u+b->current.u()+w);
                    break;
                case Ptex::e_top:
                    d += stride*(b->v+b->current.v()+w) + bpp*(b->u+len-j-1);
                    break;
                case Ptex::e_left:
                    d += stride*(b->v+len-j-1) + bpp*(b->u-1-w);
                    break;
                }
                for(int k=0; k<bpp; k++)
                    *d++ = *s++;
            }
        }
    }

    // fix corner pixels
    int numchannels = _ptex->numChannels();
    float *accumPixel = new float[numchannels];
    int uv[4][2] = {{-1,-1}, {1,-1}, {1,1}, {-1,1}};
    for (int edge=0; edge<4; edge++) {

        int du = (b->u+gwidth*uv[edge][0]);
        int dv = (b->v+gwidth*uv[edge][1]);

        /*  There are 3 cases when filling a corner pixel on gutter.
            
            case 1: Regular 4 valence
                    We already have correct 'B' and 'C' pixels by edge resampling above.
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
                    If the current face comes from non-quad root face, there could be a T-vertex
                    on its corner. Just like case 1, need to fill border corner with pixel 'D'.
               +-----+-----+
               |     |     |<-current
               |    B|A    |
               |     *-----+
               |    D|C    |
               |     |     |
               +-----+-----+

            case 3: Other than 4 valence case (everything else, including boundary)
                    Since guttering pixels are placed on the border of each ptex faces,
                    It's not possible to store more than 4 pixels at a coner for a reasonable
                    interpolation.
                    In this case, we need to average all corner pixels and overwrite with an
                    averaged value, so that every face vertex picks the same value.
               +---+---+
               |   |   |<-current
               |  B|A  |
               +---*---|
               | D/E\C |
               | /   \ |
               |/     \| 
               +-------+
         */

        if (getCornerPixel(_ptex, accumPixel, numchannels, b->idx, edge, bpp, lineBuffer)) {
            // case 1 and case 2
            if (edge==1||edge==2) du += b->current.u()-gwidth;
            if (edge==2||edge==3) dv += b->current.v()-gwidth;
            for (int u=0; u<gwidth; ++u) {
                for (int v=0; v<gwidth; ++v) {
                    unsigned char *d = pptr + (dv+u)*stride + (du+v)*bpp;
                    Ptex::ConvertFromFloat(d, accumPixel, _ptex->dataType(), numchannels);
                }
            }
        } else {
            // case 3
            if (edge==1||edge==2) du += b->current.u()-gwidth-1;
            if (edge==2||edge==3) dv += b->current.v()-gwidth-1;
            // set accumPixel to 4 corners
            // .. over (gwidth+1)x(gwidth+1) pixels for each corner
            for (int u=0; u<=gwidth; ++u) {
                for (int v=0; v<=gwidth; ++v) {
                    unsigned char *d = pptr + (dv+u)*stride + (du+v)*bpp;
                    Ptex::ConvertFromFloat(d, accumPixel, _ptex->dataType(), numchannels);
                }
            }
        }
    }
    delete[] lineBuffer;
    delete[] accumPixel;
}

// prepares the data for the texture samplers used by the GLSL tables to render
// PTex texels
bool
OsdPtexTextureLoader::GenerateBuffers( )
{
    if (_pages.size()==0) return false;

    // populate the page index lookup texture ------------------------
    _indexBuffer = new unsigned int[ _blocks.size() ];
    for (unsigned long int i=0; i<_pages.size(); ++i) {
        page * p = _pages[i];
        for (page::blist::iterator j=p->blocks.begin(); j!=p->blocks.end(); ++j)
            _indexBuffer[ (*j)->idx ] = i;
    }

    // populate the layout lookup texture ----------------------------
    float * lptr = _layoutBuffer = new float[ 4 * _blocks.size() ];
    for (unsigned long int i=0; i<_blocks.size(); ++ i) {
        // normalize coordinates by pagesize resolution !
        *lptr++ = (float) _blocks[i].u / (float) _pagesize;
        *lptr++ = (float) _blocks[i].v / (float) _pagesize;
        *lptr++ = (float) _blocks[i].current.u() / (float) _pagesize;
        *lptr++ = (float) _blocks[i].current.v() / (float) _pagesize;
    }

    // populate the texels -------------------------------------------
    int stride = _bpp * _pagesize,
        pagestride = stride * _pagesize;

    unsigned char * pptr = _texelBuffer = new unsigned char[ pagestride * _pages.size() ];

    for (unsigned long int i=0; i<_pages.size(); i++) {

        page * p = _pages[i];

        for (page::blist::iterator b=p->blocks.begin(); b!=p->blocks.end(); ++b) {
            _ptex->getData( (*b)->idx, pptr + stride*(*b)->v + _bpp*(*b)->u, stride, (*b)->current );

            if(GetGutterWidth() > 0)
                guttering(_ptex, *b, pptr, _bpp, _pagesize, stride, GetGutterWidth());
        }

        pptr += pagestride;
    }

    return true;
}

void
OsdPtexTextureLoader::ClearBuffers( )
{   delete [] _indexBuffer;
    delete [] _layoutBuffer;
    delete [] _texelBuffer;
}

// returns a ratio of texels wasted in the final GPU texture : anything under 5%
// is pretty good compared to our previous solution...
float
OsdPtexTextureLoader::EvaluateWaste( ) const
{
    unsigned long int wasted=0;
    for( unsigned long int i=0; i<_pages.size(); i++ ) {
        page * p = _pages[i];
        for( page::slist::iterator s=p->slots.begin(); s!=p->slots.end(); ++s )
            wasted += s->ures * s->vres;
    }
    return (float)((double)wasted/(double)_txc);
}

void
OsdPtexTextureLoader::ClearPages( )
{   for( unsigned long int i=0; i<_pages.size(); i++ )
        delete _pages[i];
    _pages.clear();
}

void
OsdPtexTextureLoader::PrintBlocks() const
{ for( unsigned long int i=0; i<_blocks.size(); ++i )
    std::cout<<_blocks[i]<<std::endl;
}

void
OsdPtexTextureLoader::PrintPages() const
{ for( unsigned long int i=0; i<_pages.size(); ++i )
    std::cout<<*(_pages[i])<<std::endl;
}

std::ostream & operator <<(std::ostream &s, const OsdPtexTextureLoader::block & b)
{ s<<"block "<<b.idx<<" = { ";
  s<<"native=("<<b.native.u()<<","<<b.native.v()<<") ";
  s<<"current=("<<b.current.u()<<","<<b.current.v()<<") ";
  s<<"}";
  return s;
}

std::ostream & operator <<(std::ostream &s, const OsdPtexTextureLoader::page & p)
{
  s<<"page {\n";
  s<<"        slots {";
  for (OsdPtexTextureLoader::page::slist::const_iterator i=p.slots.begin(); i!=p.slots.end(); ++i)
    s<<" { "<<i->u<<" "<<i->v<<" "<<i->ures<<" "<<i->vres<<"} ";
  s<<"        }\n";

  s<<"        blocks {";
  for (OsdPtexTextureLoader::page::blist::const_iterator i=p.blocks.begin(); i!=p.blocks.end(); ++i)
    s<<" "<< **i;
  s<<"        }\n";

  s<<"}";
  return s;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

