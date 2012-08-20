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

#if not defined(__APPLE__)
    #include <GL/glew.h>
#else
    #include <OpenGL/gl3.h>
#endif

#include <Ptexture.h>

#include <string.h>
#include <list>

#include "../osd/local.h"
#include "../osd/pTexture.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

int OsdPTexture::_gutterWidth = 0;
int OsdPTexture::_pageMargin = 0;
int OsdPTexture::_gutterDebug = 0;

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
struct block {

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
struct page {

    //----------------------------------------------------------------
    // slot : rectangular block of available texels in a page
    struct slot {
        GLushort u, v, ures, vres;

        slot(  GLushort size ) : u(0), v(0), ures(size), vres(size) { }

        slot( GLushort iu, GLushort iv,  GLushort iures, GLushort ivres ) :
              u(iu), v(iv), ures(iures), vres(ivres) { }

        // true if a block can fit in this slot
        bool fits( block const * b ) {
            return ( (b->current.u()+2*OsdPTexture::GetGutterWidth())<=ures ) &&
                ((b->current.v()+2*OsdPTexture::GetGutterWidth())<=vres);
        }
    };

    //----------------------------------------------------------------
    typedef std::list<block *> blist;
    blist blocks;

    typedef std::list<slot> slist;
    slist slots;

    // construct a page with a single empty slot the size of the page
    page( GLushort pagesize ) {
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
    bool addBlock( block * b ) {
        for (slist::iterator i=slots.begin(); i!=slots.end(); ++i) {

            if (i->fits( b )) {

                blocks.push_back( b );

                int w = OsdPTexture::GetGutterWidth();

                b->u=i->u + w;
                b->v=i->v + w;

                // add new slot to the right
                if (i->ures > (b->current.u()+2*w)) {
                    slots.push_front( slot( i->u+b->current.u()+2*w,
                                            i->v,
                                            i->ures-b->current.u()-2*w,
                                            b->current.v()+2*w));
                }

                // add new slot to the bottom
                if (i->vres > (b->current.v()+2*w)) {
                    slots.push_back( slot( i->u,
                                           i->v+b->current.v()+2*w,
                                           i->ures,
                                           i->vres-b->current.v()-2*w ));
                }

                slots.erase( i );
                return true;
            }
        }
        return false;
    }

    friend std::ostream & operator <<(std::ostream &s, const page & p);
};

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
class loader {
public:
    loader( PtexTexture * );

    ~loader( ) {
        ClearPages();
    }

    const GLushort GetPageSize( ) const {
        return _pagesize;
    }

    const unsigned long int GetNumBlocks( ) const {
        return (unsigned long int)_blocks.size();
    }

    const unsigned long int GetNumPages( ) const {
        return (unsigned long int)_pages.size();
    }

    const GLuint * GetIndexBuffer( ) const {
        return _indexBuffer;
    }

    const GLfloat * GetLayoutBuffer( ) const {
        return _layoutBuffer;
    }

    const GLubyte * GetTexelBuffer( ) const {
        return _texelBuffer;
    }

    unsigned long int GetUncompressedSize() const {
        return _txc * _bpp;
    }

    unsigned long int GetNativeUncompressedSize() const {
        return _txn * _bpp;
    }

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
    GLushort            _pagesize;

    GLuint *  _indexBuffer;
    GLfloat * _layoutBuffer;
    GLubyte * _texelBuffer;
};


loader::loader( PtexTexture * p ) :
        _ptex(p), _indexBuffer( NULL ), _layoutBuffer( NULL ), _texelBuffer(NULL)
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

// attempt to re-size per-face resolutions to hit the uncompressed texel
// memory use requirement
void
loader::OptimizeResolution( unsigned long int memrec )
{
    unsigned long int txrec = memrec / _bpp;

    if (txrec==_txc)
        return;
    else
    {
        unsigned long int txcur = _txc;

        if (_blocks.size()==0)
            return;

        std::vector<block *> blocks( _blocks.size() );
        for (unsigned long int i=0; i<blocks.size(); ++i)
            blocks[i] = &(_blocks[i]);

        // reducing footprint ----------------------------------------
        if (txrec < _txc)
        {
            // blocks that have already been resized heavily will be considered last
            std::sort(blocks.begin(), blocks.end(), block::downsizePredicate );

            while ( (txcur>0) && (txcur>txrec) )
            {
                unsigned long int txsaved = txcur;

                // start stealing from largest to smallest down
                for (int i=(int)blocks.size()-1; i>=0; --i)
                {
                    block * b = blocks[i];

                    // we have already hit rock bottom resolution... skip this block
                    if (b->current.ulog2==0 || b->current.vlog2==0)
                         continue;

                    GLushort ures = (1<<(unsigned)(b->current.ulog2-1)),
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

            while ( (txcur < _txn) && (txcur < txrec) )
            {
                unsigned long int txsaved = txcur;

                // start adding back to the largest faces first
                for (int i=0; i<(int)blocks.size(); ++i)
                {
                    block * b = blocks[i];

                    // already at native resolution... nothing to be done
                    if (b->current == b->native)
                        continue;

                    GLushort ures = (1<<(unsigned)(b->current.ulog2+1)),
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
loader::OptimizePacking( int maxnumpages )
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
    _pagesize = blocks[0]->current.ulog2 > blocks[0]->current.vlog2 ?
                blocks[0]->current.u() : blocks[0]->current.v();

    // at least 2*GUTTER_WIDTH of margin required for each page to fit
    _pagesize += OsdPTexture::GetPageMargin();

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
            if( (added=_pages[p]->addBlock( b )) ) {
                break;
            }

        // if none was found : start new page
        if( !added ) {
            page * p = new page( _pagesize );
            p->addBlock(b);
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
static void
resampleBorder(PtexTexture * ptex, int face, int edgeId, GLubyte *result, int edge,
               int dstLength, int bpp, float srcStart=0.0f, float srcEnd=1.0f)
{
    const Ptex::FaceInfo & pf = ptex->getFaceInfo(face);
    PtexFaceData * data = ptex->getData(face);

    int edgeLength = (edgeId==0||edgeId==2) ? pf.res.u() : pf.res.v();
    int srcOffset = (int)(srcStart*edgeLength);
    int srcLength = (int)((srcEnd-srcStart)*edgeLength);

    GLubyte *border = new GLubyte[bpp*srcLength];

    // order of the result will be flipped to match adjacent pixel order
    for(int i=0;i<srcLength; ++i) {
        int u, v;
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

    // gutter visualize debug
    if (OsdPTexture::GetGutterDebug()) {
        float debugColors[4][4] = { {1, 0, 0, 0},    // bottom = red
                                    {0, 1, 0, 0},    // right  = green
                                    {0, 0, 1, 0},    // left   = blue
                                    {1, 1, 0, 0} };  // top    = yellow
        for(int i=0;i<dstLength;++i){
            float *fb = (float*)result;
            Ptex::ConvertFromFloat(result+i*bpp, debugColors[edge], ptex->dataType(), 4);
        }
    }

    delete[] border;
}

// flip order of pixel buffer
static void
flipBuffer(GLubyte *buffer, int length, int bpp)
{
    for(int i=0; i<length/2; ++i){
        for(int j=0; j<bpp; j++){
            std::swap(buffer[i*bpp+j], buffer[(length-1-i)*bpp+j]);
        }
    }
}

// sample neighbor face's edge
static void
sampleNeighbor(PtexTexture * ptex, GLubyte *border, int face, int edge, int length, int bpp)
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
            resampleBorder(ptex, adjface, ae, border, edge, length/2, bpp);
            const Ptex::FaceInfo &sfi1 = ptex->getFaceInfo(adjface);
            adjface = sfi1.adjface((ae+3)%4);
            const Ptex::FaceInfo &sfi2 = ptex->getFaceInfo(adjface);
            ae = (sfi1.adjedge((ae+3)%4)+3)%4;
            resampleBorder(ptex, adjface, ae, border+(length/2*bpp), edge, length/2, bpp);

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
                resampleBorder(ptex, adjface, ae, border, edge, length, bpp, 0.0, 0.5);
            else  // case 2
                resampleBorder(ptex, adjface, ae, border, edge, length, bpp, 0.5, 1.0);

        } else {
            /*  ordinary case (1:1 match)
                +------------------+
                |       face       |
                +--------edge------+
                |    adj face      |
                +----------+-------+
            */
            resampleBorder(ptex, adjface, ae, border, edge, length, bpp);
        }
    } else {
        /* border edge. duplicate itself
           +-----------------+
           |       face      |
           +-------edge------+
        */
        resampleBorder(ptex, face, edge, border, edge, length, bpp);
        flipBuffer(border, length, bpp);
    }
}

// average corner pixels by traversing all adjacent faces around vertex
//
static bool
averageCorner(PtexTexture *ptex, float *accumPixel, int numchannels, int face, int edge)
{
    const Ptex::FaceInfo &fi = ptex->getFaceInfo(face);

    int adjface = fi.adjface(edge);

    // don't average T-vertex.
    if (fi.isSubface() && !ptex->getFaceInfo(adjface).isSubface())
        return false;

    int valence = 0;
    int currentFace = face;
    int currentEdge = edge;
    int uv[4][2] = {{0,0}, {1,0}, {1,1}, {0,1}};
    float *pixel = (float*)alloca(sizeof(float)*numchannels);

    // clear result buffer
    memset(accumPixel, 0, sizeof(float)*numchannels);

    do {
        valence++;
        Ptex::FaceInfo info = ptex->getFaceInfo(currentFace);
        ptex->getPixel(currentFace,
                        uv[currentEdge][0] * (info.res.u()-1),
                        uv[currentEdge][1] * (info.res.v()-1),
                        pixel, 0, numchannels);
        for(int j=0; j<numchannels; ++j) {
            accumPixel[j] += pixel[j];
        }

        // next face
        currentFace = info.adjface(currentEdge);
        currentEdge = info.adjedge(currentEdge);
        currentEdge = (currentEdge+1)%4;
    } while(currentFace != -1 && currentFace != face);

    for(int j=0; j<numchannels; ++j) {
        accumPixel[j] /= valence;
    }

    return true;
}

// sample neighbor pixels and populate around blocks
static void
guttering(PtexTexture *_ptex, block *b, GLubyte *pptr, int _bpp, int _pagesize, int stride, int gwidth)
{
    const Ptex::FaceInfo &fi = _ptex->getFaceInfo(b->idx);
    GLubyte * border = new GLubyte[_pagesize * _bpp];

    for(int w=0; w<gwidth; ++w) {
        for(int edge=0; edge<4; edge++) {

            int len = (edge==0 or edge==2) ? b->current.u() : b->current.v();
            // XXX: for now, sample same edge regardless of gutter depth
            sampleNeighbor(_ptex, border, b->idx, edge, len, _bpp);

            GLubyte *s = border, *d;
            for(int j=0;j<len;++j) {
                d = pptr;
                switch(edge) {
                case Ptex::e_bottom:
                    d += stride*(b->v-1-w) + _bpp*(b->u+j);
                    break;
                case Ptex::e_right:
                    d += stride*(b->v+j) + _bpp*(b->u+b->current.u()+w);
                    break;
                case Ptex::e_top:
                    d += stride*(b->v+b->current.v()+w) + _bpp*(b->u+len-j-1);
                    break;
                case Ptex::e_left:
                    d += stride*(b->v+len-j-1) + _bpp*(b->u-1-w);
                    break;
                }
                for(int k=0; k<_bpp; k++)
                    *d++ = *s++;
            }
        }
    }
    delete[] border;

    // average corner pixels
    int numchannels = _ptex->numChannels();
    float *accumPixel = new float[numchannels];
    int uv[4][2] = {{-1,-1}, {1,-1}, {1,1}, {-1,1}};
    for(int edge=0; edge<4; edge++) {

        if(averageCorner(_ptex, accumPixel, numchannels, b->idx, edge)) {
            // set accumPixel to 4 corner
            int du = (b->u+gwidth*uv[edge][0]);
            int dv = (b->v+gwidth*uv[edge][1]);
            if(edge==1||edge==2) du += b->current.u()-gwidth-1;
            if(edge==2||edge==3) dv += b->current.v()-gwidth-1;
            // .. over (gwidth+1)x(gwidth+1) pixels for each corner
            for(int u=0; u<=gwidth; ++u) {
                for(int v=0; v<=gwidth; ++v) {
                    GLubyte *d = pptr + (dv+u)*stride + (du+v)*_bpp;
                    Ptex::ConvertFromFloat(d, accumPixel, _ptex->dataType(), numchannels);
                }
            }
        }
    }
    delete[] accumPixel;
}

// prepares the data for the texture samplers used by the GLSL tables to render
// PTex texels
bool
loader::GenerateBuffers( )
{
    if (_pages.size()==0) return false;

    // populate the page index lookup texture ------------------------
    _indexBuffer = new GLuint[ _blocks.size() ];
    for (unsigned long int i=0; i<_pages.size(); ++i) {
        page * p = _pages[i];
        for (page::blist::iterator j=p->blocks.begin(); j!=p->blocks.end(); ++j)
            _indexBuffer[ (*j)->idx ] = i;
    }

    // populate the layout lookup texture ----------------------------
    GLfloat * lptr = _layoutBuffer = new GLfloat[ 4 * _blocks.size() ];
    for (unsigned long int i=0; i<_blocks.size(); ++ i) {
        // normalize coordinates by pagesize resolution !
        *lptr++ = (GLfloat) _blocks[i].u / (GLfloat) _pagesize;
        *lptr++ = (GLfloat) _blocks[i].v / (GLfloat) _pagesize;
        *lptr++ = (GLfloat) _blocks[i].current.u() / (GLfloat) _pagesize;
        *lptr++ = (GLfloat) _blocks[i].current.v() / (GLfloat) _pagesize;
    }

    // populate the texels -------------------------------------------
    int stride = _bpp * _pagesize,
        pagestride = stride * _pagesize;

    GLubyte * pptr = _texelBuffer = new GLubyte[ pagestride * _pages.size() ];

    for (unsigned long int i=0; i<_pages.size(); i++) {

        page * p = _pages[i];

        for (page::blist::iterator b=p->blocks.begin(); b!=p->blocks.end(); ++b) {
            _ptex->getData( (*b)->idx, pptr + stride*(*b)->v + _bpp*(*b)->u, stride, (*b)->current );

            if(OsdPTexture::GetGutterWidth() > 0)
                guttering(_ptex, *b, pptr, _bpp, _pagesize, stride, OsdPTexture::GetGutterWidth());
        }

        pptr += pagestride;
    }

    return true;
}

void
loader::ClearBuffers( )
{   delete [] _indexBuffer;
    delete [] _layoutBuffer;
    delete [] _texelBuffer;
}

// returns a ratio of texels wasted in the final GPU texture : anything under 5%
// is pretty good compared to our previous solution...
float
loader::EvaluateWaste( ) const
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
loader::ClearPages( )
{   for( unsigned long int i=0; i<_pages.size(); i++ )
        delete _pages[i];
    _pages.clear();
}

void
loader::PrintBlocks() const
{ for( unsigned long int i=0; i<_blocks.size(); ++i )
    std::cout<<_blocks[i]<<std::endl;
}

void
loader::PrintPages() const
{ for( unsigned long int i=0; i<_pages.size(); ++i )
    std::cout<<*(_pages[i])<<std::endl;
}

std::ostream & operator <<(std::ostream &s, const block & b)
{ s<<"block "<<b.idx<<" = { ";
  s<<"native=("<<b.native.u()<<","<<b.native.v()<<") ";
  s<<"current=("<<b.current.u()<<","<<b.current.v()<<") ";
  s<<"}";
  return s;
}

std::ostream & operator <<(std::ostream &s, const page & p)
{
  s<<"page {\n";
  s<<"        slots {";
  for (page::slist::const_iterator i=p.slots.begin(); i!=p.slots.end(); ++i)
    s<<" { "<<i->u<<" "<<i->v<<" "<<i->ures<<" "<<i->vres<<"} ";
  s<<"        }\n";

  s<<"        blocks {";
  for (page::blist::const_iterator i=p.blocks.begin(); i!=p.blocks.end(); ++i)
    s<<" "<< **i;
  s<<"        }\n";

  s<<"}";
  return s;
}

OsdPTexture::~OsdPTexture()
{
    // delete pages lookup ---------------------------------
    if (glIsTexture(_pages))
       glDeleteTextures(1,&_pages);

    // delete layout lookup --------------------------------
    if (glIsTexture(_layout))
       glDeleteTextures(1,&_layout);

    // delete textures lookup ------------------------------
    if (glIsTexture(_texels))
       glDeleteTextures(1,&_texels);
}

static GLuint genTextureBuffer( GLenum format, GLsizeiptr size, GLvoid const * data ) {

    GLuint buffer, result;
    glGenBuffers(1, & buffer );
    glBindBuffer( GL_TEXTURE_BUFFER, buffer );
    glBufferData( GL_TEXTURE_BUFFER, size, data, GL_STATIC_DRAW);

    glGenTextures(1, & result);
    glBindTexture( GL_TEXTURE_BUFFER, result);
    glTexBuffer( GL_TEXTURE_BUFFER, format, buffer);

    glDeleteBuffers(1,&buffer);

    return result;
}

OsdPTexture::OsdPTexture()
    : _width(0), _height(0), _depth(0),_pages(0), _layout(0), _texels(0)
{ }

OsdPTexture *
OsdPTexture::Create( PtexTexture * reader, unsigned long int targetMemory ) {
    OsdPTexture * result=NULL;

    // Read the ptexture data and pack the texels
    loader ldr( reader );

    unsigned long int nativeSize = ldr.GetNativeUncompressedSize(),
           targetSize = targetMemory;

    if (targetSize!=0 && targetSize!=nativeSize)
        ldr.OptimizeResolution( targetSize );

    GLint maxnumpages = 0;
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxnumpages);

    ldr.OptimizePacking( maxnumpages );

    if (!ldr.GenerateBuffers( ))
        return result;

    // Setup GPU memory
    unsigned long int nfaces = ldr.GetNumBlocks();

    GLuint pages = genTextureBuffer( GL_R32I,
                                     nfaces * sizeof(GLint),
                                     ldr.GetIndexBuffer() );

    GLuint layout = genTextureBuffer( GL_RGBA32F,
                                      nfaces * 4 * sizeof(GLfloat),
                                      ldr.GetLayoutBuffer() );

    GLenum format, type;
    switch(reader->dataType())
    {
        case Ptex::dt_uint16 : type = GL_UNSIGNED_SHORT; break;
        case Ptex::dt_float  : type = GL_FLOAT; break;
        case Ptex::dt_half   : type = GL_HALF_FLOAT; break;
        default              : type = GL_UNSIGNED_BYTE; break;
    }

    switch(reader->numChannels())
    {
        case 1 : format = GL_RED; break;
        case 2 : format = GL_RG; break;
        case 3 : format = GL_RGB; break;
        case 4 : format = GL_RGBA; break;
        default: format = GL_RED; break;
    }

    // actual texels texture array
    GLuint texels;
    glGenTextures(1,&texels);
    glBindTexture(GL_TEXTURE_2D_ARRAY,texels);

    // XXXX for the time being, filtering is off - once cross-patch filtering
    // is in place, we will use glGenSamplers to dynamically access these settings.
    if (GetGutterWidth() > 0) {
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);

    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0,
                     (type==GL_FLOAT) ? GL_RGBA32F : GL_RGBA,
                     ldr.GetPageSize(),
                     ldr.GetPageSize(),
                     ldr.GetNumPages(),
                     0, format, type,
                     ldr.GetTexelBuffer());

    if (GLuint err = glGetError()) {
        printf("(OsdPtexture::Create) GL error %x :", err);
        return result;
    }

    ldr.ClearBuffers( );

    // Return the Osd Ptexture object
    result = new OsdPTexture;

    result->_width = ldr.GetPageSize();
    result->_height = ldr.GetPageSize();
    result->_depth = ldr.GetNumPages();

    result->_format = format;

    result->_pages = pages;
    result->_layout = layout;
    result->_texels = texels;

    return result;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

