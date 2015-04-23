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

#include <cassert>
#include <cstdio>


#include "../../regression/common/hbr_utils.h"
#include "../../regression/common/vtr_utils.h"

#include "init_shapes.h"

//
// Regression testing matching Far to Hbr (default CPU implementation)
//
// Notes:
// - precision is currently held at 1e-6
//
// - results cannot be bitwise identical as some vertex interpolations
//   are not happening in the same order.
//
// - only vertex interpolation is being tested at the moment.
//
#define PRECISION 1e-6

static bool g_debugmode = false;

//------------------------------------------------------------------------------
// Vertex class implementation
struct xyzVV {

    xyzVV() { /* _pos[0]=_pos[1]=_pos[2]=0.0f; */ }

    xyzVV( int /*i*/ ) { }

    xyzVV( float x, float y, float z ) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    xyzVV( const xyzVV & src ) { _pos[0]=src._pos[0]; _pos[1]=src._pos[1]; _pos[2]=src._pos[2]; }

   ~xyzVV( ) { }

    void AddWithWeight(const xyzVV& src, float weight) {
        _pos[0]+=weight*src._pos[0];
        _pos[1]+=weight*src._pos[1];
        _pos[2]+=weight*src._pos[2];
    }

    void AddVaryingWithWeight(const xyzVV& , float) { }

    void Clear( void * =0 ) { _pos[0]=_pos[1]=_pos[2]=0.0f; }

    void SetPosition(float x, float y, float z) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    void ApplyVertexEdit(const OpenSubdiv::HbrVertexEdit<xyzVV> & edit) {
        const float *src = edit.GetEdit();
        switch(edit.GetOperation()) {
          case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Set:
            _pos[0] = src[0];
            _pos[1] = src[1];
            _pos[2] = src[2];
            break;
          case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Add:
            _pos[0] += src[0];
            _pos[1] += src[1];
            _pos[2] += src[2];
            break;
          case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Subtract:
            _pos[0] -= src[0];
            _pos[1] -= src[1];
            _pos[2] -= src[2];
            break;
        }
    }

    void ApplyMovingVertexEdit(const OpenSubdiv::HbrMovingVertexEdit<xyzVV> &) { }

    const float * GetPos() const { return _pos; }

    bool operator==(xyzVV const & other) const {
        if (_pos[0]==other._pos[0] and
            _pos[1]==other._pos[1] and
            _pos[2]==other._pos[2]) {
            return true;
        }
        return false;
    }

private:
    float _pos[3];
};

//------------------------------------------------------------------------------
typedef OpenSubdiv::HbrMesh<xyzVV>           Hmesh;
typedef OpenSubdiv::HbrFace<xyzVV>           Hface;
typedef OpenSubdiv::HbrVertex<xyzVV>         Hvertex;
typedef OpenSubdiv::HbrHalfedge<xyzVV>       Hhalfedge;

static Hmesh *
interpolateHbrVertexData(ShapeDesc const & desc, int maxlevel) {

    // Hbr interpolation
    Hmesh * hmesh = simpleHbr<xyzVV>(desc.data.c_str(), desc.scheme, /*verts vector*/ 0, /*fvar*/ false);
    assert(hmesh);

    for (int level=0, firstface=0; level<maxlevel; ++level ) {
        int nfaces = hmesh->GetNumFaces();
        for (int i=firstface; i<nfaces; ++i) {

            Hface * f = hmesh->GetFace(i);
            assert(f->GetDepth()==level);
            if (not f->IsHole()) {
                f->Refine();
            }
        }
        // Hbr allocates faces sequentially, skip faces that have already been refined.
        firstface = nfaces;
    }

    return hmesh;
}

//------------------------------------------------------------------------------
typedef OpenSubdiv::Far::TopologyRefiner               FarTopologyRefiner;
typedef OpenSubdiv::Far::TopologyRefinerFactory<Shape> FarTopologyRefinerFactory;

static FarTopologyRefiner *
interpolateVtrVertexData(ShapeDesc const & desc, int maxlevel, std::vector<xyzVV> & data) {

    // Vtr interpolation
    Shape * shape = Shape::parseObj(desc.data.c_str(), desc.scheme);

    FarTopologyRefiner * refiner =
        FarTopologyRefinerFactory::Create(*shape,
            FarTopologyRefinerFactory::Options(GetSdcType(*shape), GetSdcOptions(*shape)));
    assert(refiner);

    FarTopologyRefiner::UniformOptions options(maxlevel);
    options.fullTopologyInLastLevel=true;
    refiner->RefineUniform(options);

    // populate coarse mesh positions
    data.resize(refiner->GetNumVerticesTotal());
    for (int i=0; i<refiner->GetNumVertices(0); i++) {
        data[i].SetPosition(shape->verts[i*3+0],
                            shape->verts[i*3+1],
                            shape->verts[i*3+2]);
    }

    xyzVV * verts = &data[0];
    refiner->Interpolate(verts, verts+refiner->GetNumVertices(0));

    delete shape;
    return refiner;
}

//------------------------------------------------------------------------------
#ifdef foo
static void
printVertexData(std::vector<xyzVV> const & hbrBuffer, std::vector<xyzVV> const & vtrBuffer) {

    assert(hbrBuffer.size()==vtrBuffer.size());
    for (int i=0; i<(int)hbrBuffer.size(); ++i) {

        float const * hbr = hbrBuffer[i].GetPos(),
                    * vtr = vtrBuffer[i].GetPos();

        printf("%3d %d (%f %f %f) (%f %f %f)\n", i, hbrBuffer[i]==vtrBuffer[i],
                                                    hbr[0], hbr[1], hbr[2],
                                                    vtr[0], vtr[1], vtr[2]);
    }
}
#endif
//------------------------------------------------------------------------------
struct Mapper {

    struct LevelMap {
        std::vector<Hface *>     faces;
        std::vector<Hhalfedge *> edges;
        std::vector<Hvertex *>   verts;
    };

    std::vector<LevelMap> maps;

    Mapper(FarTopologyRefiner * refiner, Hmesh * hmesh) {

        assert(refiner and hmesh);

        maps.resize(refiner->GetMaxLevel()+1);

        typedef OpenSubdiv::Far::Index Index;
        typedef OpenSubdiv::Far::ConstIndexArray ConstIndexArray;

        {   // Populate base level
            // note : topological ordering is identical between Hbr and Vtr for the
            // base level

            int nfaces = refiner->GetNumFaces(0),
                nedges = refiner->GetNumEdges(0),
                nverts = refiner->GetNumVertices(0);

            maps[0].faces.resize(nfaces, 0);
            maps[0].edges.resize(nedges, 0);
            maps[0].verts.resize(nverts, 0);

            for (int face=0; face<nfaces; ++face) {
                maps[0].faces[face] = hmesh->GetFace(face);
            }

            for (int edge = 0; edge <nedges; ++edge) {

                ConstIndexArray vtrVerts = refiner->GetEdgeVertices(0, edge);

                Hvertex const * v0 = hmesh->GetVertex(vtrVerts[0]),
                              * v1 = hmesh->GetVertex(vtrVerts[1]);

                Hhalfedge * e = v0->GetEdge(v1);
                if (not e) {
                    e = v1->GetEdge(v0);
                }
                assert(e);

                maps[0].edges[edge] = e;
            }

            for (int vert = 0; vert<nverts; ++vert) {
                maps[0].verts[vert] = hmesh->GetVertex(vert);
            }
        }

        // Populate refined levels
        for (int level=1, ecount=0; level<=refiner->GetMaxLevel(); ++level) {

            LevelMap & previous = maps[level-1],
                     & current = maps[level];

            current.faces.resize(refiner->GetNumFaces(level), 0);
            current.edges.resize(refiner->GetNumEdges(level), 0);
            current.verts.resize(refiner->GetNumVertices(level), 0);

            for (int face=0; face < refiner->GetNumFaces(level-1); ++face) {

                // populate child faces
                Hface * f = previous.faces[face];

                ConstIndexArray childFaces = refiner->GetFaceChildFaces(level-1, face);
                assert(childFaces.size()==f->GetNumVertices());

                for (int i=0; i<childFaces.size(); ++i) {
                    current.faces[childFaces[i]] = f->GetChild(i);
                }

                // populate child face-verts
                Index childVert = refiner->GetFaceChildVertex(level-1, face);
                Hvertex * v = f->Subdivide();
                assert(v->GetParentFace());
                current.verts[childVert] = v;
            }

            for (int edge=0; edge < refiner->GetNumEdges(level-1); ++edge) {
                // populate child edge-verts
                Index childVert = refiner->GetEdgeChildVertex(level-1,edge);
                Hhalfedge * e = previous.edges[edge];
                Hvertex * v = e->Subdivide();
                assert(v->GetParentEdge());
                current.verts[childVert] = v;
            }

            for (int vert = 0; vert < refiner->GetNumVertices(level-1); ++vert) {
                // populate child vert-verts
                Index childVert = refiner->GetVertexChildVertex(level-1, vert);
                Hvertex * v = previous.verts[vert]->Subdivide();
                current.verts[childVert] = v;
                assert(v->GetParentVertex());
            }

            // populate child edges
            for (int edge=0; edge < refiner->GetNumEdges(level); ++edge) {

                ConstIndexArray vtrVerts = refiner->GetEdgeVertices(level, edge);

                Hvertex const * v0 = current.verts[vtrVerts[0]],
                              * v1 = current.verts[vtrVerts[1]];
                assert(v0 and v1);

                Hhalfedge * e= v0->GetEdge(v1);
                if (not e) {
                    e = v1->GetEdge(v0);
                }
                assert(e);
                current.edges[edge] = e;
            }
            ecount += refiner->GetNumEdges(level-1);
        }
    }
};

//------------------------------------------------------------------------------
static int
checkMesh(ShapeDesc const & desc, int maxlevel) {

    static char const * schemes[] = { "Bilinear", "Catmark", "Loop" };
    printf("- %-25s ( %-8s ): \n", desc.name.c_str(), schemes[desc.scheme]);

    int count=0;
    float deltaAvg[3] = {0.0f, 0.0f, 0.0f},
          deltaCnt[3] = {0.0f, 0.0f, 0.0f};

    std::vector<xyzVV> hbrVertexData,
                       vtrVertexData;

    Hmesh *  hmesh =
        interpolateHbrVertexData(desc, maxlevel);

    FarTopologyRefiner * refiner =
        interpolateVtrVertexData(desc, maxlevel, vtrVertexData);

    {   // copy Hbr vertex data into a re-ordered buffer (for easier comparison)

        Mapper mapper(refiner, hmesh);

        int nverts = hmesh->GetNumVertices();
        assert( nverts==refiner->GetNumVerticesTotal() );

        hbrVertexData.resize(nverts);

        for (int level=0, ofs=0; level<(maxlevel+1); ++level) {

           Mapper::LevelMap & map = mapper.maps[level];
           for (int i=0; i<(int)map.verts.size(); ++i) {
                Hvertex * v = map.verts[i];
                hbrVertexData[ofs++] = v->GetData();
           }
        }

        //printVertexData(hbrVertexData, vtrVertexData);
    }

    int nverts = (int)vtrVertexData.size();

    for (int i=0; i<nverts; ++i) {

        xyzVV & hbrVert = hbrVertexData[i],
              & vtrVert = vtrVertexData[i];

#ifdef __INTEL_COMPILER // remark #1572: floating-point equality and inequality comparisons are unreliable
#pragma warning disable 1572
#endif
        if ( hbrVert.GetPos()[0] != vtrVert.GetPos()[0] )
            deltaCnt[0]++;
        if ( hbrVert.GetPos()[1] != vtrVert.GetPos()[1] )
            deltaCnt[1]++;
        if ( hbrVert.GetPos()[2] != vtrVert.GetPos()[2] )
            deltaCnt[2]++;
#ifdef __INTEL_COMPILER
#pragma warning enable 1572
#endif
        float delta[3] = { hbrVert.GetPos()[0] - vtrVert.GetPos()[0],
                           hbrVert.GetPos()[1] - vtrVert.GetPos()[1],
                           hbrVert.GetPos()[2] - vtrVert.GetPos()[2] };

        deltaAvg[0]+=delta[0];
        deltaAvg[1]+=delta[1];
        deltaAvg[2]+=delta[2];

        float dist = sqrtf( delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
        if ( dist > PRECISION ) {
            if (not g_debugmode)
                printf("// HbrVertex<T> %d fails : dist=%.10f (%.10f %.10f %.10f)"
                       " (%.10f %.10f %.10f)\n", i, dist, hbrVert.GetPos()[0],
                                                          hbrVert.GetPos()[1],
                                                          hbrVert.GetPos()[2],
                                                          vtrVert.GetPos()[0],
                                                          vtrVert.GetPos()[1],
                                                          vtrVert.GetPos()[2] );
           count++;
        }
    }

    if (deltaCnt[0])
        deltaAvg[0]/=deltaCnt[0];
    if (deltaCnt[1])
        deltaAvg[1]/=deltaCnt[1];
    if (deltaCnt[2])
        deltaAvg[2]/=deltaCnt[2];

    if (not g_debugmode) {
        printf("  delta ratio : (%d/%d %d/%d %d/%d)\n", (int)deltaCnt[0], nverts,
                                                        (int)deltaCnt[1], nverts,
                                                        (int)deltaCnt[2], nverts );
        printf("  average delta : (%.10f %.10f %.10f)\n", deltaAvg[0],
                                                          deltaAvg[1],
                                                          deltaAvg[2] );
        if (count==0)
            printf("  success !\n");
    }

    return count;
}

//------------------------------------------------------------------------------
int main(int /* argc */, char ** /* argv */) {

    int levels=5, total=0;

    initShapes();

    if (g_debugmode)
        printf("[ ");
    else
        printf("precision : %f\n",PRECISION);
    for (int i=0; i<(int)g_shapes.size(); ++i) {
        total+=checkMesh(g_shapes[i], levels);
    }

    if (g_debugmode)
        printf("]\n");
    else {
        if (total==0)
          printf("All tests passed.\n");
        else
          printf("Total failures : %d\n", total);
    }
}

//------------------------------------------------------------------------------
