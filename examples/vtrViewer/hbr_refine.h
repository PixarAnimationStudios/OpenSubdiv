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

#ifndef HBR_REFINE_H
#define HBR_REFINE_H

#ifndef HBR_ADAPTIVE
#define HBR_ADAPTIVE
#endif

#include <far/patchTables.h>

#include <common/hbr_utils.h>

//------------------------------------------------------------------------------
// Vertex class implementation
struct Vertex {

    Vertex() { /* _pos[0]=_pos[1]=_pos[2]=0.0f; */ }

    Vertex( int /*i*/ ) { }

    Vertex( float x, float y, float z ) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    Vertex( const Vertex & src ) { _pos[0]=src._pos[0]; _pos[1]=src._pos[1]; _pos[2]=src._pos[2]; }

   ~Vertex( ) { }

    void AddWithWeight(const Vertex& src, float weight) {
        _pos[0]+=weight*src._pos[0];
        _pos[1]+=weight*src._pos[1];
        _pos[2]+=weight*src._pos[2];
    }

    void AddVaryingWithWeight(const Vertex& , float) { }

    void Clear( void * =0 ) { _pos[0]=_pos[1]=_pos[2]=0.0f; }

    void SetPosition(float x, float y, float z) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    void ApplyVertexEdit(const OpenSubdiv::HbrVertexEdit<Vertex> & edit) {
        const float *src = edit.GetEdit();
        switch(edit.GetOperation()) {
          case OpenSubdiv::HbrHierarchicalEdit<Vertex>::Set:
            _pos[0] = src[0];
            _pos[1] = src[1];
            _pos[2] = src[2];
            break;
          case OpenSubdiv::HbrHierarchicalEdit<Vertex>::Add:
            _pos[0] += src[0];
            _pos[1] += src[1];
            _pos[2] += src[2];
            break;
          case OpenSubdiv::HbrHierarchicalEdit<Vertex>::Subtract:
            _pos[0] -= src[0];
            _pos[1] -= src[1];
            _pos[2] -= src[2];
            break;
        }
    }

    void ApplyMovingVertexEdit(const OpenSubdiv::HbrMovingVertexEdit<Vertex> &) { }

    const float * GetPos() const { return _pos; }

private:
    float _pos[3];
};

//------------------------------------------------------------------------------

typedef OpenSubdiv::HbrMesh<Vertex>           Hmesh;
typedef OpenSubdiv::HbrFace<Vertex>           Hface;
typedef OpenSubdiv::HbrVertex<Vertex>         Hvertex;
typedef OpenSubdiv::HbrHalfedge<Vertex>       Hhalfedge;

//------------------------------------------------------------------------------

// refine the Hbr mesh uniformly
void RefineUniform(Hmesh & mesh, int maxlevel, std::vector<Hface const *> & refinedFaces);

// refine the Hbr mesh adaptively
int RefineAdaptive(Hmesh & mesh, int maxlevel, std::vector<Hface const *> & refinedFaces);

OpenSubdiv::Far::PatchTables const * CreatePatchTables(Hmesh & mesh, int maxvalence);

//------------------------------------------------------------------------------

#endif // HBR_REFINE_H
