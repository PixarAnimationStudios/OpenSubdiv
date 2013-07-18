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
#ifndef EXAMPLES_MAYAVIEWER_HBRUTIL_H_
#define EXAMPLES_MAYAVIEWER_HBRUTIL_H_

#include <far/meshFactory.h>  // need to define HBR_ADAPTIVE
#include <hbr/mesh.h>
#include <osd/vertex.h>

#include <vector>

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;
typedef OpenSubdiv::HbrFVarData<OpenSubdiv::OsdVertex> OsdHbrFVarData;

// XXX placeholder for enums used by ConvertToHBR
//     would be nice if these came from OsdHbrMesh, is there a util
//     class where these could live?
typedef struct
{
    enum SchemeType { kCatmark=0, 
                      kLoop=1, 
                      kBilinear=2 };
} HbrMeshUtil;


//
//  Face-varying data descriptor
//  Wrapper for basic information needed to request
//  face-varying data allocation from HBR
//
class FVarDataDesc
{
public:

    // Must be instantiated with descriptor information
    FVarDataDesc(      int   count, 
                  const int *indices,   // start index for each face-varying variable
                  const int *widths,    // width for each face-varying variable
                        int  width,
                        OsdHbrMesh::InterpolateBoundaryMethod 
                             boundary=OsdHbrMesh::k_InterpolateBoundaryNone
                        ) 
    {
        _fvarCount      = count; 
        _totalfvarWidth = width;
        _fvarIndices.assign( indices, indices+count );
        _fvarWidths.assign( widths, widths+count );
        _interpBoundary = boundary;
    }

    ~FVarDataDesc() {}


    // Accessors
          int  getCount() const          { return _fvarCount; }
    const int *getIndices() const        { return &_fvarIndices.front(); }
    const int *getWidths() const         { return &_fvarWidths.front(); }
          int  getTotalWidth() const     { return _totalfvarWidth; }
    OsdHbrMesh::InterpolateBoundaryMethod 
               getInterpBoundary() const { return _interpBoundary; }

private:

    // Number of facevarying datums
    int  _fvarCount;        

    // Start indices of the facevarying data we want to store
    std::vector<int> _fvarIndices;      

    // Individual widths of the facevarying data we want to store
    std::vector<int> _fvarWidths;       

    // Total widths of the facevarying data
    int  _totalfvarWidth;   

    // How to interpolate boundaries
    OsdHbrMesh::InterpolateBoundaryMethod _interpBoundary;
};


extern "C" OsdHbrMesh * 
ConvertToHBR( int nVertices,
              std::vector<int>   const & faceVertCounts,
              std::vector<int>   const & faceIndices,
              std::vector<int>   const & vtxCreaseIndices,
              std::vector<double> const & vtxCreases,
              std::vector<int>   const & edgeCrease1Indices,
              std::vector<float> const & edgeCreases1,
              std::vector<int>   const & edgeCrease2Indices,
              std::vector<double> const & edgeCreases2,
              OsdHbrMesh::InterpolateBoundaryMethod interpBoundary,
              HbrMeshUtil::SchemeType scheme,
              bool usingPtex=false,
              FVarDataDesc const * fvarDesc=NULL,
              std::vector<float> const * fvarData=NULL
            );

#endif  // EXAMPLES_MAYAVIEWER_HBRUTIL_H_
