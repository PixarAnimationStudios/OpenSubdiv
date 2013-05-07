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
