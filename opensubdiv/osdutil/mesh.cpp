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

#include "mesh.h"

#define HBR_ADAPTIVE
#include "../hbr/mesh.h"
#include "../hbr/bilinear.h"
#include "../hbr/catmark.h"
#include "../hbr/loop.h"
#include "../far/stencilTablesFactory.h"
#include "../osd/vertex.h"

#include <sstream>

using namespace std;
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T>
static bool _ProcessTagsAndFinishMesh(
    OpenSubdiv::HbrMesh<T> *mesh,
    const OsdUtilTagData &tagData,
    std::string *errorMessage);

template <class T>  
OsdUtilMesh<T>::OsdUtilMesh() :
    _hmesh(NULL),
    _valid(false)
{
}

template <class T>
OsdUtilMesh<T>::~OsdUtilMesh()
{
    delete _hmesh;
}

template <class T>  bool
OsdUtilMesh<T>::Initialize(const OsdUtilSubdivTopology &topology,
                             std::string *errorMessage,
                             Scheme scheme)
{
    
    if (not topology.IsValid(errorMessage)) {
        return false;
    }

    _t = topology;    

    static HbrCatmarkSubdivision<T>  _catmark;
    static HbrBilinearSubdivision<T>  _bilinear;
    static HbrLoopSubdivision<T>  _loop;

    HbrSubdivision<T> *subdivisions;
    switch (scheme) {
        case SCHEME_CATMARK:
            subdivisions = &_catmark;
            break;
        case SCHEME_BILINEAR:
            subdivisions = &_bilinear;
            break;
        case SCHEME_LOOP:
            subdivisions = &_loop;
            break;
    }

    if (_t.fvNames.empty()) {
        _hmesh = new HbrMesh<T>(subdivisions);
    } else {

        int fvarcount = (int) _t.fvNames.size();

        // For now we only handle 1 float per FV variable.
        _fvarwidths.assign(fvarcount, 1);

        int startIndex = 0;
        for (int fvarindex = 0; fvarindex < fvarcount; ++fvarindex) {
            _fvarindices.push_back(startIndex);
            _fvaroffsets[_t.fvNames[fvarindex]] = startIndex;
            startIndex += _fvarwidths[fvarindex];
        }

        _hmesh = new HbrMesh<T>(
            subdivisions, fvarcount, &_fvarindices[0],
            &_fvarwidths[0], fvarcount);
    }

    T v;
    for (int i = 0; i < _t.numVertices; ++i) {
        HbrVertex<T>* hvert = _hmesh->NewVertex(i, v);
        if (!hvert) {
            if (errorMessage)  
                *errorMessage = "Unable to create call NewVertex for Hbr";
            return false;
        }
    }

    // Sanity check
    int fvarWidth = _hmesh->GetTotalFVarWidth();
    if (_t.fvData.size() < _t.nverts.size() * fvarWidth ||
        fvarWidth != (int)_t.fvNames.size()) {
        if (errorMessage)  {                 
            stringstream ss;
            ss << "Incorrectly sized face data: name count = " <<
                _t.fvNames.size() << 
                ",  data width = " << fvarWidth <<
                ",  face count = " <<  _t.nverts.size() <<
                ",  total data size = %d." << _t.fvData.size();
            *errorMessage = ss.str();
        }
        return false;
    }

    // ptex index is not necessarily the same as the face index
    int ptexIndex = 0;

    // face-vertex count offset
    int fvcOffset = 0;

    int facesCreated = 0;
    for (int i=0; i<(int)_t.nverts.size(); ++i) {
        int nv = _t.nverts[i];
        
/*XXX  No loop yet
        if ((_scheme==kLoop) and (nv!=3)) {
            if (errorMessage)            
                *errorMessage = TfStringPrintf(
                    "Trying to create a Loop surbd with non-triangle face\n");
            return false;
        }
*/        

        for(int j=0;j<nv;j++) {
            HbrVertex<T> * origin      =
                _hmesh->GetVertex(_t.indices[fvcOffset + j]);
            HbrVertex<T> * destination =
                _hmesh->GetVertex(_t.indices[fvcOffset + (j+1)%nv] );
            HbrHalfedge<T> * opposite  = destination->GetEdge(origin);

            if(origin==NULL || destination==NULL) {
                if (errorMessage)
                    *errorMessage =
                  " An edge was specified that connected a nonexistent vertex";
                return false;
            }

            if(origin == destination) {

                if (errorMessage)                
                    *errorMessage = 
                    " An edge was specified that connected a vertex to itself";

                return false;
            }

            if(opposite && opposite->GetOpposite() ) {
                if (errorMessage)                
                    *errorMessage =
                " A non-manifold edge incident to more than 2 faces was found";
                return false;
            }
            
            if(origin->GetEdge(destination)) {
                if (errorMessage)                
                    *errorMessage =
              " An edge connecting two vertices was specified more than once."
              " It's likely that an incident face was flipped\n";
                return false;
            }
        }


        HbrFace<T>* hface = _hmesh->NewFace(
            nv, &_t.indices[fvcOffset], 0);
        
        ++facesCreated;
        
        if (!hface) {
             if (errorMessage)  
                 *errorMessage = "Unable to create Hbr face";
             return false;
        }

        // The ptex index isn't a straight-up polygon index; rather,
        // it's an index into a "minimally quadrangulated" base mesh.
        // Take all non-rect polys and subdivide them once.
        hface->SetPtexIndex(ptexIndex);
        ptexIndex += (nv == 4) ? 1 : nv;

        // prideout: 3/21/2013 - Inspired by "GetFVarData" in examples/mayaViewer/hbrUtil.cpp
        if (!_t.fvNames.empty()) {
            const float* faceData = &(_t.fvData[fvcOffset*fvarWidth]);
            for (int fvi = 0; fvi < nv; ++fvi) {
                int vindex = _t.indices[fvi + fvcOffset];
                HbrVertex<T>* v = _hmesh->GetVertex(vindex);
                HbrFVarData<T>& fvarData = v->GetFVarData(hface);
                if (!fvarData.IsInitialized()) {
                    fvarData.SetAllData(fvarWidth, faceData); 
                } else if (!fvarData.CompareAll(fvarWidth, faceData)) {

                    // If data exists for this face vertex, but is different
                    // (e.g. we're on a UV seam) create another fvar datum
                    HbrFVarData<T>& fvarData = v->NewFVarData(hface);
                    fvarData.SetAllData(fvarWidth, faceData);
                }

                // Advance pointer to next set of face-varying data
                faceData += fvarWidth;
            }
        }

        fvcOffset += nv;
    }

    if (not _ProcessTagsAndFinishMesh( _hmesh, _t.tagData, errorMessage))
        return false;

    _valid = true;

    return true;
}


// Interleave the face-varying sets specified by "names", adding
// floats into the "fvdata" vector.  The number of added floats is:
//    names.size() * NumRefinedFaces * 4
template <class T> void
OsdUtilMesh<T>::GetRefinedFVData(
    int level, const vector<string>& names, vector<float>* outdata)
{

    // First some sanity checking.
    if (!outdata) {
        return;
    }
    for (int i=0; i<(int)names.size(); ++i) {
        const string &name = names[i];
        if (_fvaroffsets.find(name) == _fvaroffsets.end()) {
            /* XXX 
            printf("Can't find facevarying variable %s\n", name.c_str());
            */
            return;
        }
    }

    // Fetch *all* faces; this includes all subdivision levels.
    vector<HbrFace<T> *> faces;
    _hmesh->GetFaces(std::back_inserter(faces));

    // Iterate through all faces, filtering on the requested subdivision level.
    for (int i=0; i<(int)faces.size(); ++i) {
        HbrFace<T>* face = faces[i];
        if (face->GetDepth() != level) {
            continue;
        }
        int ncorners = face->GetNumVertices();
        for (int corner = 0; corner < ncorners; ++corner) {
            HbrFVarData<T>& fvariable = face->GetFVarData(corner);
            
            for (int j=0; j<(int)names.size(); ++j) {
                const string &name = names[j];
            
                int offset = _fvaroffsets[name];
                const float* data = fvariable.GetData(offset);
                outdata->push_back(*data);
            }
        }
    }
}



// ProcessTagsAndFinishMesh(...)
// This translates prman-style lists of tags into OSD method calls.
//
// prideout: 3/19/2013 - since tidSceneRenderer has a similar
//           function, we should factor this into an amber utility, or
//           into osd itself.  I'd vote for the latter.  It already has 
//           a shapeUtils in its regression suite that almost fits the bill.
//
// prideout: 3/19/2013 - edits are not yet supported.
template <class T>
bool _ProcessTagsAndFinishMesh(
    OpenSubdiv::HbrMesh<T> *mesh,
    const OsdUtilTagData &tagData,
    std::string *errorMessage
    )           
{
    // Boundary interpolation "none" can yield uninitialized memory access in
    // hbr.  Default to edge and corner interpolation because that is what
    // users want almost all the time.
    mesh->SetInterpolateBoundaryMethod(
        OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeAndCorner);

    const int* currentInt = tagData.intArgs.size() ? &tagData.intArgs[0] : NULL;
    const float* currentFloat = tagData.floatArgs.size() ? &tagData.floatArgs[0] : NULL;
    const string* currentString = tagData.stringArgs.size() ? &tagData.stringArgs[0] : NULL;

    // TAGS (crease, corner, hole, smooth triangles, edits(vertex,
    // edge, face), creasemethod, facevaryingpropagatecorners, interpolateboundary
    for(int i = 0; i < (int)tagData.tags.size(); ++i){
        OsdUtilTagData::TagType tag = tagData.tags[i];
	int nint = tagData.numArgs[3*i];
	int nfloat = tagData.numArgs[3*i+1];
	int nstring = tagData.numArgs[3*i+2];

	switch (tag) {
        case OsdUtilTagData::INTERPOLATE_BOUNDARY:
	    // Interp boundaries
	    assert(nint == 1);
	    switch(currentInt[0]) {
            case 0:
                mesh->SetInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryNone);
                break;
            case 1:
                mesh->SetInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeAndCorner);
                break;
            case 2:
                mesh->SetInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly);
                break;
            default: {
                stringstream ss;
                ss << "Subdivmesh contains unknown interpolate boundary method: " << currentInt[0];
                *errorMessage = ss.str();
                return false;
               }
            }
            break;
            
	    // Processing of this tag is done in mesh->Finish()
        case OsdUtilTagData::CREASE:
	    for(int j = 0; j < nint-1; ++j) {
		// Find the appropriate edge
                HbrVertex<T>* v = mesh->GetVertex(currentInt[j]);
                HbrVertex<T>* w = mesh->GetVertex(currentInt[j+1]);
                HbrHalfedge<T>* e = NULL;
		if(v && w) {
		    e = v->GetEdge(w);
		    if(!e) {
			// The halfedge might be oriented the other way
			e = w->GetEdge(v);
		    }
		}
		if(!e) {
                    stringstream ss;
                    ss << "Subdivmesh has non-existent sharp edge (" <<
                        currentInt[j]<< ", " <<  currentInt[j+1] << ")";
                    *errorMessage = ss.str();            
                    return false;
		} else {
                    float sharpness = std::max(0.0f, ((nfloat > 1) ? currentFloat[j] : currentFloat[0]));
		    e->SetSharpness(sharpness);
		}
	    }
            break;
            
        case OsdUtilTagData::CORNER: {
	    for(int j = 0; j < nint; ++j) {
                HbrVertex<T>* v = mesh->GetVertex(currentInt[j]);
		if(v) {
		    v->SetSharpness(std::max(0.0f, ((nfloat > 1) ? currentFloat[j] : currentFloat[0])));
		} else {
                    stringstream ss;
                    ss << "Subdivmesh has non-existent sharp vertex " << currentInt[j];
                    *errorMessage = ss.str();
                    return false;
		}
	    }
        }
            break;
            
        case OsdUtilTagData::HOLE:
	    for(int j = 0; j < nint; ++j) {
                HbrFace<T>* f = mesh->GetFace(currentInt[j]);
		if(f) {
		    f->SetHole();
		} else {
                    stringstream ss;
                    ss << "Subdivmesh has hole at non-existent face " << currentInt[j];
                    *errorMessage = ss.str();
                    return false;
		}
	    }
            break;

        case OsdUtilTagData::FACE_VARYING_INTERPOLATE_BOUNDARY:            
	    switch(currentInt[0]) {
            case 0:
                mesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryNone);
                break;
            case 1:
		mesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeAndCorner);
		break;
            case 2:
		mesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly);
		break;
            case 3:
		mesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryAlwaysSharp);
		break;
            default: {
                stringstream ss;
                ss << "Subdivmesh contains unknown facevarying interpolate boundary method "
                    << currentInt[0];
                *errorMessage = ss.str();
                return false;                
            }
            }
            break;
            
        case OsdUtilTagData::SMOOTH_TRIANGLES: 
	    // Do nothing - CatmarkMesh should handle it
            break;
        case OsdUtilTagData::CREASE_METHOD: {
	    if(nstring < 1) {
                *errorMessage = "Creasemethod tag missing string argument on SubdivisionMesh";
                return false;                                
	    } 
            HbrSubdivision<T>* subdivisionMethod = mesh->GetSubdivision();
            if(strcmp(currentString->c_str(), "normal") == 0) {
                subdivisionMethod->SetCreaseSubdivisionMethod(
                    HbrSubdivision<T>::k_CreaseNormal);
            } else if(strcmp(currentString->c_str(), "chaikin") == 0) {
                subdivisionMethod->SetCreaseSubdivisionMethod(
                    HbrSubdivision<T>::k_CreaseChaikin);		    
            } else {
                stringstream ss;
                ss << "Creasemethod tag specifies unknown crease subdivision method "
                   << currentString->c_str() << " SubdivisionMesh";
                *errorMessage = ss.str();
                return false;                                                    
            }
        }
            break;
            
        case OsdUtilTagData::FACE_VARYING_PROPOGATE_CORNERS:
	    if(nint != 1) {
                stringstream ss;
                ss << "Expecting single integer argument for \"facevaryingpropagatecorners\" on SubdivisionMesh";
                *errorMessage = ss.str();
                return false;                                                    
	    } else {
		mesh->SetFVarPropagateCorners(currentInt[0] != 0);
	    }
            break;
            
        case OsdUtilTagData::VERTEX_EDIT:
        case OsdUtilTagData::EDGE_EDIT:
                // XXX DO EDITS
                *errorMessage = "vertexedit and edgeedit not yet supported";
                break;
                
         default: {
             stringstream ss;
             ss << "Unknown tag: " << tag;
             *errorMessage = ss.str();
             return false;                                                    
         }
        }

	// update the tag data pointers
	currentInt += nint;
	currentFloat += nfloat;
	currentString += nstring;
    }

    mesh->Finish();

    return true;
}



//XXX Note that these explicit template instantiations
// need to live at the _bottom_ of the file.

// Explicitly instantiate OsdUtilMesh for these
// two vertex types.  Since the class members are in
// the .cpp file, clients can't create template
// instances other than these two vertex classes.
//template class OsdUtilMesh<OsdVertex>;
template class OsdUtilMesh<OsdVertex>;

//template class OsdUtilMesh<FarStencilFactoryVertex>;
template class OsdUtilMesh<FarStencilFactoryVertex>;

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
