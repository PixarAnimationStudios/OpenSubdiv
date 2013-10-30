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

#include <osd/vertex.h>
#include <hbr/catmark.h>

#include <sstream>

using namespace std;
using namespace OpenSubdiv;

static HbrCatmarkSubdivision<OsdVertex>  _catmark;


PxOsdUtilSubdivTopology::PxOsdUtilSubdivTopology():
    name("noname"),
    numVertices(0),
    maxLevels(2)  // arbitrary, start with a reasonable subdivision level
{
    std::cout << "Creating subdiv topology object\n";    
}

PxOsdUtilSubdivTopology::~PxOsdUtilSubdivTopology()
{
    std::cout << "Destroying subdiv topology object\n";
}


bool
PxOsdUtilSubdivTopology::Initialize(
    int numVerticesParam,
    const int *nvertsParam, int numFaces,
    const int *indicesParam, int indicesLen,
    int levels,
    string *errorMessage)
{

    numVertices = numVerticesParam;
    maxLevels = levels;

    nverts.resize(numFaces);
    for (int i=0; i<numFaces; ++i) {
        nverts[i] = nvertsParam[i];
    }
        
    indices.resize(indicesLen);    
    for (int i=0; i<indicesLen; ++i) {
        indices[i] = indicesParam[i];
    }

    return IsValid(errorMessage);
}

bool
PxOsdUtilSubdivTopology::IsValid(string *errorMessage) const
{
    if (numVertices == 0) {
        if (errorMessage) {
            stringstream ss;
            ss << "Topology " << name << " has no vertices";
            *errorMessage = ss.str();
        }
        return false;
    }
    
    for (int i=0; i<(int)indices.size(); ++i) {
        if ((indices[i] < 0) or
            (indices[i] >= numVertices)) {
            if (errorMessage) {
                stringstream ss;
                ss << "Topology " << name << " has bad index " << indices[i] << " at index " << i;
                *errorMessage = ss.str();
            }
            return false;
        }

    }

    int totalNumIndices = 0;
    for (int i=0; i< (int)nverts.size(); ++i) {
        if (nverts[i] < 1) {
            if (errorMessage) {
                stringstream ss;
                ss << "Topology " << name << " has bad nverts " << nverts[i] << " at index " << i;
                *errorMessage = ss.str();
            }
            return false;            
        }
        totalNumIndices += nverts[i];
    }
    
    if (totalNumIndices != (int)indices.size()) {
        if (errorMessage) {
            *errorMessage = "Bad indexing for face topology";
        }
  
        return false;
    }
    
    std::cout << "\n";

    return true;
}


void
PxOsdUtilSubdivTopology::Print() const
{
    std::cout << "Mesh " << name << "\n";
    std::cout << "\tnumVertices = " << numVertices << "\n";
    std::cout << "\tmaxLevels = " << maxLevels << "\n";
    std::cout << "\tindices ( " << indices.size() << ") : ";
    for (int i=0; i<(int)indices.size(); ++i) {
        std::cout << indices[i] << ", ";
    }
    std::cout << "\n";

    std::cout << "\tnverts ( " << nverts.size() << ") : ";
    for (int i=0; i<(int)nverts.size(); ++i) {
        std::cout << nverts[i] << ", ";
    }
    std::cout << "\n";    

}

PxOsdUtilMesh::PxOsdUtilMesh(
    const PxOsdUtilSubdivTopology &topology,
    std::string *errorMessage):
    _t(topology),
    _hmesh(NULL),
    _valid(false)
{

    if (not topology.IsValid(errorMessage)) {
        return;
    }
    
    if (_t.fvNames.empty()) {

        std::cout << "Creating NON face varying hbrmesh\n";
        
        _hmesh = new HbrMesh<OsdVertex>(&_catmark);
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


        std::cout << "Creating face varying hbrmesh\n";
        
        _hmesh = new HbrMesh<OsdVertex>(
            &_catmark, fvarcount, &_fvarindices[0],
            &_fvarwidths[0], fvarcount);
    }

    OsdVertex v;
    for (int i = 0; i < _t.numVertices; ++i) {
        HbrVertex<OsdVertex>* hvert = _hmesh->NewVertex(i, OsdVertex());
        if (!hvert) {
            if (errorMessage)  
                *errorMessage = "Unable to create call NewVertex for Hbr";
            return;
        }
    }

    std::cout << "Created " << _t.numVertices << " vertices for hbr mesh\n";

    // Sanity check
    int fvarWidth = _hmesh->GetTotalFVarWidth();

    std::cout << "Total fvarWidth = " << fvarWidth << "\n";
    
    if (_t.fvData.size() < _t.nverts.size() * fvarWidth ||
        fvarWidth != (int)_t.fvNames.size()) {
/*XXX            if (errorMessage)  
                *errorMessage = TfStringPrintf(
                    "Incorrectly sized face data: name count = %d, "
                    "data width = %d, face count = %d, total data size = %d.",
                    (int) _t.fvNames.size(),
                    fvarWidth,
                    (int) _t.nverts.size(),
                    (int) _t.fvData.size());
*/                    
            return;
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
            HbrVertex<OsdVertex> * origin      =
                _hmesh->GetVertex(_t.indices[fvcOffset + j]);
            HbrVertex<OsdVertex> * destination =
                _hmesh->GetVertex(_t.indices[fvcOffset + (j+1)%nv] );
            HbrHalfedge<OsdVertex> * opposite  = destination->GetEdge(origin);

            if(origin==NULL || destination==NULL) {
                if (errorMessage)
                    *errorMessage =
                  " An edge was specified that connected a nonexistent vertex";
                return;
            }

            if(origin == destination) {

                if (errorMessage)                
                    *errorMessage = 
                    " An edge was specified that connected a vertex to itself";

                return;
            }

            if(opposite && opposite->GetOpposite() ) {
                if (errorMessage)                
                    *errorMessage =
                " A non-manifold edge incident to more than 2 faces was found";
                return;
            }
            
            if(origin->GetEdge(destination)) {
                if (errorMessage)                
                    *errorMessage =
              " An edge connecting two vertices was specified more than once."
              " It's likely that an incident face was flipped\n";
                return;
            }
        }

        std::cout << "Creating face with indices at offset " << fvcOffset << " ";
        for (int k=0; k<nv; ++k) {
            std::cout << _t.indices[fvcOffset+k] << " ";
        }
        std::cout << "\n";
        
        HbrFace<OsdVertex>* hface = _hmesh->NewFace(
            nv, &(_t.indices[fvcOffset]), 0);
        

        ++facesCreated;
        
        if (!hface) {
             if (errorMessage)  
                 *errorMessage = "Unable to create Hbr face";
             return;
        }

        // The ptex index isn't a straight-up polygon index; rather,
        // it's an index into a "minimally quadrangulated" base mesh.
        // Take all non-rect polys and subdivide them once.
        hface->SetPtexIndex(ptexIndex);
        ptexIndex += (nv == 4) ? 1 : nv;

        // prideout: 3/21/2013 - Inspired by "GetFVarData" in examples/mayaViewer/hbrUtil.cpp
        if (!_t.fvNames.empty()) {
            std::cout << "found fvNames!\n";
                
            const float* faceData = &(_t.fvData[fvcOffset*fvarWidth]);
            for (int fvi = 0; fvi < nv; ++fvi) {
                int vindex = _t.indices[fvi + fvcOffset];
                HbrVertex<OsdVertex>* v = _hmesh->GetVertex(vindex);
                HbrFVarData<OsdVertex>& fvarData = v->GetFVarData(hface);
                if (!fvarData.IsInitialized()) {
                    fvarData.SetAllData(fvarWidth, faceData); 
                } else if (!fvarData.CompareAll(fvarWidth, faceData)) {

                    // If data exists for this face vertex, but is different
                    // (e.g. we're on a UV seam) create another fvar datum
                    HbrFVarData<OsdVertex>& fvarData = v->NewFVarData(hface);
                    fvarData.SetAllData(fvarWidth, faceData);
                }

                // Advance pointer to next set of face-varying data
                faceData += fvarWidth;
            }
        }

        fvcOffset += nv;
    }

    std::cout << "Create " << facesCreated << " faces in hbrMesh\n";
    
    _ProcessTagsAndFinishMesh(
        _hmesh, _t.tagData.tags, _t.tagData.numArgs, _t.tagData.intArgs,
        _t.tagData.floatArgs, _t.tagData.stringArgs);

    _valid = true;
}

PxOsdUtilMesh::~PxOsdUtilMesh()
{
    std::cout << "Deleting PxOsdUtilMesh\n";
    delete _hmesh;
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
void
PxOsdUtilMesh::_ProcessTagsAndFinishMesh(
    HbrMesh<OsdVertex> *mesh,
    const vector<string> &tags,
    const vector<int> &numArgs,
    const vector<int> &intArgs,
    const vector<float> &floatArgs,
    const vector<string> &stringArgs)
{
    mesh->SetInterpolateBoundaryMethod(HbrMesh<OsdVertex>::k_InterpolateBoundaryEdgeOnly);

    const int* currentInt = &intArgs[0];
    const float* currentFloat = &floatArgs[0];
    const string* currentString = &stringArgs[0];

    // TAGS (crease, corner, hole, smooth triangles, edits(vertex,
    // edge, face), creasemethod, facevaryingpropagatecorners, interpolateboundary
    for(int i = 0; i < (int)tags.size(); ++i){
	const char * tag = tags[i].c_str();
	int nint = numArgs[3*i];
	int nfloat = numArgs[3*i+1];
	int nstring = numArgs[3*i+2];

	// XXX could use tokens here to reduce string matching overhead
	if(strcmp(tag, "interpolateboundary") == 0) {
	    // Interp boundaries
	    assert(nint == 1);
	    switch(currentInt[0]) {
            case 0:
                mesh->SetInterpolateBoundaryMethod(HbrMesh<OsdVertex>::k_InterpolateBoundaryNone);
                break;
            case 1:
                mesh->SetInterpolateBoundaryMethod(HbrMesh<OsdVertex>::k_InterpolateBoundaryEdgeAndCorner);
                break;
            case 2:
                mesh->SetInterpolateBoundaryMethod(HbrMesh<OsdVertex>::k_InterpolateBoundaryEdgeOnly);
                break;
            default:
/*XXX
                TF_WARN("Subdivmesh contains unknown interpolate boundary method: %d\n",
                        currentInt[0]);
*/                        
		break;
	    }
	    // Processing of this tag is done in mesh->Finish()
	} else if(strcmp(tag, "crease") == 0) {
	    for(int j = 0; j < nint-1; ++j) {
		// Find the appropriate edge
                HbrVertex<OsdVertex>* v = mesh->GetVertex(currentInt[j]);
                HbrVertex<OsdVertex>* w = mesh->GetVertex(currentInt[j+1]);
                HbrHalfedge<OsdVertex>* e = NULL;
		if(v && w) {
		    e = v->GetEdge(w);
		    if(!e) {
			// The halfedge might be oriented the other way
			e = w->GetEdge(v);
		    }
		}
		if(!e) {
/*XXX                    
		    TF_WARN("Subdivmesh has non-existent sharp edge (%d,%d).\n",
                            currentInt[j], currentInt[j+1]);
*/                            
		} else {
		    e->SetSharpness(std::max(0.0f, ((nfloat > 1) ? currentFloat[j] : currentFloat[0])));
		}
	    }
	} else if(strcmp(tag, "corner") == 0) {
	    for(int j = 0; j < nint; ++j) {
                HbrVertex<OsdVertex>* v = mesh->GetVertex(currentInt[j]);
		if(v) {
		    v->SetSharpness(std::max(0.0f, ((nfloat > 1) ? currentFloat[j] : currentFloat[0])));
		} else {
/*XXX                                        
		    TF_WARN("Subdivmesh has non-existent sharp vertex %d.\n", currentInt[j]);
*/                    
		}
	    }
	} else if(strcmp(tag, "hole") == 0) {
	    for(int j = 0; j < nint; ++j) {
                HbrFace<OsdVertex>* f = mesh->GetFace(currentInt[j]);
		if(f) {
		    f->SetHole();
		} else {
/*XXX                                                            
		    TF_WARN("Subdivmesh has hole at non-existent face %d.\n",
                            currentInt[j]);
*/                            
		}
	    }
	} else if(strcmp(tag, "facevaryinginterpolateboundary") == 0) {
	    switch(currentInt[0]) {
            case 0:
                mesh->SetFVarInterpolateBoundaryMethod(HbrMesh<OsdVertex>::k_InterpolateBoundaryNone);
                break;
            case 1:
		mesh->SetFVarInterpolateBoundaryMethod(HbrMesh<OsdVertex>::k_InterpolateBoundaryEdgeAndCorner);
		break;
            case 2:
		mesh->SetFVarInterpolateBoundaryMethod(HbrMesh<OsdVertex>::k_InterpolateBoundaryEdgeOnly);
		break;
            case 3:
		mesh->SetFVarInterpolateBoundaryMethod(HbrMesh<OsdVertex>::k_InterpolateBoundaryAlwaysSharp);
		break;
            default:
/*XXX                                                                       
		TF_WARN("Subdivmesh contains unknown facevarying interpolate "
                        "boundary method: %d.\n", currentInt[0]);
*/                        
		break;
	    }
	} else if(strcmp(tag, "smoothtriangles") == 0) {
	    // Do nothing - CatmarkMesh should handle it
	} else if(strcmp(tag, "creasemethod") == 0) {
	    if(nstring < 1) {
/*XXX                                                                   
		TF_WARN("Creasemethod tag missing string argument on SubdivisionMesh.\n");
*/                
	    } else {
                HbrSubdivision<OsdVertex>* subdivisionMethod = mesh->GetSubdivision();
		if(strcmp(currentString->c_str(), "normal") == 0) {
		    subdivisionMethod->SetCreaseSubdivisionMethod(
                        HbrSubdivision<OsdVertex>::k_CreaseNormal);
		} else if(strcmp(currentString->c_str(), "chaikin") == 0) {
		    subdivisionMethod->SetCreaseSubdivisionMethod(
                        HbrSubdivision<OsdVertex>::k_CreaseChaikin);		    
		} else {
/*XXX                                                              
		    TF_WARN("Creasemethod tag specifies unknown crease "
                            "subdivision method '%s' on SubdivisionMesh.\n",
                            currentString->c_str());
*/                            
		}
	    }
	} else if(strcmp(tag, "facevaryingpropagatecorners") == 0) {
	    if(nint != 1) {
/*XXX                                                                     
		TF_WARN("Expecting single integer argument for "
                        "\"facevaryingpropagatecorners\" on SubdivisionMesh.\n");
*/                        
	    } else {
		mesh->SetFVarPropagateCorners(currentInt[0] != 0);
	    }
	} else if(strcmp(tag, "vertexedit") == 0
		  || strcmp(tag, "edgeedit") == 0) {
	    // XXX DO EDITS
/*XXX                                                                                 
            TF_WARN("vertexedit and edgeedit not yet supported.\n");
*/            
	} else {
/*XXX                                                                                             
	    // Complain
            TF_WARN("Unknown tag: %s.\n", tag);
*/                        
	}

	// update the tag data pointers
	currentInt += nint;
	currentFloat += nfloat;
	currentString += nstring;
    }

    std::cout << "Finishing mesh\n";
    mesh->Finish();
}



// Interleave the face-varying sets specified by "names", adding
// floats into the "fvdata" vector.  The number of added floats is:
//    names.size() * NumRefinedFaces * 4
void
PxOsdUtilMesh::GetRefinedFVData(
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
    vector<HbrFace<OsdVertex> *> faces;
    _hmesh->GetFaces(std::back_inserter(faces));

    // Iterate through all faces, filtering on the requested subdivision level.
    for (int i=0; i<(int)faces.size(); ++i) {
        HbrFace<OsdVertex>* face = faces[i];
        if (face->GetDepth() != level) {
            continue;
        }
        int ncorners = face->GetNumVertices();
        for (int corner = 0; corner < ncorners; ++corner) {
            HbrFVarData<OsdVertex>& fvariable = face->GetFVarData(corner);
            
            for (int j=0; j<(int)names.size(); ++j) {
                const string &name = names[j];
            
                int offset = _fvaroffsets[name];
                const float* data = fvariable.GetData(offset);
                outdata->push_back(*data);
            }
        }
    }
}

