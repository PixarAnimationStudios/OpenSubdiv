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

#ifndef HBR_UTILS_H
#define HBR_UTILS_H

#ifndef HBR_ADAPTIVE
#define HBR_ADAPTIVE
#endif

#include "shape_utils.h"

#include <opensubdiv/hbr/mesh.h>
#include <opensubdiv/hbr/bilinear.h>
#include <opensubdiv/hbr/loop.h>
#include <opensubdiv/hbr/catmark.h>
#include <opensubdiv/hbr/vertexEdit.h>
#include <opensubdiv/hbr/cornerEdit.h>
#include <opensubdiv/hbr/holeEdit.h>

#include <sstream>

//------------------------------------------------------------------------------
template <class T>
void applyTags( OpenSubdiv::HbrMesh<T> * mesh, Shape const * sh ) {

    for (int i=0; i<(int)sh->tags.size(); ++i) {
        Shape::tag * t = sh->tags[i];

        if (t->name=="crease") {
            for (int j=0; j<(int)t->intargs.size()-1; j += 2) {
                OpenSubdiv::HbrVertex<T> * v = mesh->GetVertex( t->intargs[j] ),
                                         * w = mesh->GetVertex( t->intargs[j+1] );
                OpenSubdiv::HbrHalfedge<T> * e = 0;
                if( v && w ) {
                    if((e = v->GetEdge(w)) == 0)
                        e = w->GetEdge(v);
                    if(e) {
                        int nfloat = (int) t->floatargs.size();
                        e->SetSharpness( std::max(0.0f, ((nfloat > 1) ? t->floatargs[j] : t->floatargs[0])) );
                    } else
                       printf("cannot find edge for crease tag (%d,%d)\n", t->intargs[j], t->intargs[j+1] );
                }
            }
        } else if (t->name=="corner") {
            for (int j=0; j<(int)t->intargs.size(); ++j) {
                OpenSubdiv::HbrVertex<T> * v = mesh->GetVertex( t->intargs[j] );
                if(v) {
                    int nfloat = (int) t->floatargs.size();
                    v->SetSharpness( std::max(0.0f, ((nfloat > 1) ? t->floatargs[j] : t->floatargs[0])) );
                } else
                   printf("cannot find vertex for corner tag (%d)\n", t->intargs[j] );
            }
        } else if (t->name=="hole") {
            for (int j=0; j<(int)t->intargs.size(); ++j) {
                OpenSubdiv::HbrFace<T> * f = mesh->GetFace( t->intargs[j] );
                if(f) {
                    f->SetHole();
                } else
                   printf("cannot find face for hole tag (%d)\n", t->intargs[j] );
            }
        } else if (t->name=="interpolateboundary") {
            if ((int)t->intargs.size()!=1) {
                printf("expecting 1 integer for \"interpolateboundary\" tag n. %d\n", i);
                continue;
            }
            switch( t->intargs[0] ) {
                case 0 : mesh->SetInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryNone); break;
                case 1 : mesh->SetInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeAndCorner); break;
                case 2 : mesh->SetInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly); break;
                default: printf("unknown interpolate boundary : %d\n", t->intargs[0] ); break;
            }
        } else if (t->name=="facevaryinginterpolateboundary") {
            if ((int)t->intargs.size()!=1) {
                printf("expecting 1 integer for \"facevaryinginterpolateboundary\" tag n. %d\n", i);
                continue;
            }
            switch( t->intargs[0] ) {
                case 0 : mesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryNone); break;
                case 1 : mesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeAndCorner); break;
                case 2 : mesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly); break;
                case 3 : mesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryAlwaysSharp); break;
                default: printf("unknown facevarying interpolate boundary : %d\n", t->intargs[0] ); break;
            }
        } else if (t->name=="facevaryingpropagatecorners") {
            if ((int)t->intargs.size()==1)
                mesh->SetFVarPropagateCorners( t->intargs[0] != 0 );
            else
                printf( "expecting single int argument for \"facevaryingpropagatecorners\"\n" );
        } else if (t->name=="smoothtriangles") {

            OpenSubdiv::HbrCatmarkSubdivision<T> * scheme =
                dynamic_cast<OpenSubdiv::HbrCatmarkSubdivision<T> *>( mesh->GetSubdivision() );

            if (! scheme) {
                printf("the \"smoothtriangles\" tag can only be applied to Catmark meshes\n");
                continue;
            }

            if ((int)t->intargs.size()==0) {
                printf("the \"smoothtriangles\" tag expects an int argument\n");
                continue;
            }

            if( t->intargs[0]==1 )
                scheme->SetTriangleSubdivisionMethod(
                    OpenSubdiv::HbrCatmarkSubdivision<T>::k_Old);
            else if( t->intargs[0]==2 )
                scheme->SetTriangleSubdivisionMethod(
                    OpenSubdiv::HbrCatmarkSubdivision<T>::k_New);
            else
                printf("the \"smoothtriangles\" tag only accepts 1 or 2 as value (%d)\n", t->intargs[0]);

        } else if (t->name=="creasemethod") {

            OpenSubdiv::HbrSubdivision<T> * scheme = mesh->GetSubdivision();

            assert(scheme);

            if ((int)t->stringargs.size()==0) {
                printf("the \"creasemethod\" tag expects a string argument\n");
                continue;
            }

            if( t->stringargs[0]=="normal" )
                scheme->SetCreaseSubdivisionMethod(
                    OpenSubdiv::HbrSubdivision<T>::k_CreaseNormal);
            else if( t->stringargs[0]=="chaikin" )
                scheme->SetCreaseSubdivisionMethod(
                    OpenSubdiv::HbrSubdivision<T>::k_CreaseChaikin);
            else
                printf("the \"creasemethod\" tag only accepts \"normal\" or \"chaikin\" as value (%s)\n", t->stringargs[0].c_str());

        } else if (t->name=="vertexedit" || t->name=="edgeedit") {
            int nops = 0;
            int floatstride = 0;
            int maxfloatwidth = 0;
            std::vector<typename OpenSubdiv::HbrHierarchicalEdit<T>::Operation > ops;
            std::vector<std::string> opnames;
            std::vector<std::string> varnames;
            std::vector<typename OpenSubdiv::HbrHierarchicalEdit<T>::Operation > opmodifiers;
            std::vector<int> floatwidths;
            std::vector<bool> isP;
            std::vector<int> vvindex;

            for (int j=0; j<(int)t->stringargs.size(); j+=3) {
                const std::string & opname = t->stringargs[j+2];
                const std::string & opmodifiername = t->stringargs[j];
                const std::string & varname = t->stringargs[j+1];

                typename OpenSubdiv::HbrHierarchicalEdit<T>::Operation opmodifier = OpenSubdiv::HbrVertexEdit<T>::Set;
                if (opmodifiername == "set") {
                    opmodifier = OpenSubdiv::HbrHierarchicalEdit<T>::Set;
                } else if (opmodifiername == "add") {
                    opmodifier = OpenSubdiv::HbrHierarchicalEdit<T>::Add;
                } else if (opmodifiername == "subtract") {
                    opmodifier = OpenSubdiv::HbrHierarchicalEdit<T>::Subtract;
                } else {
                    printf("invalid modifier %s\n", opmodifiername.c_str());
                    continue;
                }

                if ((t->name=="vertexedit" && opname=="value") || opname=="sharpness") {
                    nops++;

                    // only varname="P" is supported here for now.
                    if (varname != "P") continue;

                    vvindex.push_back(0);
                    isP.push_back(true);
                    opnames.push_back(opname);
                    opmodifiers.push_back(opmodifier);
                    varnames.push_back(varname);

                    if (opname=="sharpness") {
                        floatwidths.push_back(1);
                        floatstride += 1;
                    } else {
                        // assuming width of P == 3. should be replaced with 'P 0 3' like declaration
                        int numElements = 3;
                        maxfloatwidth = std::max(maxfloatwidth, numElements);
                        floatwidths.push_back(numElements);
                        floatstride += numElements;
                    }
                } else {
                    printf("%s tag specifies invalid operation '%s %s' on Subdivmesh\n", t->name.c_str(), opmodifiername.c_str(), opname.c_str());
                }
            }

            float *xformed = (float*)alloca(maxfloatwidth * sizeof(float));

            int floatoffset = 0;
            for(int j=0; j<nops; ++j) {
                int floatidx = floatoffset;
                for (int k=0; k < (int)t->intargs.size();) {
                    int pathlength = t->intargs[k];

                    int faceid = t->intargs[k+1];
                    int vertexid = t->intargs[k+pathlength];
                    int nsubfaces = pathlength - 2;
                    int *subfaces = &t->intargs[k+2];
                    OpenSubdiv::HbrFace<T> * f = mesh->GetFace(faceid);
                    if (!f) {
                        printf("Invalid face %d specified for %s tag on SubdivisionMesh.\n", faceid, t->name.c_str());
                        goto nexttag;
                    }
                    // Found the face. Do some preliminary error checking to make sure the path is
                    // correct.  First value in path depends on the number of vertices of the face
                    // which we have in hand
                    if (nsubfaces && (subfaces[0] < 0 || subfaces[0] >= f->GetNumVertices()) ) {
                        printf("Invalid path component %d in %s tag on SubdivisionMesh.\n", subfaces[0], t->name.c_str());
                        goto nexttag;
                    }

                    // All subsequent values must be less than 4 (FIXME or 3 in the loop case?)
                    for (int l=1; l<nsubfaces; ++l) {
                        if (subfaces[l] < 0 || subfaces[l] > 3) {
                            printf("Invalid path component %d in %s tag on SubdivisionMesh.\n", subfaces[0], t->name.c_str());
                            goto nexttag;
                        }
                    }
                    if (vertexid < 0 || vertexid > 3) {
                        printf("Invalid path component (vertexid) %d in %s tag on SubdivisionMesh.\n", vertexid, t->name.c_str());
                        goto nexttag;
                    }

                    // Transform all the float values associated with the tag if needed
                    if(opnames[j] != "sharpness") {
                        for(int l=0; l<floatwidths[j]; ++l) {
                            xformed[l] = t->floatargs[l + floatidx];
                        }

                        // Edits of facevarying data are a different hierarchical edit type altogether
                        OpenSubdiv::HbrVertexEdit<T> * edit = new OpenSubdiv::HbrVertexEdit<T>(faceid, nsubfaces, subfaces,
                                                                                               vertexid, vvindex[j], floatwidths[j],
                                                                                               isP[j], opmodifiers[j], xformed);
                        mesh->AddHierarchicalEdit(edit);
                    } else {
                        if (t->name == "vertexedit") {
                            OpenSubdiv::HbrCornerEdit<T> * edit = new OpenSubdiv::HbrCornerEdit<T>(faceid, nsubfaces, subfaces,
                                                                                                   vertexid, opmodifiers[j], t->floatargs[floatidx]);
                            mesh->AddHierarchicalEdit(edit);
                        } else {
                            OpenSubdiv::HbrCreaseEdit<T> * edit = new OpenSubdiv::HbrCreaseEdit<T>(faceid, nsubfaces, subfaces,
                                                                                                   vertexid, opmodifiers[j], t->floatargs[floatidx]);
                            mesh->AddHierarchicalEdit(edit);
                        }
                    }

                    // Advance to next path
                    k += pathlength + 1;

                    // Advance to start of float data
                    floatidx += floatstride;
                } // End of integer processing loop

                // Next subop
                floatoffset += floatwidths[j];

            } // End of subop processing loop
        } else if (t->name=="faceedit") {

            int nint = (int)t->intargs.size();
            for (int k=0; k<nint; ) {
                int pathlength = t->intargs[k];

                if (k+pathlength>=nint) {
                    printf("Invalid path length for %s tag on SubdivisionMesh", t->name.c_str());
                    goto nexttag;
                }

                int faceid = t->intargs[k+1];
                int nsubfaces = pathlength - 1;
                int *subfaces = &t->intargs[k+2];
                OpenSubdiv::HbrFace<T> * f = mesh->GetFace(faceid);
                if (!f) {
                    printf("Invalid face %d specified for %s tag on SubdivisionMesh.\n", faceid, t->name.c_str());
                    goto nexttag;
                }

                // Found the face. Do some preliminary error checking to make sure the path is
                // correct.  First value in path depends on the number of vertices of the face
                // which we have in hand
                if (nsubfaces && (subfaces[0] < 0 || subfaces[0] >= f->GetNumVertices()) ) {
                    printf("Invalid path component %d in %s tag on SubdivisionMesh.\n", subfaces[0], t->name.c_str());
                    goto nexttag;
                }

                // All subsequent values must be less than 4 (FIXME or 3 in the loop case?)
                for (int l=1; l<nsubfaces; ++l) {
                    if (subfaces[l] < 0 || subfaces[l] > 3) {
                        printf("Invalid path component %d in %s tag on SubdivisionMesh.\n", subfaces[0], t->name.c_str());
                        goto nexttag;
                    }
                }

                // Now loop over string ops
                int nstring = (int)t->stringargs.size();
                for (int l = 0; l < nstring; ) {
                    if ( t->stringargs[l] == "hole" ) {
                        // Construct the edit
                        OpenSubdiv::HbrHoleEdit<T> * edit = new OpenSubdiv::HbrHoleEdit<T>(faceid, nsubfaces, subfaces);
                        mesh->AddHierarchicalEdit(edit);
                        ++l;
                    } else if ( t->stringargs[l] == "attributes" ) {
                        // see NgpSubdivMesh.cpp:4341
                        printf("\"attributes\" face tag not supported yet.\n");
                        goto nexttag;
                    } else if ( t->stringargs[l] == "set" || t->stringargs[l] == "add" ) {
                        // see NgpSubdivMesh.cpp:4341
                        printf("\"set\" and \"add\" face tag not supported yet.\n");
                        goto nexttag;
                    } else {
                        printf("Faceedit tag specifies invalid operation '%s' on Subdivmesh.\n", t->stringargs[l].c_str());
                        goto nexttag;
                    }
                }
                // Advance to next path
                k += pathlength + 1;
            } // end face path loop

        } else {
            printf("Unknown tag : \"%s\" - skipping\n", t->name.c_str());
        }
nexttag: ;
    }
}

//------------------------------------------------------------------------------
template <class T> std::string
hbrToObj( OpenSubdiv::HbrMesh<T> * mesh ) {

    std::stringstream sh;

    sh<<"# This file uses centimeters as units for non-parametric coordinates.\n\n";

    int nv = mesh->GetNumVertices();
    for (int i=0; i<nv; ++i) {
       const float * pos = mesh->GetVertex(i)->GetData().GetPos();
       sh << "v " << pos[0] << " " << pos[1] << " " << pos[2] <<"\n";
    }

    int nf = mesh->GetNumFaces();
    for (int i=0; i<nf; ++i) {

        sh << "f ";

        OpenSubdiv::HbrFace<T> * f = mesh->GetFace(i);

        for (int j=0; j<f->GetNumVertices(); ++j) {
            int vert = f->GetVertex(j)->GetID()+1;
            sh << vert << "/" << vert << "/" << vert << " ";
        }
        sh << "\n";
    }

    sh << "\n";

    return sh.str();
}

//------------------------------------------------------------------------------
template <class T> OpenSubdiv::HbrMesh<T> *
createMesh( Scheme scheme=kCatmark, int fvarwidth=0) {

  OpenSubdiv::HbrMesh<T> * mesh = 0;

  static OpenSubdiv::HbrBilinearSubdivision<T> _bilinear;
  static OpenSubdiv::HbrLoopSubdivision<T>     _loop;
  static OpenSubdiv::HbrCatmarkSubdivision<T>  _catmark;

  static int indices[2] = { 0, 1 },
             widths[2] = { 1, 1 };

  int const   fvarcount   = fvarwidth > 0 ? 2 : 0,
            * fvarindices = fvarwidth > 0 ? indices : NULL,
            * fvarwidths  = fvarwidth > 0 ? widths : NULL;


  switch (scheme) {
    case kBilinear : mesh = new OpenSubdiv::HbrMesh<T>( &_bilinear,
                                                        fvarcount,
                                                        fvarindices,
                                                        fvarwidths,
                                                        fvarwidth ); break;

    case kLoop     : mesh = new OpenSubdiv::HbrMesh<T>( &_loop,
                                                        fvarcount,
                                                        fvarindices,
                                                        fvarwidths,
                                                        fvarwidth ); break;

    case kCatmark  : mesh = new OpenSubdiv::HbrMesh<T>( &_catmark,
                                                        fvarcount,
                                                        fvarindices,
                                                        fvarwidths,
                                                        fvarwidth ); break;
  }

  return mesh;
}

//------------------------------------------------------------------------------
template <class T> void
createVerticesWithPositions(Shape const * sh, OpenSubdiv::HbrMesh<T> * mesh) {

    T v;
    for(int i=0;i<sh->GetNumVertices(); i++ ) {
        v.SetPosition( sh->verts[i*3], sh->verts[i*3+1], sh->verts[i*3+2] );
        mesh->NewVertex( i, v );
    }
}

//------------------------------------------------------------------------------
template <class T> void
createVertices(Shape const * sh, OpenSubdiv::HbrMesh<T> * mesh) {

    T v;
    for(int i=0;i<sh->GetNumVertices(); i++ )
        mesh->NewVertex( i, v );
}

//------------------------------------------------------------------------------
template <class T> void
copyVertexPositions( Shape const * sh, OpenSubdiv::HbrMesh<T> * mesh, std::vector<float> & verts ) {

    int nverts = mesh->GetNumVertices();

    verts.resize( nverts * 3 );

    std::copy(sh->verts.begin(), sh->verts.end(), verts.begin());

    // Sometimes Hbr dupes some vertices during Mesh::Finish() and our example
    // code uses those vertices to draw coarse control cages and such
    std::vector<std::pair<int, int> > const splits = mesh->GetSplitVertices();
    for (int i=0; i<(int)splits.size(); ++i) {
        memcpy(&verts[splits[i].first*3], &sh->verts[splits[i].second*3], 3*sizeof(float));
    }
}

//------------------------------------------------------------------------------
template <class T> void
createTopology( Shape const * sh, OpenSubdiv::HbrMesh<T> * mesh, Scheme scheme) {

    const int * fv=&(sh->faceverts[0]);
    for(int f=0, ptxidx=0;f<sh->GetNumFaces(); f++ ) {

        int nv = sh->nvertsPerFace[f];

        if ((scheme==kLoop) && (nv!=3)) {
            printf("Trying to create a Loop subd with non-triangle face\n");
            exit(1);
        }

        bool valid = true;

        for(int j=0;j<nv;j++) {
            OpenSubdiv::HbrVertex<T> * origin      = mesh->GetVertex( fv[j] );
            OpenSubdiv::HbrVertex<T> * destination = mesh->GetVertex( fv[(j+1)%nv] );
            OpenSubdiv::HbrHalfedge<T> * opposite  = destination->GetEdge(origin);

            if(origin==NULL || destination==NULL) {
                printf(" An edge was specified that connected a nonexistent vertex\n");
                valid=false;
                break;
            }

            if(origin == destination) {
                printf(" An edge was specified that connected a vertex to itself\n");
                valid=false;
                break;
            }

            if(opposite && opposite->GetOpposite() ) {
                printf(" A non-manifold edge incident to more than 2 faces was found\n");
                valid=false;
                break;
            }

            if(origin->GetEdge(destination)) {
                printf(" An edge connecting two vertices was specified more than once."
                       " It's likely that an incident face was flipped\n");
                valid=false;
                break;
            }
        }

        if (valid) {

            OpenSubdiv::HbrFace<T> * face = mesh->NewFace(nv, (int *)fv, 0);

            face->SetPtexIndex(ptxidx);

            if ( (scheme==kCatmark || scheme==kBilinear) && nv != 4 ) {
                ptxidx+=nv;
            } else {
                ptxidx++;
            }
        }

        fv+=nv;
    }

    mesh->SetInterpolateBoundaryMethod(
        OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly);

    mesh->GetSubdivision()->SetCreaseSubdivisionMethod(
        OpenSubdiv::HbrSubdivision<T>::k_CreaseNormal);

    if (OpenSubdiv::HbrCatmarkSubdivision<T> * hscheme =
        dynamic_cast<OpenSubdiv::HbrCatmarkSubdivision<T> *>(mesh->GetSubdivision())) {

        hscheme->SetTriangleSubdivisionMethod(
            OpenSubdiv::HbrCatmarkSubdivision<T>::k_Normal);
    }

    applyTags<T>( mesh, sh );

    mesh->Finish();

    // check for disconnected vertices
    if (mesh->GetNumDisconnectedVertices()) {
        printf("The specified subdivmesh contains disconnected surface components.\n");
    }
}

//------------------------------------------------------------------------------
template <class T> void
createFaceVaryingUV( Shape const * sh, OpenSubdiv::HbrMesh<T> * mesh) {

    if (! sh->HasUV())
        return;

    for (int i=0, idx=0; i<sh->GetNumFaces(); ++i ) {

        OpenSubdiv::HbrFace<T> * f = mesh->GetFace(i);

        int nv = sh->nvertsPerFace[i];

        OpenSubdiv::HbrHalfedge<T> * e = f->GetFirstEdge();

        for (int j=0; j<nv; ++j, e=e->GetNext()) {

            OpenSubdiv::HbrFVarData<T> & fvt = e->GetOrgVertex()->GetFVarData(f);

            float const * fvdata = &sh->uvs[ sh->faceuvs[idx++]*2 ];

            if (! fvt.IsInitialized()) {
                fvt.SetAllData(2, fvdata);
            } else if (! fvt.CompareAll(2, fvdata)) {
                OpenSubdiv::HbrFVarData<T> & nfvt = e->GetOrgVertex()->NewFVarData(f);
                nfvt.SetAllData(2, fvdata);
            }
        }
    }
}

//------------------------------------------------------------------------------
template <class T> OpenSubdiv::HbrMesh<T> *
simpleHbr(Shape const * sh, std::vector<float> * verts=0, bool fvar=false) {

    int fvarwidth = fvar && sh->HasUV() ? 2 : 0;

    OpenSubdiv::HbrMesh<T> * mesh = createMesh<T>(sh->scheme, fvarwidth);

    createVerticesWithPositions<T>(sh, mesh);

    createTopology<T>(sh, mesh, sh->scheme);

    if (fvar)
        createFaceVaryingUV<T>(sh, mesh);

    if (verts)
        copyVertexPositions<T>(sh, mesh, *verts);

    return mesh;
}

template <class T> OpenSubdiv::HbrMesh<T> *
simpleHbr(char const * Shapestr, Scheme scheme, std::vector<float> * verts=0, bool fvar=false) {

    Shape const * sh = Shape::parseObj( Shapestr, scheme );

    OpenSubdiv::HbrMesh<T> * mesh = simpleHbr<T>(sh, verts, fvar);

    delete sh;
    return mesh;
}

//------------------------------------------------------------------------------
template <class T> OpenSubdiv::HbrMesh<T> *
simpleHbr(Shape const * sh, std::vector<float> & verts, bool fvar=false) {

    int fvarwidth = fvar && sh->HasUV() ? 2 : 0;

    OpenSubdiv::HbrMesh<T> * mesh = createMesh<T>(sh->scheme, fvarwidth);

    createVertices<T>(sh, mesh);

    createTopology<T>(sh, mesh, sh->scheme);

    if (fvar)
        createFaceVaryingUV<T>(sh, mesh);

    copyVertexPositions<T>(sh, mesh, verts);

    return mesh;
}

template <class T> OpenSubdiv::HbrMesh<T> *
simpleHbr(char const * Shapestr, Scheme scheme, std::vector<float> & verts, bool fvar=false) {

    Shape const * sh = Shape::parseObj( Shapestr, scheme );

    OpenSubdiv::HbrMesh<T> *mesh = simpleHbr<T>(sh, verts, fvar);

    delete sh;
    return mesh;
}

//------------------------------------------------------------------------------
template <class T>
OpenSubdiv::HbrMesh<T> *
interpolateHbrVertexData(Shape const * sh, int maxlevel) {

    // Hbr interpolation
    OpenSubdiv::HbrMesh<T> *hmesh = simpleHbr<T>(sh, /* verts vector */ 0, /* fvar */ false);
    assert(hmesh);

    for (int level=0, firstface=0; level<maxlevel; ++level ) {
        int nfaces = hmesh->GetNumFaces();
        for (int i=firstface; i<nfaces; ++i) {

            OpenSubdiv::HbrFace<T> * f = hmesh->GetFace(i);
            assert(f->GetDepth()==level);
            if (! f->IsHole()) {
                f->Refine();
            }
        }
        // Hbr allocates faces sequentially, skip faces that have already been
        // refined.
        firstface = nfaces;
    }
    return hmesh;
}

template <class T>
OpenSubdiv::HbrMesh<T> *
interpolateHbrVertexData(char const * Shapestr, Scheme scheme, int maxlevel) {

    Shape const * sh = Shape::parseObj( Shapestr, scheme );

    OpenSubdiv::HbrMesh<T> *mesh = interpolateHbrVertexData<T>(sh, maxlevel);

    delete sh;
    return mesh;
}

//------------------------------------------------------------------------------
// Returns true if a vertex or any of its parents is on a boundary
template <class T>
bool
hbrVertexOnBoundary(const OpenSubdiv::HbrVertex<T> *v)
{
    if (! v)
        return false;

    if (v->OnBoundary())
        return true;

    OpenSubdiv::HbrVertex<T> const * pv = v->GetParentVertex();
    if (pv)
        return hbrVertexOnBoundary(pv);
    else {
        OpenSubdiv::HbrHalfedge<T> const * pe = v->GetParentEdge();
        if (pe) {
              return hbrVertexOnBoundary(pe->GetOrgVertex()) ||
                     hbrVertexOnBoundary(pe->GetDestVertex());
        } else {
            OpenSubdiv::HbrFace<T> const * pf = v->GetParentFace(), * rootf = pf;
            while (pf) {
                pf = pf->GetParent();
                if (pf)
                    rootf=pf;
            }
            if (rootf)
                for (int i=0; i<rootf->GetNumVertices(); ++i)
                    if (rootf->GetVertex(i)->OnBoundary())
                        return true;
        }
    }
    return false;
}


#endif /* HBR_UTILS_H */
