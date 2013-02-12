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
#ifndef SHAPE_UTILS_H
#define SHAPE_UTILS_H

#include <hbr/mesh.h>
#include <hbr/bilinear.h>
#include <hbr/loop.h>
#include <hbr/catmark.h>
#include <hbr/vertexEdit.h>
#include <hbr/cornerEdit.h>

#include <stdio.h>
#include <string.h>

#include <list>
#include <string>
#include <sstream>
#include <vector>

//------------------------------------------------------------------------------
static char const * sgets( char * s, int size, char ** stream ) {
    for (int i=0; i<size; ++i) {
        if ( (*stream)[i]=='\n' or (*stream)[i]=='\0') {

            memcpy(s, *stream, i);
            s[i]='\0';

            if ((*stream)[i]=='\0')
                return 0;
            else {
                (*stream) += i+1;
                return s;
            }
        }
    }
    return 0;
}


//------------------------------------------------------------------------------
enum Scheme {
  kBilinear,
  kCatmark,
  kLoop
};

//------------------------------------------------------------------------------

struct shape {

    struct tag {

        static tag * parseTag( char const * stream );
        
        std::string genTag() const;

        std::string              name;
        std::vector<int>         intargs;
        std::vector<float>       floatargs;
        std::vector<std::string> stringargs;
    };

    static shape * parseShape(char const * shapestr, int axis=1);
    
    std::string genShape(char const * name) const;

    std::string genObj(char const * name) const;
 
    std::string genRIB() const;

    ~shape();

    int getNverts() const { return (int)verts.size()/3; }

    int getNfaces() const { return (int)nvertsPerFace.size(); }

    std::vector<float>  verts;
    std::vector<float>  uvs;
    std::vector<float>  normals;
    std::vector<int>    nvertsPerFace;
    std::vector<int>    faceverts;
    std::vector<int>    faceuvs;
    std::vector<int>    facenormals;
    std::vector<tag *>  tags;
    Scheme              scheme;
};

//------------------------------------------------------------------------------
shape::~shape() {
    for (int i=0; i<(int)tags.size(); ++i)
        delete tags[i];
}

//------------------------------------------------------------------------------
shape::tag * shape::tag::parseTag(char const * line) {
    tag * t = 0;

    const char* cp = &line[2];

    char name[50];
    while (*cp == ' ') cp++;
    if (sscanf(cp, "%s", name )!=1) return t;
    while (*cp && *cp != ' ') cp++;

    int nints=0, nfloats=0, nstrings=0;
    while (*cp == ' ') cp++;
    if (sscanf(cp, "%d/%d/%d", &nints, &nfloats, &nstrings)!=3) return t;
    while (*cp && *cp != ' ') cp++;

    std::vector<int> intargs;
    for (int i=0; i<nints; ++i) {
        int val;
        while (*cp == ' ') cp++;
        if (sscanf(cp, "%d", &val)!=1) return t;
        intargs.push_back(val);
        while (*cp && *cp != ' ') cp++;
    }

    std::vector<float> floatargs;
    for (int i=0; i<nfloats; ++i) {
        float val;
        while (*cp == ' ') cp++;
        if (sscanf(cp, "%f", &val)!=1) return t;
        floatargs.push_back(val);
        while (*cp && *cp != ' ') cp++;
    }

    std::vector<std::string> stringargs;
    for (int i=0; i<nstrings; ++i) {
        char val[512];
        while (*cp == ' ') cp++;
        if (sscanf(cp, "%s", val)!=1) return t;
        stringargs.push_back(val);
        while (*cp && *cp != ' ') cp++;
    }

    t = new shape::tag;
    t->name = name;
    t->intargs = intargs;
    t->floatargs = floatargs;
    t->stringargs = stringargs;

    return t;
}

//------------------------------------------------------------------------------
std::string shape::tag::genTag() const {
    std::stringstream t;

    t<<"\"t \""<<name<<"\" ";
    
    t<<intargs.size()<<"/"<<floatargs.size()<<"/"<<stringargs.size()<<" ";

    std::copy(intargs.begin(), intargs.end(), std::ostream_iterator<int>(t));
    t<<" ";

    std::copy(floatargs.begin(), floatargs.end(), std::ostream_iterator<float>(t));
    t<<" ";

    std::copy(stringargs.begin(), stringargs.end(), std::ostream_iterator<std::string>(t));
    t<<"\\n\"\n";
    
    return t.str();
}

//------------------------------------------------------------------------------
std::string shape::genShape(char const * name) const {
    std::stringstream sh;
    
    sh<<"static char const * "<<name<<" = \n";
    
    for (int i=0; i<(int)verts.size(); i+=3)
       sh << "\"v " << verts[i] << " " << verts[i+1] << " " << verts[i+2] <<"\\n\"\n";

    for (int i=0; i<(int)uvs.size(); i+=2)
       sh << "\"vt " << uvs[i] << " " << uvs[i+1] << "\\n\"\n";

    for (int i=0; i<(int)normals.size(); i+=3)
       sh << "\"vn " << normals[i] << " " << normals[i+1] << " " << normals[i+2] <<"\\n\"\n";

    sh << "\"s off\\n\"\n";

    for (int i=0, idx=0; i<(int)nvertsPerFace.size();++i) {
        sh << "\"f ";
        for (int j=0; j<nvertsPerFace[i];++j) {
            int vert = faceverts[idx+j]+1,
                uv = (int)faceuvs.size()>0 ? faceuvs[idx+j]+1 : vert,
                normal = (int)facenormals.size()>0 ? facenormals[idx+j]+1 : vert;
            sh << vert << "/" << uv << "/" << normal << " ";
        }
        sh << "\\n\"\n";
        idx+=nvertsPerFace[i];
    }

    for (int i=0; i<(int)tags.size(); ++i)
        sh << tags[i]->genTag();
        
    return sh.str();
}

//------------------------------------------------------------------------------
std::string shape::genObj(char const * name) const {
    std::stringstream sh;
            
    sh<<"# This file uses centimeters as units for non-parametric coordinates.\n\n";
    
    for (int i=0; i<(int)verts.size(); i+=3)
       sh << "v " << verts[i] << " " << verts[i+1] << " " << verts[i+2] <<"\n";

    for (int i=0; i<(int)uvs.size(); i+=2)
       sh << "vt " << uvs[i] << " " << uvs[i+1] << "\n";

    for (int i=0; i<(int)normals.size(); i+=3)
       sh << "vn " << normals[i] << " " << normals[i+1] << " " << normals[i+2] <<"\n";

    for (int i=0, idx=0; i<(int)nvertsPerFace.size();++i) {
        sh << "f ";
        for (int j=0; j<nvertsPerFace[i];++j) {
            int vert = faceverts[idx+j]+1,
                uv = (int)faceuvs.size()>0 ? faceuvs[idx+j]+1 : vert,
                normal = (int)facenormals.size()>0 ? facenormals[idx+j]+1 : vert;
            sh << vert << "/" << uv << "/" << normal << " ";
        }
        sh << "\n";
        idx+=nvertsPerFace[i];
    }

    for (int i=0; i<(int)tags.size(); ++i)
        sh << tags[i]->genTag();
        
    return sh.str();
}

//------------------------------------------------------------------------------
std::string shape::genRIB() const {
    std::stringstream rib;
    
    rib << "HierarchicalSubdivisionMesh \"catmull-clark\" ";
    
    rib << "[";
    std::copy(nvertsPerFace.begin(), nvertsPerFace.end(), std::ostream_iterator<int>(rib));
    rib << "] ";

    rib << "[";
    std::copy(faceverts.begin(), faceverts.end(), std::ostream_iterator<int>(rib));
    rib << "] ";
    
    std::stringstream names, nargs, intargs, floatargs, strargs;
    for (int i=0; i<(int)tags.size();) {
        tag * t = tags[i];
        
        names << t->name;

        nargs << t->intargs.size() << " " << t->floatargs.size() << " " << t->stringargs.size();
        
        std::copy(t->intargs.begin(), t->intargs.end(), std::ostream_iterator<int>(intargs));

        std::copy(t->floatargs.begin(), t->floatargs.end(), std::ostream_iterator<float>(floatargs));

        std::copy(t->stringargs.begin(), t->stringargs.end(), std::ostream_iterator<std::string>(strargs));
        
        if (++i<(int)tags.size()) {
            names << " ";
            nargs << " ";
            intargs << " ";
            floatargs << " ";
            strargs << " ";
        }
    }
    
    rib << "["<<names<<"] " << "["<<nargs<<"] " << "["<<intargs<<"] " << "["<<floatargs<<"] " << "["<<strargs<<"] ";

    rib << "\"P\" [";
    std::copy(verts.begin(), verts.end(), std::ostream_iterator<float>(rib));
    rib << "] ";
    
    return rib.str();
}

//------------------------------------------------------------------------------
shape * shape::parseShape(char const * shapestr, int axis ) {

    shape * s = new shape;

    char * str=const_cast<char *>(shapestr), line[256];
    bool done = false;
    while( not done )
    {   done = sgets(line, sizeof(line), &str)==0;
        char* end = &line[strlen(line)-1];
        if (*end == '\n') *end = '\0'; // strip trailing nl
        float x, y, z, u, v;
        switch (line[0]) {
            case 'v': switch (line[1])
                      {       case ' ': if(sscanf(line, "v %f %f %f", &x, &y, &z) == 3) {
                                             s->verts.push_back(x);
                                             switch( axis ) {
                                                 case 0 : s->verts.push_back(-z);
                                                          s->verts.push_back(y); break;
                                                 case 1 : s->verts.push_back(y);
                                                          s->verts.push_back(z); break;
                                             } 
                                        } break;
                              case 't': if(sscanf(line, "vt %f %f", &u, &v) == 2) {
                                            s->uvs.push_back(u);
                                            s->uvs.push_back(v);
                                        } break;
                              case 'n' : if(sscanf(line, "vn %f %f %f", &x, &y, &z) == 3) {
                                            s->normals.push_back(x);
                                            s->normals.push_back(y);
                                            s->normals.push_back(z);
                                         }
                                         break; // skip normals for now
                          }
                          break;
            case 'f': if(line[1] == ' ') {
                              int vi, ti, ni;
                              const char* cp = &line[2];
                              while (*cp == ' ') cp++;
                              int nverts = 0, nitems=0;
                              while( (nitems=sscanf(cp, "%d/%d/%d", &vi, &ti, &ni))>0) {
                                  nverts++;
                                  s->faceverts.push_back(vi-1);
                                  if(nitems >= 1) s->faceuvs.push_back(ti-1);
                                  if(nitems >= 2) s->facenormals.push_back(ni-1);
                                  while (*cp && *cp != ' ') cp++;
                                  while (*cp == ' ') cp++;
                              }
                              s->nvertsPerFace.push_back(nverts);
                          }
                          break;
            case 't' : if(line[1] == ' ') {
                           shape::tag * t = tag::parseTag( line );
                           if (t)
                               s->tags.push_back(t);
                       } break;
        }
    }
    return s;
}

//------------------------------------------------------------------------------
template <class T>
void applyTags( OpenSubdiv::HbrMesh<T> * mesh, shape const * sh ) {

    for (int i=0; i<(int)sh->tags.size(); ++i) {
        shape::tag * t = sh->tags[i];

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
                default: printf("unknown interpolation boundary : %d\n", t->intargs[0] ); break;
            }
        } else if (t->name=="facevaryingpropagatecorners") {
            if ((int)t->intargs.size()==1)
                mesh->SetFVarPropagateCorners( t->intargs[0] != 0 );
            else
                printf( "expecting single int argument for \"facevaryingpropagatecorners\"\n" );
        } else if (t->name=="creasemethod") {

            OpenSubdiv::HbrCatmarkSubdivision<T> * scheme =
                dynamic_cast<OpenSubdiv::HbrCatmarkSubdivision<T> *>( mesh->GetSubdivision() );

            if (not scheme) {
                printf("the \"creasemethod\" tag can only be applied to Catmark meshes\n");
                continue;
            }

            if ((int)t->stringargs.size()==0) {
                printf("the \"creasemethod\" tag expects a string argument\n");
                continue;
            }

            if( t->stringargs[0]=="normal" )
                scheme->SetTriangleSubdivisionMethod(
                    OpenSubdiv::HbrCatmarkSubdivision<T>::k_Old);
            else if( t->stringargs[0]=="chaikin" )
                scheme->SetTriangleSubdivisionMethod(
                    OpenSubdiv::HbrCatmarkSubdivision<T>::k_New);
            else
                printf("the \"creasemethod\" tag only accepts \"normal\" or \"chaikin\" as value (%s)\n", t->stringargs[0].c_str());

        } else if (t->name=="vertexedit" or t->name=="edgeedit") {
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
            printf("hierarchical face edits not supported (yet)\n");
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
createMesh( Scheme scheme=kCatmark) {

  OpenSubdiv::HbrMesh<T> * mesh = 0;

  static OpenSubdiv::HbrBilinearSubdivision<T> _bilinear;
  static OpenSubdiv::HbrLoopSubdivision<T>     _loop;
  static OpenSubdiv::HbrCatmarkSubdivision<T>  _catmark;

  switch (scheme) {
    case kBilinear : mesh = new OpenSubdiv::HbrMesh<T>( &_bilinear ); break;
    case kLoop     : mesh = new OpenSubdiv::HbrMesh<T>( &_loop     ); break;
    case kCatmark  : mesh = new OpenSubdiv::HbrMesh<T>( &_catmark  ); break;
  }

  return mesh;
}

//------------------------------------------------------------------------------
template <class T> void
createVertices( shape const * sh, OpenSubdiv::HbrMesh<T> * mesh, std::vector<float> * verts ) {

    T v;
    for(int i=0;i<sh->getNverts(); i++ ) {
        v.SetPosition( sh->verts[i*3], sh->verts[i*3+1], sh->verts[i*3+2] );
        mesh->NewVertex( i, v );
    }
}

//------------------------------------------------------------------------------
template <class T> void
createVertices( shape const * sh, OpenSubdiv::HbrMesh<T> * mesh, std::vector<float> & verts ) {

    T v;
    for(int i=0;i<sh->getNverts(); i++ )
        mesh->NewVertex( i, v );
}

//------------------------------------------------------------------------------
template <class T> void
copyVertexPositions( shape const * sh, OpenSubdiv::HbrMesh<T> * mesh, std::vector<float> & verts ) {

    int nverts = mesh->GetNumVertices();
    
    verts.resize( nverts * 3 );
    
    std::copy(sh->verts.begin(), sh->verts.end(), verts.begin());
    
    // Sometimes Hbr dupes some vertices during Mesh::Finish()
    if (nverts > sh->getNverts()) {
    
        for (int i=sh->getNverts(); i<nverts; ++i) {
        
            OpenSubdiv::HbrVertex<T> * v = mesh->GetVertex(i);
            
            OpenSubdiv::HbrFace<T> * f = v->GetIncidentEdge()->GetFace();
            
            int vidx = -1;
            for (int j=0; j<f->GetNumVertices(); ++j)
                if (f->GetVertex(j)==v) {
                    vidx = j;
                    break;
                }
            assert(vidx>-1);
        
            const int * shfaces = &sh->faceverts[0];
            for (int j=0; j<f->GetID(); ++j)
                shfaces += sh->nvertsPerFace[j];
        
            int shvert = shfaces[vidx];
            
            verts[i*3+0] = sh->verts[shvert*3+0];
            verts[i*3+1] = sh->verts[shvert*3+1];
            verts[i*3+2] = sh->verts[shvert*3+2];
        }
    }
}

//------------------------------------------------------------------------------
template <class T> void
createTopology( shape const * sh, OpenSubdiv::HbrMesh<T> * mesh, Scheme scheme) {

    const int * fv=&(sh->faceverts[0]);
    for(int f=0, ptxidx=0;f<sh->getNfaces(); f++ ) {

        int nv = sh->nvertsPerFace[f];

        if ((scheme==kLoop) and (nv!=3)) {
            printf("Trying to create a Loop surbd with non-triangle face\n");
            exit(1);
        }

        for(int j=0;j<nv;j++) {
            OpenSubdiv::HbrVertex<T> * origin      = mesh->GetVertex( fv[j] );
            OpenSubdiv::HbrVertex<T> * destination = mesh->GetVertex( fv[ (j+1)%nv] );
            OpenSubdiv::HbrHalfedge<T> * opposite  = destination->GetEdge(origin);

            if(origin==NULL || destination==NULL) {
                printf(" An edge was specified that connected a nonexistent vertex\n");
                exit(1);
            }

            if(origin == destination) {
                printf(" An edge was specified that connected a vertex to itself\n");
                exit(1);
            }

            if(opposite && opposite->GetOpposite() ) {
                printf(" A non-manifold edge incident to more than 2 faces was found\n");
                exit(1);
            }

            if(origin->GetEdge(destination)) {
                printf(" An edge connecting two vertices was specified more than once."
                       " It's likely that an incident face was flipped\n");
                exit(1);
            }
        }

        OpenSubdiv::HbrFace<T> * face = mesh->NewFace(nv, (int *)fv, 0);

        face->SetPtexIndex(ptxidx);

        if ( (scheme==kCatmark or scheme==kBilinear) and nv != 4 )
            ptxidx+=nv;
        else
            ptxidx++;

        fv+=nv;
    }

    mesh->SetInterpolateBoundaryMethod( OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly );

    applyTags<T>( mesh, sh );

    mesh->Finish();
}

//------------------------------------------------------------------------------
template <class T> OpenSubdiv::HbrMesh<T> *
simpleHbr(char const * shapestr, Scheme scheme, std::vector<float> * verts=0) {

    shape * sh = shape::parseShape( shapestr );

    OpenSubdiv::HbrMesh<T> * mesh = createMesh<T>(scheme);

    createVertices<T>(sh, mesh, verts);

    createTopology<T>(sh, mesh, scheme);

    if(verts)
        copyVertexPositions<T>(sh,mesh,*verts);

    delete sh;

    return mesh;
}

//------------------------------------------------------------------------------
template <class T> OpenSubdiv::HbrMesh<T> *
simpleHbr(char const * shapestr, Scheme scheme, std::vector<float> & verts) {

    shape * sh = shape::parseShape( shapestr );

    OpenSubdiv::HbrMesh<T> * mesh = createMesh<T>(scheme);

    createVertices<T>(sh, mesh, verts);

    createTopology<T>(sh, mesh, scheme);

    copyVertexPositions<T>(sh,mesh,verts);

    delete sh;

    return mesh;
}

#endif /* SHAPE_UTILS_H */
