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


#include "shape_utils.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <fstream>
#include <sstream>

//------------------------------------------------------------------------------
static char const * sgets( char * s, int size, char ** stream ) {
    for (int i=0; i<size; ++i) {
        if ( (*stream)[i]=='\n' || (*stream)[i]=='\0') {

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
Shape::~Shape() {
    for (int i=0; i<(int)tags.size(); ++i)
        delete tags[i];

    for (int i=0; i<(int)mtls.size(); ++i)
        delete mtls[i];
}

//------------------------------------------------------------------------------
Shape * Shape::parseObj(char const * shapestr, Scheme shapescheme, bool isLeftHanded,
                        bool parsemtl) {

    Shape * s = new Shape;

    s->scheme = shapescheme;
    s->isLeftHanded = isLeftHanded;

    char * str=const_cast<char *>(shapestr), line[256], buf[256], usemtl=-1;
    bool done = false;
    while (! done) {
        done = sgets(line, sizeof(line), &str)==0;
        if (line[0]) {
          char* end = &line[strlen(line)-1];
          if (*end == '\n') *end = '\0'; // strip trailing nl
        }
        float x, y, z, u, v;
        switch (line[0]) {
            case 'v': switch (line[1]) {
                          case ' ': if (sscanf(line, "v %f %f %f", &x, &y, &z) == 3) {
                                         s->verts.push_back(x);
                                         s->verts.push_back(y);
                                         s->verts.push_back(z);
                                    } break;
                          case 't': if (sscanf(line, "vt %f %f", &u, &v) == 2) {
                                        s->uvs.push_back(u);
                                        s->uvs.push_back(v);
                                    } break;
                          case 'n' : if (sscanf(line, "vn %f %f %f", &x, &y, &z) == 3) {
                                        s->normals.push_back(x);
                                        s->normals.push_back(y);
                                        s->normals.push_back(z);
                                     } break; // skip normals for now
                      } break;
            case 'f': if (line[1] == ' ') {
                          int vi, ti, ni;
                          const char* cp = &line[2];
                          while (*cp == ' ') cp++;
                          int nverts = 0, nitems=0;
                          while( (nitems=sscanf(cp, "%d/%d/%d", &vi, &ti, &ni))>0) {
                              nverts++;
                              s->faceverts.push_back(vi-1);
                              if(nitems > 1) s->faceuvs.push_back(ti-1);
                              if(nitems > 2) s->facenormals.push_back(ni-1);
                              while (*cp && *cp != ' ') cp++;
                              while (*cp == ' ') cp++;
                          }
                          s->nvertsPerFace.push_back(nverts);
                          if (! s->mtls.empty()) {
                              s->mtlbind.push_back(usemtl);
                          }
                      } break;
            case 't' : if (line[1] == ' ') {
                           Shape::tag * t = tag::parseTag( line );
                           if (t)
                               s->tags.push_back(t);
                       } break;
            case 'u' : if (parsemtl && sscanf(line, "usemtl %s", buf)==1) {
                           usemtl = s->FindMaterial(buf);
                       } break;
            case 'm' : if (parsemtl && sscanf(line, "mtllib %s", buf)==1) {
                           std::ifstream ifs(buf);
                           if (ifs) {
                               std::stringstream ss;
                               ss << ifs.rdbuf();
                               ifs.close();
                               std::string tmpStr = ss.str();
                               s->parseMtllib(tmpStr.c_str());
                               s->mtllib = buf;
                           }
                       } break;
        }
    }
    return s;
}

//------------------------------------------------------------------------------
Shape * Shape::parseObj(ShapeDesc const & shapeDesc, bool parsemtl) {
    return parseObj(shapeDesc.data.c_str(), shapeDesc.scheme, shapeDesc.isLeftHanded,
                    parsemtl);
}

//------------------------------------------------------------------------------
Shape::tag * Shape::tag::parseTag(char const * line) {
    tag * t = 0;

    const char* cp = &line[2];

    char tname[50];
    while (*cp == ' ') cp++;
    if (sscanf(cp, "%s", tname )!=1) return t;
    while (*cp && *cp != ' ') cp++;

    int nints=0, nfloats=0, nstrings=0;
    while (*cp == ' ') cp++;
    if (sscanf(cp, "%d/%d/%d", &nints, &nfloats, &nstrings)!=3) return t;
    while (*cp && *cp != ' ') cp++;

    std::vector<int> tintargs;
    for (int i=0; i<nints; ++i) {
        int val;
        while (*cp == ' ') cp++;
        if (sscanf(cp, "%d", &val)!=1) return t;
        tintargs.push_back(val);
        while (*cp && *cp != ' ') cp++;
    }

    std::vector<float> tfloatargs;
    for (int i=0; i<nfloats; ++i) {
        float val;
        while (*cp == ' ') cp++;
        if (sscanf(cp, "%f", &val)!=1) return t;
        tfloatargs.push_back(val);
        while (*cp && *cp != ' ') cp++;
    }

    std::vector<std::string> tstringargs;
    for (int i=0; i<nstrings; ++i) {
        char val[512];
        while (*cp == ' ') cp++;
        if (sscanf(cp, "%s", val)!=1) return t;
        tstringargs.push_back(std::string(val));
        while (*cp && *cp != ' ') cp++;
    }

    t = new Shape::tag;
    t->name = tname;
    t->intargs = tintargs;
    t->floatargs = tfloatargs;
    t->stringargs = tstringargs;

    return t;
}

//------------------------------------------------------------------------------
Shape::material::material() {
    memset(ka, 0, sizeof(float)*3);
    memset(kd, 0, sizeof(float)*3);
    memset(ks, 0, sizeof(float)*3);
    memset(tf, 0, sizeof(float)*3);
    ns = ni = d = 0.0f;
    illum=0;
}

//------------------------------------------------------------------------------
void Shape::parseMtllib(char const * mtlstr) {

    char * str=const_cast<char *>(mtlstr), line[256];

    material * mtl=0;

    bool done = false;
    float r, g, b, a;
    while (! done) {
        done = sgets(line, sizeof(line), &str)==0;
        char* end = &line[strlen(line)-1];
        if (*end == '\n') *end = '\0'; // strip trailing nl
        switch (line[0]) {
            case 'n': char name[256];
                      if (sscanf(line, "newmtl %s", name) == 1) {
                          mtl = new material;
                          mtl->name = name;
                          mtls.push_back(mtl);
                      } break;
            case 'K': if (sscanf(line+2, " %f %f %f", &r, &g, &b) == 3) {
                          switch (line[1]) {
                              case 'a': mtl->ka[0]=r; mtl->ka[1]=g; mtl->ka[2]=b; break;
                              case 'd': mtl->kd[0]=r; mtl->kd[1]=g; mtl->kd[2]=b; break;
                              case 's': mtl->ks[0]=r; mtl->ks[1]=g; mtl->ks[2]=b; break;
                          }
                      } break;
            case 'N': if (sscanf(line+2, " %f", &a) == 1) {
                          switch (line[1]) {
                              case 's' : mtl->ns = a; break;
                              case 'i' : mtl->ni = a; break;
                          }
                      } break;
            case 'd': if (sscanf(line, "d %f", &a) == 1) {
                          mtl->d = a;
                      } break;
            case 'T': if (line[1]=='f') {
                         if (sscanf(line, "Tf %f %f %f", &r, &g, &b) == 3) {
                             mtl->tf[0]=r; mtl->tf[1]=g; mtl->tf[2]=b;
                         } break;
                      } break;
            case 'i': int illum;
                      if (sscanf(line, "illum %d", &illum) == 1) {
                          mtl->illum = illum;
                      } break;
            case 's': if (sscanf(line, "sharpness %f", &a) == 1) {
                          mtl->sharpness = a;
                      } break;
        }
    }
}

//------------------------------------------------------------------------------
std::string Shape::tag::genTag() const {
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
std::string Shape::genShape(char const * name) const {
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
std::string Shape::genObj() const {
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
std::string Shape::genRIB() const {
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

    rib << "["<<names.str()<<"] " << "["<<nargs.str()<<"] " << "["<<intargs.str()<<"] " << "["<<floatargs.str()<<"] " << "["<<strargs.str()<<"] ";

    rib << "\"P\" [";
    std::copy(verts.begin(), verts.end(), std::ostream_iterator<float>(rib));
    rib << "] ";

    return rib.str();
}
