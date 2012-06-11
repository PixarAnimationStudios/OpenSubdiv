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

#include <stdio.h>
#include <string.h>

#include <list>
#include <string>
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

struct shape {

    struct tag {
    
        static tag * parseTag( char const * stream );
    
        std::string              name;            
        std::vector<int>         intargs;
        std::vector<float>       floatargs;
        std::vector<std::string> stringargs;
    };

    static shape * parseShape(char const * shapestr, int axis=1);
  
    ~shape();
  
    int getNverts() const { return (int)verts.size()/3; }

    int getNfaces() const { return (int)nvertsPerFace.size(); }

    std::vector<float>  verts;
    std::vector<float>  uvs;
    std::vector<int>    nvertsPerFace;  
    std::vector<int>    faceverts;	      
    std::vector<int>    faceuvs;
    std::vector<tag *>  tags;
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

    char name[15];
    while (*cp == ' ') cp++;
    if (sscanf(cp, "%s", &name )!=1) return t;
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
	char * val;
	while (*cp == ' ') cp++;
	if (sscanf(cp, "%s", &val)!=1) return t;
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
                      {       case ' ': if(sscanf(line, "v %f %f %f", &x, &y, &z) == 3) 
                                             s->verts.push_back(x); 
                                        switch( axis ) {  
                                            case 0 : s->verts.push_back(-z); 
                                                     s->verts.push_back(y); break;
                                            case 1 : s->verts.push_back(y); 
                                                    s->verts.push_back(z); break;
                                        } break; 
                              case 't': if(sscanf(line, "vt %f %f", &u, &v) == 2) { 
                                            s->uvs.push_back(u); 
                                            s->uvs.push_back(v); 
                                        } break;
                              case 'n' : break; // skip normals for now
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
void applyTags( HbrMesh<T> * mesh, shape const * sh ) {

    for (int i=0; i<(int)sh->tags.size(); ++i) {
        shape::tag * t = sh->tags[i];

        if (t->name=="crease") {
            for (int j=0; j<(int)t->intargs.size()-1; ++j) {
                HbrVertex<T> * v = mesh->GetVertex( t->intargs[j] ),
                             * w = mesh->GetVertex( t->intargs[j+1] );
                HbrHalfedge<T> * e = 0;
                if( v && w ) { 
                    if( !(e = v->GetEdge(w) ) )
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
                HbrVertex<T> * v = mesh->GetVertex( t->intargs[j] );
                if(v) {
                    int nfloat = (int) t->floatargs.size();
                    v->SetSharpness( std::max(0.0f, ((nfloat > 1) ? t->floatargs[j] : t->floatargs[0])) );
                } else
                   printf("cannot find vertex for corner tag (%d)\n", t->intargs[j] );
            }
        } else if (t->name=="hole") {
            for (int j=0; j<(int)t->intargs.size(); ++j) {
                HbrFace<T> * f = mesh->GetFace( t->intargs[j] );
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
	    switch( t->intargs[0] )
	    { case 0 : mesh->SetInterpolateBoundaryMethod(HbrMesh<T>::k_InterpolateBoundaryNone); break;
              case 1 : mesh->SetInterpolateBoundaryMethod(HbrMesh<T>::k_InterpolateBoundaryEdgeAndCorner); break;
	      case 2 : mesh->SetInterpolateBoundaryMethod(HbrMesh<T>::k_InterpolateBoundaryEdgeOnly); break;
	      default: printf("unknown interpolation boundary : %d\n", t->intargs[0] ); break;
	    }
        } else if (t->name=="facevaryingpropagatecorners") {
	    if ((int)t->intargs.size()==1)
	        mesh->SetFVarPropagateCorners( t->intargs[0] != 0 );
	    else
	        printf( "expecting single int argument for \"facevaryingpropagatecorners\"\n" );
        } else if (t->name=="creasemethod") {
        
	    HbrCatmarkSubdivision<T> * scheme = 
	        dynamic_cast<HbrCatmarkSubdivision<T> *>( mesh->GetSubdivision() );
	
	    if (not scheme) {
	        printf("the \"creasemethod\" tag can only be applied to Catmark meshes\n");
		continue;
	    }
	
	    if ((int)t->stringargs.size()==0) {
	        printf("the \"creasemethod\" tag expects a string argument\n");
		continue;
	    }
	
            if( t->stringargs[0]=="normal" )
                scheme->SetTriangleSubdivisionMethod(HbrCatmarkSubdivision<T>::k_Old);
            else if( t->stringargs[0]=="chaikin" )
                scheme->SetTriangleSubdivisionMethod(HbrCatmarkSubdivision<T>::k_New);
	    else
	        printf("the \"creasemethod\" tag only accepts \"normal\" or \"chaikin\" as value (%s)\n", t->stringargs[0].c_str());
	    
        } else if (t->name=="vertexedit" or t->name=="edgeedit") {
            printf("hierarchical edits not supported (yet)\n");
        } else {
            printf("Unknown tag : \"%s\" - skipping\n", t->name.c_str());
        }        
    }
}


enum Scheme {
  kBilinear,
  kCatmark,
  kLoop
};

//------------------------------------------------------------------------------
template <class T> HbrMesh<T> *
createMesh( Scheme scheme=kCatmark) {

  HbrMesh<T> * mesh = 0;
  
  static HbrBilinearSubdivision<T> _bilinear;
  static HbrLoopSubdivision<T>     _loop;
  static HbrCatmarkSubdivision<T>  _catmark;
  
  switch (scheme) {
    case kBilinear : mesh = new HbrMesh<T>( &_bilinear ); break;
    case kLoop : mesh = new HbrMesh<T>(  &_loop ); break;
    case kCatmark : mesh = new HbrMesh<T>(  &_catmark ); break;
  }
  
  return mesh;
}

//------------------------------------------------------------------------------
template <class T> void
createVertices( shape const * sh, HbrMesh<T> * mesh ) {
  
  T v;
  for(int i=0;i<sh->getNverts(); i++ ) {
    v.SetPosition( sh->verts[i*3], sh->verts[i*3+1], sh->verts[i*3+2] );
    mesh->NewVertex( i, v );
  }
}

//------------------------------------------------------------------------------
template <class T> void
createVertices( shape const * sh, HbrMesh<T> * mesh, std::vector<float> & verts ) {
  
    int nverts = sh->getNverts();
    verts.resize(nverts*3);

    T v;
    for(int i=0;i<nverts; i++ ) {
        mesh->NewVertex( i, v );
        
        verts[i*3  ]=sh->verts[i*3  ];
        verts[i*3+1]=sh->verts[i*3+1];
        verts[i*3+2]=sh->verts[i*3+2];
    }
}

//------------------------------------------------------------------------------
template <class T> void
createTopology( shape const * sh, HbrMesh<T> * mesh, Scheme scheme) {

      const int * fv=&(sh->faceverts[0]);
      for(int f=0, ptxidx=0;f<sh->getNfaces(); f++ ) {
      
          int nv = sh->nvertsPerFace[f];

          if ((scheme==kLoop) and (nv!=3)) {
              printf("Trying to create a Loop surbd with non-triangle face\n"); 
              exit(1); 
          }

          for(int j=0;j<nv;j++) { 
              HbrVertex<T> * origin      = mesh->GetVertex( fv[j] );                                                           
              HbrVertex<T> * destination = mesh->GetVertex( fv[ (j+1)%nv] );
              HbrHalfedge<T> * opposite  = destination->GetEdge(origin);

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

          HbrFace<T> * face = mesh->NewFace(nv, (int *)fv, 0);

          face->SetPtexIndex(ptxidx);

          if ( (scheme==kCatmark or scheme==kBilinear) and nv != 4 )
              ptxidx+=nv;
          else
              ptxidx++;

          fv+=nv;
      }

      applyTags<T>( mesh, sh );

      mesh->Finish();
}

//------------------------------------------------------------------------------
template <class T> HbrMesh<T> *
simpleHbr( char const * shapestr, Scheme scheme=kCatmark) {

  shape * sh = shape::parseShape( shapestr );

  HbrMesh<T> * mesh = createMesh<T>(scheme);
  
  createVertices<T>(sh, mesh);
  
  createTopology<T>(sh, mesh, scheme);
  
  delete sh;
  
  return mesh;
}

//------------------------------------------------------------------------------
template <class T> HbrMesh<T> *
simpleHbr( char const * shapestr, std::vector<float> & verts, Scheme scheme=kCatmark) {

  shape * sh = shape::parseShape( shapestr );

  HbrMesh<T> * mesh = createMesh<T>(scheme);
  
  createVertices<T>(sh, mesh, verts);
  
  createTopology<T>(sh, mesh, scheme);
  
  delete sh;
  
  return mesh;
}

#endif /* SHAPE_UTILS_H */
