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

#if defined(__APPLE__)
    #include <GLUT/glut.h>
#else
    #include <GL/glew.h>
    #include <GL/glut.h>
#endif

#include <osd/vertex.h>
#include <osd/mesh.h>
#include <osd/cpuDispatcher.h>
#include <osd/glslDispatcher.h>
#include <osd/pTexture.h>

#include "../common/stopwatch.h"

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clDispatcher.h>
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaDispatcher.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"
#endif

#include "Ptexture.h"
#include "PtexUtils.h"

static const char *shaderSource =
#include "shader.inc"
;

#include <vector>

//------------------------------------------------------------------------------
int   g_frame = 0,
      g_repeatCount = 0;

// GLUT GUI variables
int   g_wire = 0,
      g_drawNormals = 0,
      g_mbutton[3] = {0, 0, 0},
      g_level = 2,
      g_kernel = OpenSubdiv::OsdKernelDispatcher::kCPU,
      g_scheme = 0,
      g_gutterWidth = 1,
      g_ptexDebug = 0,
      g_gutterDebug = 0;
float g_moveScale = 1.0f;

// ptex switch
int   g_color = 1,
      g_occlusion = 0,
      g_displacement = 0;

// camera
float g_rotate[2] = {0, 0},
      g_prev_x = 0,
      g_prev_y = 0,
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

// viewport
int   g_width,
      g_height;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
Stopwatch g_fpsTimer;

// geometry
std::vector<float> g_positions,
                   g_normals;
int g_numIndices = 0;

GLuint g_indexBuffer;
GLuint g_program = 0;
GLuint g_debugProgram = 0;

OpenSubdiv::OsdMesh * g_osdmesh = 0;
OpenSubdiv::OsdVertexBuffer * g_vertexBuffer = 0;
OpenSubdiv::OsdPTexture * g_osdPTexImage = 0;
OpenSubdiv::OsdPTexture * g_osdPTexDisplacement = 0;
OpenSubdiv::OsdPTexture * g_osdPTexOcclusion = 0;
const char * g_ptexColorFile = 0;
const char * g_ptexDisplacementFile = 0;
const char * g_ptexOcclusionFile = 0;

//------------------------------------------------------------------------------
inline void
cross(float *n, const float *p0, const float *p1, const float *p2) {

    float a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
    float b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
    n[0] = a[1]*b[2]-a[2]*b[1];
    n[1] = a[2]*b[0]-a[0]*b[2];
    n[2] = a[0]*b[1]-a[1]*b[0];

    float rn = 1.0f/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    n[0] *= rn;
    n[1] *= rn;
    n[2] *= rn;
}

//------------------------------------------------------------------------------
inline void
normalize(float * p) {

    float dist = sqrtf( p[0]*p[0] + p[1]*p[1]  + p[2]*p[2] );
    p[0]/=dist;
    p[1]/=dist;
    p[2]/=dist;
}

//------------------------------------------------------------------------------
inline void
multMatrix(float *d, const float *a, const float *b) {

    for (int i=0; i<4; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            d[i*4 + j] =
                a[i*4 + 0] * b[0*4 + j] +
                a[i*4 + 1] * b[1*4 + j] +
                a[i*4 + 2] * b[2*4 + j] +
                a[i*4 + 3] * b[3*4 + j];
        }
    }
}

//------------------------------------------------------------------------------
static void
calcNormals(OpenSubdiv::OsdHbrMesh * mesh, std::vector<float> const & pos, std::vector<float> & result ) {

    // calc normal vectors
    int nverts = (int)pos.size()/3;

    int nfaces = mesh->GetNumCoarseFaces();
    for (int i = 0; i < nfaces; ++i) {

        OpenSubdiv::OsdHbrFace * f = mesh->GetFace(i);

        float const * p0 = &pos[f->GetVertex(0)->GetID()*3],
                    * p1 = &pos[f->GetVertex(1)->GetID()*3],
                    * p2 = &pos[f->GetVertex(2)->GetID()*3];

        float n[3];
        cross( n, p0, p1, p2 );

        for (int j = 0; j < f->GetNumVertices(); j++) {
            int idx = f->GetVertex(j)->GetID() * 3;
            result[idx  ] += n[0];
            result[idx+1] += n[1];
            result[idx+2] += n[2];
        }
    }
    for (int i = 0; i < nverts; ++i)
        normalize( &result[i*3] );
}

//------------------------------------------------------------------------------
void
updateGeom() {

    int nverts = (int)g_positions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*6);

    const float *p = &g_positions[0];
    const float *n = &g_normals[0];

    for (int i = 0; i < nverts; ++i) {
        float move = g_size*0.005f*cosf(p[0]*100/g_size+g_frame*0.01f);
        vertex.push_back(p[0]);
        vertex.push_back(p[1]+g_moveScale*move);
        vertex.push_back(p[2]);
        vertex.push_back(n[0]);
        vertex.push_back(n[1]);
        vertex.push_back(n[2]);
        p += 3;
        n += 3;
    }

    if (!g_vertexBuffer)
        g_vertexBuffer = g_osdmesh->InitializeVertexBuffer(6);
    g_vertexBuffer->UpdateData(&vertex[0], nverts);

    Stopwatch s;
    s.Start();

    g_osdmesh->Subdivide(g_vertexBuffer, NULL);

    s.Stop();
    g_cpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    g_osdmesh->Synchronize();
    s.Stop();
    g_gpuTime = float(s.GetElapsed() * 1000.0f);

    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer->GetGpuBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//-------------------------------------------------------------------------------
void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//-------------------------------------------------------------------------------
template <class T>
OpenSubdiv::HbrMesh<T> * createPTexGeo(PtexTexture * r)
{
  PtexMetaData* meta = r->getMetaData();
  if(meta->numKeys()<3) return NULL;

  const float* vp;
  const int *vi, *vc;
  int nvp, nvi, nvc;

  meta->getValue("PtexFaceVertCounts", vc, nvc);
  if (nvc==0)
      return NULL;

  meta->getValue("PtexVertPositions", vp, nvp);
  if (nvp==0)
      return NULL;

  meta->getValue("PtexFaceVertIndices", vi, nvi);
  if (nvi==0)
      return NULL;

  static OpenSubdiv::HbrCatmarkSubdivision<T>  _catmark;
  static OpenSubdiv::HbrBilinearSubdivision<T>  _bilinear;
  OpenSubdiv::HbrMesh<T> * mesh;
  if(g_scheme == 0)
      mesh = new OpenSubdiv::HbrMesh<T>(&_catmark);
  else
      mesh = new OpenSubdiv::HbrMesh<T>(&_bilinear);

  g_positions.clear();
  g_positions.reserve(nvp);

  // compute model bounding
  float min[3] = {vp[0], vp[1], vp[2]};
  float max[3] = {vp[0], vp[1], vp[2]};
  for (int i=0; i<nvp/3; ++i) {
      for(int j=0; j<3; ++j) {
          float v = vp[i*3+j];
          g_positions.push_back(v);
          min[j] = std::min(min[j], v);
          max[j] = std::max(max[j], v);
      }
      mesh->NewVertex(i, T());
  }
  for (int j=0; j<3; ++j) {
      g_center[j] = (min[j] + max[j]) * 0.5f;
      g_size += (max[j]-min[j])*(max[j]-min[j]);
  }
  g_size = sqrtf(g_size);

  const int *fv = vi;
  for (int i=0, ptxidx=0; i<nvc; ++i) {
      int nv = vc[i];
      OpenSubdiv::HbrFace<T> * face = mesh->NewFace(nv, (int *)fv, 0);

      face->SetPtexIndex(ptxidx);
      if(nv != 4)
          ptxidx+=nv;
      else
          ptxidx++;

      fv += nv;
  }
  mesh->SetInterpolateBoundaryMethod( OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly );
//  set creases here
//  applyTags<T>( mesh, sh );
  mesh->Finish();

  return mesh;
}


//------------------------------------------------------------------------------
void
reshape(int width, int height) {

    g_width = width;
    g_height = height;
}

#if _MSC_VER
    #define snprintf _snprintf
#endif

#define drawFmtString(x, y, fmt, ...)         \
    { char line[1024]; \
      snprintf(line, 1024, fmt, __VA_ARGS__); \
      const char *p = line; \
      glWindowPos2i(x, y); \
      while(*p) { glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p++); } }

#define drawString(x, y, str)                 \
    { const char *p = str; \
      glWindowPos2i(x, y); \
      while(*p) { glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p++); } }

//------------------------------------------------------------------------------
const char *getKernelName(int kernel) {

         if (kernel == OpenSubdiv::OsdKernelDispatcher::kCPU)
        return "CPU";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kOPENMP)
        return "OpenMP";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kCUDA)
        return "Cuda";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kGLSL)
        return "GLSL";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kCL)
        return "OpenCL";
    return "Unknown";
}

//------------------------------------------------------------------------------
static GLuint compileShader(GLenum shaderType, const char *section)
{
    const char *sources[2];
    char define[1024];
    sprintf(define,
            "#define %s\n"
            "#define USE_PTEX_COLOR %d\n"
            "#define USE_PTEX_OCCLUSION %d\n"
            "#define USE_PTEX_DISPLACEMENT %d\n",
            section, g_color, g_occlusion, g_displacement);

    sources[0] = define;
    sources[1] = shaderSource;

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 2, sources, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if( status == GL_FALSE ) {
        GLchar emsg[1024];
        glGetShaderInfoLog(shader, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error compiling GLSL shader (%s): %s\n", section, emsg );
        exit(0);
    }

    return shader;
}

void bindPTexture(OpenSubdiv::OsdPTexture *osdPTex, GLuint data, GLuint packing, GLuint pages, int samplerUnit)
{
    glProgramUniform1i(g_program, data, samplerUnit + 0);
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, osdPTex->GetTexelsTexture());

    glProgramUniform1i(g_program, packing, samplerUnit + 1);
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 1);
    glBindTexture(GL_TEXTURE_BUFFER, osdPTex->GetLayoutTextureBuffer());

    glProgramUniform1i(g_program, pages, samplerUnit + 2);
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 2);
    glBindTexture(GL_TEXTURE_BUFFER, osdPTex->GetPagesTextureBuffer());

    glActiveTexture(GL_TEXTURE0);
}

void linkDebugProgram() {

    if (g_debugProgram)
        glDeleteProgram(g_debugProgram);

    GLuint vertexShader = compileShader(GL_VERTEX_SHADER,
                                          "PTEX_DEBUG_VERTEX_SHADER");
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER,
                                          "PTEX_DEBUG_FRAGMENT_SHADER");

    g_debugProgram = glCreateProgram();
    glAttachShader(g_debugProgram, vertexShader);
    glAttachShader(g_debugProgram, fragmentShader);
    glLinkProgram(g_debugProgram);
    glDeleteShader(fragmentShader);

    GLint status;
    glGetProgramiv(g_debugProgram, GL_LINK_STATUS, &status );
    if( status == GL_FALSE ) {
        GLchar emsg[1024];
        glGetProgramInfoLog(g_debugProgram, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error linking GLSL program : %s\n", emsg );
        exit(0);
    }

    GLint texData = glGetUniformLocation(g_debugProgram, "ptexDebugData");
    glProgramUniform1i(g_debugProgram, texData, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, g_osdPTexImage->GetTexelsTexture());
}

void linkProgram() {

    if (g_program)
        glDeleteProgram(g_program);

    GLuint vertexShader         = compileShader(GL_VERTEX_SHADER,
                                                "VERTEX_SHADER");
    GLuint geometryShader       = compileShader(GL_GEOMETRY_SHADER,
                                                "GEOMETRY_SHADER");
    GLuint fragmentShader       = compileShader(GL_FRAGMENT_SHADER,
                                                "FRAGMENT_SHADER");

    g_program = glCreateProgram();
    glAttachShader(g_program, vertexShader);
    glAttachShader(g_program, geometryShader);
    glAttachShader(g_program, fragmentShader);

    glBindAttribLocation(g_program, 0, "position");
    glBindAttribLocation(g_program, 1, "normal");

    glLinkProgram(g_program);

    glDeleteShader(vertexShader);
    glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);

    GLint status;
    glGetProgramiv(g_program, GL_LINK_STATUS, &status );
    if( status == GL_FALSE ) {
        GLchar emsg[1024];
        glGetProgramInfoLog(g_program, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error linking GLSL program : %s\n", emsg );
        exit(0);
    }

    // bind ptexture
    GLint texIndices = glGetUniformLocation(g_program, "ptexIndices");
    GLint ptexLevel = glGetUniformLocation(g_program, "ptexLevel");

    glProgramUniform1i(g_program, ptexLevel, 1<<g_level);
    glProgramUniform1i(g_program, texIndices, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, g_osdmesh->GetPtexCoordinatesTextureBuffer(g_level));

    // color ptex
    GLint texData = glGetUniformLocation(g_program, "textureImage_Data");
    GLint texPacking = glGetUniformLocation(g_program, "textureImage_Packing");
    GLint texPages = glGetUniformLocation(g_program, "textureImage_Pages");
    bindPTexture(g_osdPTexImage, texData, texPacking, texPages, 1);

    // displacement ptex
    if (g_displacement) {
        texData = glGetUniformLocation(g_program, "textureDisplace_Data");
        texPacking = glGetUniformLocation(g_program, "textureDisplace_Packing");
        texPages = glGetUniformLocation(g_program, "textureDisplace_Pages");
        bindPTexture(g_osdPTexDisplacement, texData, texPacking, texPages, 4);
    }

    // occlusion ptex
    if (g_occlusion) {
        texData = glGetUniformLocation(g_program, "textureOcclusion_Data");
        texPacking = glGetUniformLocation(g_program, "textureOcclusion_Packing");
        texPages = glGetUniformLocation(g_program, "textureOcclusion_Pages");
        bindPTexture(g_osdPTexOcclusion, texData, texPacking, texPages, 7);
    }

}

//------------------------------------------------------------------------------
void
createOsdMesh(int level, int kernel) {

    Ptex::String ptexError;
    PtexTexture *ptexColor = PtexTexture::open(g_ptexColorFile, ptexError, true);

    // generate Hbr representation from ptex
    OpenSubdiv::OsdHbrMesh * hmesh = createPTexGeo<OpenSubdiv::OsdVertex>(ptexColor);
    if(hmesh == NULL) return;

    g_normals.resize(g_positions.size(),0.0f);
    calcNormals( hmesh, g_positions, g_normals );

    // generate Osd mesh from Hbr mesh
    if (g_osdmesh) delete g_osdmesh;
    g_osdmesh = new OpenSubdiv::OsdMesh();
    g_osdmesh->Create(hmesh, level, kernel);
    if (g_vertexBuffer) {
        delete g_vertexBuffer;
        g_vertexBuffer = NULL;
    }

    // Hbr mesh can be deleted
    delete hmesh;

    // update element array buffer
    const std::vector<int> &indices = g_osdmesh->GetFarMesh()->GetFaceVertices(level);

    // generate oOsdPTexture
    if (g_osdPTexDisplacement) delete g_osdPTexDisplacement;
    if (g_osdPTexOcclusion) delete g_osdPTexOcclusion;
    g_osdPTexDisplacement = NULL;
    g_osdPTexOcclusion = NULL;

    OpenSubdiv::OsdPTexture::SetGutterWidth(g_gutterWidth);
    OpenSubdiv::OsdPTexture::SetPageMargin(g_gutterWidth*8);
    OpenSubdiv::OsdPTexture::SetGutterDebug(g_gutterDebug);

    if (g_osdPTexImage) delete g_osdPTexImage;
    g_osdPTexImage = OpenSubdiv::OsdPTexture::Create(ptexColor, 0 /*targetmemory*/);
    ptexColor->release();

    if (g_ptexDisplacementFile) {
        PtexTexture *ptexDisplacement = PtexTexture::open(g_ptexDisplacementFile, ptexError, true);
        g_osdPTexDisplacement = OpenSubdiv::OsdPTexture::Create(ptexDisplacement, 0);
        ptexDisplacement->release();
    }
    if (g_ptexOcclusionFile) {
        PtexTexture *ptexOcclusion = PtexTexture::open(g_ptexOcclusionFile, ptexError, true);
        g_osdPTexOcclusion = OpenSubdiv::OsdPTexture::Create(ptexOcclusion, 0);
        ptexOcclusion->release();
    }

    // bind index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_indexBuffer);

    g_numIndices = (int)indices.size();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*g_numIndices, &(indices[0]), GL_STATIC_DRAW);

    updateGeom();

    linkProgram();
    linkDebugProgram();
}

//------------------------------------------------------------------------------
void
drawNormals() {

    float * data=0;
    int datasize = g_osdmesh->GetTotalVertices() * g_vertexBuffer->GetNumElements();

    data = new float[datasize];

    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer->GetGpuBuffer());
    glGetBufferSubData(GL_ARRAY_BUFFER,0,datasize*sizeof(float),data);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_LIGHTING);
    glColor3f(0.0f, 0.0f, 0.5f);
    glBegin(GL_LINES);

    int start = g_osdmesh->GetFarMesh()->GetSubdivision()->GetFirstVertexOffset(g_level) *
                g_vertexBuffer->GetNumElements();

    for (int i=start; i<datasize; i+=6) {
        glVertex3f( data[i  ],
                    data[i+1],
                    data[i+2] );

        float n[3] = { data[i+3], data[i+4], data[i+5] };
        normalize(n);

        glVertex3f( data[i  ]+n[0]*0.2f,
                    data[i+1]+n[1]*0.2f,
                    data[i+2]+n[2]*0.2f );
    }
    glEnd();

    delete [] data;
}

//------------------------------------------------------------------------------
void
drawPtexLayout(int page) {

    glUseProgram(g_debugProgram);

    GLint width, height, depth;
    glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT, &height);
    glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH, &depth);

    GLint pageUniform = glGetUniformLocation(g_debugProgram, "ptexDebugPage");
    glProgramUniform1i(g_debugProgram, pageUniform, page);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBegin(GL_TRIANGLE_STRIP);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 1, 0);
    glVertex3f(1, 0, 0);
    glVertex3f(1, 1, 0);
    glEnd();

    glUseProgram(0);

    drawFmtString(g_width/2, g_height - 10, "Size = %dx%d, Page = %d/%d", width, height, page, depth);
}

//------------------------------------------------------------------------------
void
display() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, g_width, g_height);
    double aspect = g_width/(double)g_height;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, aspect, g_size*0.001f, g_size+g_dolly);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-g_pan[0], -g_pan[1], -g_dolly);
    glRotatef(g_rotate[1], 1, 0, 0);
    glRotatef(g_rotate[0], 0, 1, 0);
    glTranslatef(-g_center[0], -g_center[1], -g_center[2]);

    glUseProgram(g_program);

    {
        // shader uniform setting
        GLint position = glGetUniformLocation(g_program, "lightSource[0].position");
        GLint ambient = glGetUniformLocation(g_program, "lightSource[0].ambient");
        GLint diffuse = glGetUniformLocation(g_program, "lightSource[0].diffuse");
        GLint specular = glGetUniformLocation(g_program, "lightSource[0].specular");

        glProgramUniform4f(g_program, position, 0, 0.2f, 1, 0);
        glProgramUniform4f(g_program, ambient, 0.4f, 0.4f, 0.4f, 1.0f);
        glProgramUniform4f(g_program, diffuse, 0.3f, 0.3f, 0.3f, 1.0f);
        glProgramUniform4f(g_program, specular, 0.2f, 0.2f, 0.2f, 1.0f);

        GLint otcMatrix = glGetUniformLocation(g_program, "objectToClipMatrix");
        GLint oteMatrix = glGetUniformLocation(g_program, "objectToEyeMatrix");
        GLfloat modelView[16], proj[16], mvp[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
        glGetFloatv(GL_PROJECTION_MATRIX, proj);
        multMatrix(mvp, modelView, proj);
        glProgramUniformMatrix4fv(g_program, otcMatrix, 1, false, mvp);
        glProgramUniformMatrix4fv(g_program, oteMatrix, 1, false, modelView);
    }

    GLuint bVertex = g_vertexBuffer->GetGpuBuffer();
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, bVertex);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_indexBuffer);

    glPolygonMode(GL_FRONT_AND_BACK, g_wire==0 ? GL_LINE : GL_FILL);

//    glPatchParameteri(GL_PATCH_VERTICES, 4);
//    glDrawElements(GL_PATCHES, g_numIndices, GL_UNSIGNED_INT, 0);
    glDrawElements(GL_LINES_ADJACENCY, g_numIndices, GL_UNSIGNED_INT, 0);

    glUseProgram(0);

    if (g_drawNormals)
        drawNormals();

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    if (g_ptexDebug)
        drawPtexLayout(g_ptexDebug-1);

    glColor3f(1, 1, 1);
    drawFmtString(10, 10, "LEVEL = %d", g_level);
    drawFmtString(10, 30, "# of Vertices = %d", g_osdmesh->GetFarMesh()->GetNumVertices());
    drawFmtString(10, 50, "KERNEL = %s", getKernelName(g_kernel));
    drawFmtString(10, 70, "CPU TIME = %.3f ms", g_cpuTime);
    drawFmtString(10, 90, "GPU TIME = %.3f ms", g_gpuTime);
    g_fpsTimer.Stop();
    drawFmtString(10, 110, "FPS = %3.1f", 1.0/g_fpsTimer.GetElapsed());
    g_fpsTimer.Start();
    drawFmtString(10, 130, "SUBDIVISION = %s", g_scheme==0 ? "CATMARK" : "BILINEAR");

    drawString(10, g_height-10, "a:   ambient occlusion on/off");
    drawString(10, g_height-30, "c:   color on/off");
    drawString(10, g_height-50, "d:   displacement on/off");
    drawString(10, g_height-70, "e:   show normal vector");
    drawString(10, g_height-90, "f:   fit frame");
    drawString(10, g_height-110, "w:   toggle wireframe");
    drawString(10, g_height-130, "m:   toggle vertex moving");
    drawString(10, g_height-150, "s:   bilinear / catmark");
    drawString(10, g_height-170, "1-7: subdivision level");


    glFinish();
    glutSwapBuffers();
}

//------------------------------------------------------------------------------
void mouse(int button, int state, int x, int y) {

    g_prev_x = float(x);
    g_prev_y = float(y);
    g_mbutton[button] = !state;
}

//------------------------------------------------------------------------------
void motion(int x, int y) {

    if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
        // orbit
        g_rotate[0] += x - g_prev_x;
        g_rotate[1] += y - g_prev_y;
    } else if (!g_mbutton[0] && g_mbutton[1] && !g_mbutton[2]) {
        // pan
        g_pan[0] -= g_dolly*(x - g_prev_x)/g_width;
        g_pan[1] += g_dolly*(y - g_prev_y)/g_height;
    } else if (g_mbutton[0] && g_mbutton[1] && !g_mbutton[2]) {
        // dolly
        g_dolly -= g_dolly*0.01f*(x - g_prev_x);
        if(g_dolly <= 0.01) g_dolly = 0.01f;
    }

    g_prev_x = float(x);
    g_prev_y = float(y);
}

//------------------------------------------------------------------------------
void quit() {

    if(g_osdmesh)
        delete g_osdmesh;

    if (g_vertexBuffer)
        delete g_vertexBuffer;

#ifdef OPENSUBDIV_HAS_CUDA
    cudaDeviceReset();
#endif
    exit(0);
}

//------------------------------------------------------------------------------
void kernelMenu(int k) {

    g_kernel = k;
    createOsdMesh(g_level, g_kernel);
}

//------------------------------------------------------------------------------
void
levelMenu(int l) {

    g_level = l;
    createOsdMesh(g_level, g_kernel);
}

//------------------------------------------------------------------------------
void
schemeMenu(int s) {

    g_scheme = s;
    createOsdMesh(g_level, g_kernel);
}

//------------------------------------------------------------------------------
void
menu(int m) {

    // top menu
}

//------------------------------------------------------------------------------
void
keyboard(unsigned char key, int x, int y) {

    switch (key) {
        case 'q': quit();
        case 'w': g_wire = (g_wire+1)%2; break;
        case 'e': g_drawNormals = (g_drawNormals+1)%2; break;
        case 'f': fitFrame(); break;
        case 'a': if (g_osdPTexOcclusion) g_occlusion = !g_occlusion; linkProgram(); break;
        case 'd': if (g_osdPTexDisplacement) g_displacement = !g_displacement; linkProgram();break;
        case 'c': g_color = !g_color; linkProgram(); break;
        case 's': schemeMenu(!g_scheme); break;
        case 'm': g_moveScale = 1.0f - g_moveScale; break;
        case 'p': g_ptexDebug++; break;
        case 'o': g_ptexDebug = std::max(0, g_ptexDebug-1); break;
        case 'g': g_gutterWidth = (g_gutterWidth+1)%8; createOsdMesh(g_level, g_kernel); break;
        case 'h': g_gutterDebug = !g_gutterDebug; createOsdMesh(g_level, g_kernel); break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7': levelMenu(key-'0'); break;
    }
}

//------------------------------------------------------------------------------
void
idle() {

    g_frame++;
    updateGeom();
    glutPostRedisplay();

    if(g_repeatCount != 0 && g_frame >= g_repeatCount)
        quit();
}

//------------------------------------------------------------------------------
void
initGL() {

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_LIGHT0);
    glColor3f(1, 1, 1);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    GLfloat color[4] = {1, 1, 1, 1};
    GLfloat position[4] = {5, 5, 10, 1};
    GLfloat ambient[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    GLfloat diffuse[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat shininess = 25.0;

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, &shininess);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA |GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(1024, 1024);
    glutCreateWindow("OpenSubdiv ptexViewer");

    int lmenu = glutCreateMenu(levelMenu);
    for(int i = 1; i < 8; ++i){
        char level[16];
        sprintf(level, "Level %d\n", i);
        glutAddMenuEntry(level, i);
    }
    int smenu = glutCreateMenu(schemeMenu);
    glutAddMenuEntry("Catmark", 0);
    glutAddMenuEntry("Bilinear", 1);

    // Register Osd compute kernels
    OpenSubdiv::OsdCpuKernelDispatcher::Register();
    OpenSubdiv::OsdGlslKernelDispatcher::Register();

#if OPENSUBDIV_HAS_OPENCL
    OpenSubdiv::OsdClKernelDispatcher::Register();
#endif

#if OPENSUBDIV_HAS_CUDA
    OpenSubdiv::OsdCudaKernelDispatcher::Register();

    // Note: This function randomly crashes with linux 5.0-dev driver.
    // cudaGetDeviceProperties overrun stack..?
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
#endif

    int kmenu = glutCreateMenu(kernelMenu);
    int nKernels = OpenSubdiv::OsdKernelDispatcher::kMAX;

    for(int i = 0; i < nKernels; ++i)
        if(OpenSubdiv::OsdKernelDispatcher::HasKernelType(
               OpenSubdiv::OsdKernelDispatcher::KernelType(i)))
            glutAddMenuEntry(getKernelName(i), i);

    glutCreateMenu(menu);
    glutAddSubMenu("Level", lmenu);
    glutAddSubMenu("Scheme", smenu);
    glutAddSubMenu("Kernel", kmenu);
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glewInit();
    initGL();

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d"))
            g_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c"))
            g_repeatCount = atoi(argv[++i]);
        else if (g_ptexColorFile == NULL)
            g_ptexColorFile = argv[i];
        else if (g_ptexDisplacementFile == NULL)
            g_ptexDisplacementFile = argv[i];
        else if (g_ptexOcclusionFile == NULL)
            g_ptexOcclusionFile = argv[i];

    }

    if (g_ptexColorFile == NULL) {
        printf("Usage: %s <color.ptx> [<displacement.ptx>] [<occlusion.ptx>] \n", argv[0]);
        return 1;
    }

    glGenBuffers(1, &g_indexBuffer);

    createOsdMesh(g_level, g_kernel);

    fitFrame();

    glutIdleFunc(idle);
    glutMainLoop();

    quit();
}

//------------------------------------------------------------------------------
