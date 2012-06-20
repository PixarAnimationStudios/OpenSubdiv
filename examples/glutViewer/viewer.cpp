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

#include <common/shape_utils.h>

#include "../common/stopwatch.h"

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clDispatcher.h>
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaDispatcher.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "cudaInit.h"
#endif

#include <vector>

//------------------------------------------------------------------------------
struct SimpleShape {
    std::string  name;
    Scheme       scheme;
    char const * data;
    
    SimpleShape( char const * idata, char const * iname, Scheme ischeme )
        : name(iname), scheme(ischeme), data(idata) { }
};

std::vector<SimpleShape> g_defaultShapes;

int g_currentShape = 0;


void 
initializeShapes( ) {

#include <shapes/bilinear_cube.h>
    g_defaultShapes.push_back(SimpleShape(bilinear_cube, "bilinear_cube", kBilinear));

#include <shapes/catmark_cube_corner0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner0, "catmark_cube_corner0", kCatmark));

#include <shapes/catmark_cube_corner1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner1, "catmark_cube_corner1", kCatmark));

#include <shapes/catmark_cube_corner2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner2, "catmark_cube_corner2", kCatmark));

#include <shapes/catmark_cube_corner3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner3, "catmark_cube_corner3", kCatmark));

#include <shapes/catmark_cube_corner4.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner4, "catmark_cube_corner4", kCatmark));

#include <shapes/catmark_cube_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_creases0, "catmark_cube_creases0", kCatmark));

#include <shapes/catmark_cube_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_creases1, "catmark_cube_creases1", kCatmark));

#include <shapes/catmark_cube.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube, "catmark_cube", kCatmark));

#include <shapes/catmark_dart_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(catmark_dart_edgecorner, "catmark_dart_edgecorner", kCatmark));

#include <shapes/catmark_dart_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(catmark_dart_edgeonly, "catmark_dart_edgeonly", kCatmark));

#include <shapes/catmark_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(catmark_edgecorner ,"catmark_edgecorner", kCatmark));

#include <shapes/catmark_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(catmark_edgeonly, "catmark_edgeonly", kCatmark));

#include <shapes/catmark_pyramid_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid_creases0, "catmark_pyramid_creases0", kCatmark));

#include <shapes/catmark_pyramid_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid_creases1, "catmark_pyramid_creases1", kCatmark));

#include <shapes/catmark_pyramid.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid, "catmark_pyramid", kCatmark));

#include <shapes/catmark_tent_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent_creases0, "catmark_tent_creases0", kCatmark));

#include <shapes/catmark_tent_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent_creases1, "catmark_tent_creases1", kCatmark));

#include <shapes/catmark_tent.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent, "catmark_tent", kCatmark));

#include <shapes/loop_cube_creases0.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube_creases0, "loop_cube_creases0", kLoop));

#include <shapes/loop_cube_creases1.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube_creases1, "loop_cube_creases1", kLoop));

#include <shapes/loop_cube.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube, "loop_cube", kLoop));

#include <shapes/loop_icosahedron.h>
    g_defaultShapes.push_back(SimpleShape(loop_icosahedron, "loop_icosahedron", kLoop));

#include <shapes/loop_saddle_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(loop_saddle_edgecorner, "loop_saddle_edgecorner", kLoop));

#include <shapes/loop_saddle_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(loop_saddle_edgeonly, "loop_saddle_edgeonly", kLoop));

#include <shapes/loop_triangle_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(loop_triangle_edgecorner, "loop_triangle_edgecorner", kLoop));

#include <shapes/loop_triangle_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(loop_triangle_edgeonly, "loop_triangle_edgeonly", kLoop));
}

//------------------------------------------------------------------------------
int   g_frame = 0,
      g_repeatCount = 0;

// GLUT GUI variables
int   g_wire = 0,
      g_mbutton;

float g_rx = 0, 
      g_ry = 0, 
      g_prev_x = 0, 
      g_prev_y = 0,
      g_dolly = 5;

int   g_width, 
      g_height;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;

// geometry
std::vector<float> g_positions,
                   g_normals;
                   
Scheme             g_scheme;                   

int g_numIndices = 0;
int g_level = 2;
int g_kernel = OpenSubdiv::OsdKernelDispatcher::kCPU;

GLuint g_indexBuffer;

OpenSubdiv::OsdMesh * g_osdmesh = 0;
OpenSubdiv::OsdVertexBuffer * g_vertexBuffer = 0;

//------------------------------------------------------------------------------                                        
inline void 
cross(float *n, const float *p0, const float *p1, const float *p2) {

    float a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
    float b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
    n[0] = a[1]*b[2]-a[2]*b[1];
    n[1] = a[2]*b[0]-a[0]*b[2];
    n[2] = a[0]*b[1]-a[1]*b[0];
    
    float rn = 1.0f/sqrtf(n[0]*n[0] + n[1] * n[1] + n[2] * n[2]);
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
            result[ idx  ] += n[0];
            result[ idx+1] += n[1];
            result[ idx+2] += n[2];
        }
    }
    for (int i = 0; i < nverts; ++i)
        normalize( &result[i*3] );
}

//------------------------------------------------------------------------------
void 
updateGeom()
{
    int nverts = (int)g_positions.size() / 3;
    
    std::vector<float> vertex;
    vertex.reserve(nverts*6);
    
    const float *p = &g_positions[0];
    const float *n = &g_normals[0];
       
    for (int i = 0; i < nverts; ++i) {
        float move = 0.05*cos(p[0]*20+g_frame*0.01);
        vertex.push_back(p[0]);
        vertex.push_back(p[1]+move);
        vertex.push_back(p[2]);
        vertex.push_back(n[0]);
        vertex.push_back(n[1]);
        vertex.push_back(n[2]);
        p += 3;
        n += 3;
    }

    if (!g_vertexBuffer) g_vertexBuffer = g_osdmesh->InitializeVertexBuffer(6);
    g_vertexBuffer->UpdateData(&vertex[0], nverts);

    Stopwatch s;
    s.Start();

    g_osdmesh->Subdivide(g_vertexBuffer, NULL);

    s.Stop();
    g_cpuTime = s.GetElapsed() * 1000.0f;
    s.Start();
    g_osdmesh->Synchronize();
    s.Stop();
    g_gpuTime = s.GetElapsed() * 1000.0f;

    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer->GetGpuBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//------------------------------------------------------------------------------
void
createOsdMesh( const char * shape, int level, int kernel, Scheme scheme=kCatmark ) { 

    // generate Hbr representation from "obj" description
    OpenSubdiv::OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape, scheme, g_positions);

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

    g_numIndices = indices.size();
    g_scheme = scheme;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*g_numIndices, &(indices[0]), GL_STATIC_DRAW);
       
    updateGeom();
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

#define drawString(x, y, fmt, ...)               \
    { char line[1024]; \
      snprintf(line, 1024, fmt, __VA_ARGS__); \
      char *p = line; \
      glWindowPos2f(x, y); \
      while(*p) { glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p++); } }

//------------------------------------------------------------------------------
const char *getKernelName(int kernel)
{
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
void 
display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, g_width, g_height);
    double aspect = g_width/(double)g_height;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, aspect, 0.001, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -g_dolly);
    glRotatef(g_ry, 1, 0, 0);
    glRotatef(g_rx, 0, 1, 0);

    GLuint bVertex = g_vertexBuffer->GetGpuBuffer();
#ifdef VARYING_NORMAL
    GLuint bVarying = g_varyingBuffer->GetGpuBuffer();
    glBindBuffer(GL_ARRAY_BUFFER, bVertex);
    glVertexPointer(3, GL_FLOAT, 12, ((float*)(0)));

    glBindBuffer(GL_ARRAY_BUFFER, bVarying);
    glNormalPointer(GL_FLOAT, 12, ((float*)(0)));
#else
    glBindBuffer(GL_ARRAY_BUFFER, bVertex);
    glVertexPointer(3, GL_FLOAT, 24, ((float*)(0)));
    glNormalPointer(GL_FLOAT, 24, ((float*)(12)));
#endif

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_indexBuffer);

    if(g_wire == 0){
        glColor3f(1.0f, 1.0f, 1.0f);
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDisable(GL_LIGHTING);
        glDrawElements(g_scheme==kLoop ? GL_TRIANGLES : GL_QUADS, g_numIndices, GL_UNSIGNED_INT, NULL);
    }else{
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_LIGHTING);
        glDrawElements(g_scheme==kLoop ? GL_TRIANGLES : GL_QUADS, g_numIndices, GL_UNSIGNED_INT, NULL);

        if(g_wire == 2){
            glColor3f(0.0f, 0.0f, 0.5f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDisable(GL_LIGHTING);
            glDrawElements(g_scheme==kLoop ? GL_TRIANGLES : GL_QUADS, g_numIndices, GL_UNSIGNED_INT, NULL);
        }
        glColor3f(1.0f, 1.0f, 1.0f);
    }


    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    drawString(10, 10, "LEVEL = %d", g_level);
    drawString(10, 30, "# of Vertices = %d", g_osdmesh->GetFarMesh()->GetNumVertices());
    drawString(10, 50, "KERNEL = %s", getKernelName(g_kernel));
    drawString(10, 70, "CPU TIME = %.3f ms", g_cpuTime);
    drawString(10, 90, "GPU TIME = %.3f ms", g_gpuTime);
    drawString(10, 110, "SUBDIVISION = %s", g_scheme==kBilinear ? "BILINEAR" : (g_scheme == kLoop ? "LOOP" : "CATMARK"));

    glFinish();
    glutSwapBuffers();
}

//------------------------------------------------------------------------------
void motion(int x, int y)
{
    if(g_mbutton == 0){
        g_rx += x - g_prev_x;
        g_ry += y - g_prev_y;
    }else if(g_mbutton == 1){
        g_dolly -= 0.01*(x - g_prev_x);
        if(g_dolly <= 0.01) g_dolly = 0.01;
    }

    g_prev_x = x;
    g_prev_y = y;
}

//------------------------------------------------------------------------------
void mouse(int button, int state, int x, int y)
{
    g_prev_x = x;
    g_prev_y = y;
    g_mbutton = button;
}

//------------------------------------------------------------------------------
void quit()
{
    if(g_osdmesh) 
        delete g_osdmesh;

#ifdef OPENSUBDIV_HAS_CUDA
    cudaDeviceReset();;
#endif
    exit(0);
}

//------------------------------------------------------------------------------
void kernelMenu(int k)
{
    g_kernel = k;
    createOsdMesh( g_defaultShapes[ g_currentShape ].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

//------------------------------------------------------------------------------
void 
modelMenu(int m)
{
    if (m < 0) 
        m = 0;
        
    if (m >= (int)g_defaultShapes.size()) 
        m = g_defaultShapes.size() - 1;

    g_currentShape = m;
       
    glutSetWindowTitle( g_defaultShapes[m].name.c_str() );

    createOsdMesh( g_defaultShapes[m].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

//------------------------------------------------------------------------------
void 
levelMenu(int l)
{
    g_level = l;

    createOsdMesh( g_defaultShapes[g_currentShape].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

//------------------------------------------------------------------------------
void 
menu(int m)
{
}

//------------------------------------------------------------------------------
void 
keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 'q': quit();
        case 'w': g_wire = (g_wire+1)%3; break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7': levelMenu(key-'0'); break;
        case 'n': modelMenu(++g_currentShape); break;
        case 'p': modelMenu(--g_currentShape); break;
    }
}

//------------------------------------------------------------------------------
void 
idle()
{
    g_frame++;
    updateGeom();
    glutPostRedisplay();

    if(g_repeatCount != 0 && g_frame >= g_repeatCount){
        quit();
    }
}

//------------------------------------------------------------------------------
void 
initGL()
{
    glClearColor(0, 0, 0, 1);
    glEnable(GL_LIGHT0);
    glColor3f(1, 1, 1);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    GLfloat color[4] = {1, 1, 1, 1};
    GLfloat position[4] = {0, 1, 0, 1};
    GLfloat ambient[4] = {0.2f, 0.2f, 0.2f, 1.0f};
    GLfloat diffuse[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    glutInit(&argc, argv);
    
    glutInitDisplayMode(GLUT_RGBA |GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(1024, 1024);
    glutCreateWindow("OpenSubdiv test");

    initializeShapes();

    int smenu = glutCreateMenu(modelMenu);
    for(int i = 0; i < (int)g_defaultShapes.size(); ++i){
        glutAddMenuEntry( g_defaultShapes[i].name.c_str(), i);
    }

    int lmenu = glutCreateMenu(levelMenu);
    for(int i = 1; i < 8; ++i){
        char level[16];
        sprintf(level, "Level %d\n", i);
        glutAddMenuEntry(level, i);
    }

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

    for(int i = 0; i < nKernels; ++i) {
        if(OpenSubdiv::OsdKernelDispatcher::HasKernelType(
               OpenSubdiv::OsdKernelDispatcher::KernelType(i)))
            glutAddMenuEntry(getKernelName(i), i);
    }

    glutCreateMenu(menu);
    glutAddSubMenu("Level", lmenu);
    glutAddSubMenu("Model", smenu);
    glutAddSubMenu("Kernel", kmenu);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
    
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glewInit();
    initGL();

    const char *filename = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d"))
            g_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c"))
            g_repeatCount = atoi(argv[++i]);
        else 
            filename = argv[i];
    }
    
    glGenBuffers(1, &g_indexBuffer);

    modelMenu(0);

    glutIdleFunc(idle);
    glutMainLoop();
    
    quit();
}

//------------------------------------------------------------------------------
