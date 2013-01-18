
#if defined(__APPLE__)
    #include <GLUT/glut.h>
#else
    #include <stdlib.h>
//    #include <GL/glew.h>
    #if defined(WIN32)
//        #include <GL/wglew.h>
    #endif
//    #include <GL/glut.h>
#endif

// XXXdyu-api
#include <version.h>
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class Mutex {
public:
    void Lock() {}
    void Unlock() {}
};

}
using namespace OPENSUBDIV_VERSION;
}

#include <osd/error.h>
#include <osd/vertex.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>

#include <osd/cpuDispatcher.h>
#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>

#include <osd/glMesh.h>

// XXXdyu-api
typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;


class MyPatch {

    float _cvs[16]

};
 



static void
buildMesh(float *vertexData, int numVertices,
          int *indexData, int numIndices,
          int *faceData, int numFaces)
{
    // Create an OsdHbrMesh that represents the subdiv control hull
    // for the incoming mesh data
    static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex> _catmark;
    OsdHbrMesh *hbrMesh = new OsdHbrMesh(&_catmark);

    OpenSubdiv::OsdVertex v;
    for (int i = 0; i < numVertices; ++i) {
        hbrMesh->NewVertex(i, v);
    }

    std::vector<int> faceVerts;
    int faceDataOffset = 0;
    int ptexFaceIndex = 0;

    for (int fi = 0; fi<numFaces; ++fi) {
        int numFaceVerts = faceData[fi];

        faceVerts.resize(numFaceVerts);
        for (int fvi = 0; fvi < numFaceVerts; ++fvi) {
            faceVerts[fvi] = indexData[fvi + faceDataOffset];
        }

        OsdHbrFace *face = hbrMesh->NewFace(numFaceVerts, &faceVerts[0], 0);

        face->SetPtexIndex(ptexFaceIndex);
        ptexFaceIndex += (numFaceVerts == 4) ? 1 : numFaceVerts;

        faceDataOffset += numFaceVerts;
    }

    hbrMesh->Finish();


    // Do feature adaptive refinement on the subdiv to generate cubic
    // patches
    int level = 6;

    std::cout << "1\n";
    // true for adaptive
    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(
        hbrMesh, level, true);

    std::cout << "2\n";    
    // true for ptex data, false for fvar data
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> * farMesh =
        meshFactory.Create(true, false);

    std::cout << "3\n";        
    // We no longer need the control hull.  Note that the meshFactory.Create()
    // used hbrMesh
    delete hbrMesh;

    std::cout << "4\n";            

/*    
    // Iterate over the ptex coordinates 
    std::vector<int> const &
        ptexCoords = farMesh->GetPtexCoordinates(level);
    for (int i=0; i<ptexCoords.size(); i+=2) {
        const int *p = &ptexCoords[i];

        int faceIndex = p[0];

        int u = (p[1] >> 16) & 0xffff;
        int v = (p[1] & 0xffff);

        std::cerr << faceIndex << " " << u << " " << v;
        std::cerr << " --> face: " << faceIndex;
        std::cerr << " uvoffset: ("
                  << (float)u/(1<<level) << ", " << (float)v/(1<<level)
                  << ")";
        if (faceIndex < 0) {
            std::cerr << " non-quad coarse face";
        }
        std::cerr << "\n";
    }
*/


    const OpenSubdiv::FarPatchTables *patchTables = farMesh->GetPatchTables();
    
    if (not patchTables) {
        std::cout << "Oooops- patchTables not constructed\n";
        return;
    } else {
        std::cout << "Have some patchTables\n";
    }
    
    const OpenSubdiv::FarTable<unsigned int> &ptable =
        patchTables->GetFullRegularPatches();


    // Iterate over all patches in this table.  Don't worry about markers here,
    // those would tell use what level of subdivision the patch was created on.
    // Just iterate over all patches, this is in blocks of 16 unsigned ints
    // per patch.
    const unsigned int *vertIndices = ptable[0];
    for (int i=0; i<ptable.GetSize(); i+=16) {
        std::cout << "patch at index " << i << "\n";

        // Create a patch object
        for (int j=0; j<16; ++j) {
            std::cout << " " << vertIndices[i+j];
        }
        std::cout << "\n";
    }
    
}

static void
buildCube()
{
    float vertexData[] = {
        1.0f, 1.0f, 1.0f,
       -1.0f, 1.0f, 1.0f,
       -1.0f,-1.0f, 1.0f,
        1.0f,-1.0f, 1.0f,

       -1.0f,-1.0f,-1.0f,
       -1.0f, 1.0f,-1.0f,
        1.0f, 1.0f,-1.0f,
        1.0f,-1.0f,-1.0f,
    };
    int numVertices = (sizeof(vertexData) / sizeof(vertexData[0])) / 3;

    int indexData[] = {
        0, 1, 2, 3,
        4, 5, 6, 7,

        0, 3, 7, 6,
        4, 2, 1, 5,

        0, 6, 5, 1,
        4, 7, 3, 2,
    };
    int numIndices = sizeof(indexData) / sizeof(indexData[0]);
    
    int faceData[] = {
        4, 4, 4, 4, 4, 4,
    };
    int numFaces = sizeof(faceData) / sizeof(faceData[0]);

    buildMesh(vertexData, numVertices,
              indexData, numIndices,
              faceData, numFaces);
}

static void
buildTet()
{
    float vertexData[] = {
        -0.5, -0.78, 0.0,
         0.5, -0.78, 0.0,
         0.0,  0.78, 0.5,
         0.0,  0.78, 0.5,
    };
    int numVertices = (sizeof(vertexData) / sizeof(vertexData[0])) / 3;

    int indexData[] = {
        0, 2, 2,
        1, 3, 2,
        3, 0, 2,
        0, 3, 1,
    };
    int numIndices = sizeof(indexData) / sizeof(indexData[0]);

    int faceData[] = {
        3, 3, 3, 3,
    };
    int numFaces = sizeof(faceData) / sizeof(faceData[0]);
    
    buildMesh(vertexData, numVertices,
              indexData, numIndices,
              faceData, numFaces);
}

static void
buildPrism()
{
    float vertexData[] = {
        0.0, 0.0, 1.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        0.0, 0.0,-1.0,
        1.0, 0.0,-1.0,
        0.0, 1.0,-1.0,
    };
    int numVertices = (sizeof(vertexData) / sizeof(vertexData[0])) / 3;

    int indexData[] = {
        0, 1, 2,
        3, 5, 4,
        0, 3, 4, 1,
        0, 2, 5, 3,
        1, 4, 5, 2,
    };
    int numIndices = sizeof(indexData) / sizeof(indexData[0]);
    
    int faceData[] = {
        3, 3, 4, 4, 4,
    };
    int numFaces = sizeof(faceData) / sizeof(faceData[0]);

    buildMesh(vertexData, numVertices,
              indexData, numIndices,
              faceData, numFaces);
}

int vpWidth = 512, vpHeight = 512;

int
main(int argc, char **argv)
{
    std::cerr << "Cube\n";
//    buildCube();
    std::cerr << "\n";

    std::cerr << "Tet\n";
    buildTet();
    std::cerr << "\n";

    std::cerr << "Prism\n";
    buildPrism();
    std::cerr << "\n";

    /*
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glutMainLoop();
    */
}
