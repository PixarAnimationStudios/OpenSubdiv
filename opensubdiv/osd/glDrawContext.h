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
#ifndef OSD_GL_DRAW_CONTEXT_H
#define OSD_GL_DRAW_CONTEXT_H

#if defined(__APPLE__)
    #include "TargetConditionals.h"
    #if TARGET_OS_IPHONE or TARGET_IPHONE_SIMULATOR
        #include <OpenGLES/ES2/gl.h>
    #else
        #include <OpenGL/gl3.h>
    #endif
#elif defined(ANDROID)
    #include <GLES2/gl2.h>
#else
    #if defined(_WIN32)
        #include <windows.h>
    #endif
    #include <GL/gl.h>
#endif

#include "../version.h"

#include "../far/mesh.h"
#include "../osd/drawContext.h"
#include "../osd/drawRegistry.h"
#include "../osd/vertex.h"

#include <map>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGLDrawContext : public OsdDrawContext {
public:
    typedef GLuint VertexBufferBinding;

    OsdGLDrawContext();
    virtual ~OsdGLDrawContext();

    template<class VERTEX_BUFFER>
    static OsdGLDrawContext *
    Create(FarMesh<OsdVertex> *farMesh,
           VERTEX_BUFFER *vertexBuffer,
           bool requirePtexCoordinates=false,
           bool requireFVarData=false) {

        if (not vertexBuffer) 
            return NULL;

        OsdGLDrawContext * instance = new OsdGLDrawContext();
        if (instance->allocate(farMesh,
                               vertexBuffer->BindVBO(),
                               vertexBuffer->GetNumElements(),
                               requirePtexCoordinates,
                               requireFVarData)) return instance;
        delete instance;
        return NULL;
    }

    GLuint patchIndexBuffer;

#if defined(GL_ES_VERSION_2_0)
    GLuint patchTrianglesIndexBuffer;
#endif

    GLuint ptexCoordinateTextureBuffer;
    GLuint fvarDataTextureBuffer;

    GLuint vertexTextureBuffer;
    GLuint vertexValenceTextureBuffer;
    GLuint quadOffsetTextureBuffer;
    GLuint patchLevelTextureBuffer;

    /// true if the GL version detected supports shader tessellation
    static bool SupportsAdaptiveTessellation();

private:
    bool allocate(FarMesh<OsdVertex> *farMesh,
                  GLuint vbo, int numElements,
                  bool requirePtexCoordinates,
                  bool requireFVarData);

    void _AppendPatchArray(
                int *indexBase, int *levelBase,
                FarPatchTables::PTable const & ptable, int patchSize,
                FarPatchTables::PtexCoordinateTable const & ptexTable,
                FarPatchTables::FVarDataTable const & fvarTable, int fvarDataWidth,
                OsdPatchDescriptor const & desc,
                int gregoryQuadOffsetBase);
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_DRAW_CONTEXT_H */
