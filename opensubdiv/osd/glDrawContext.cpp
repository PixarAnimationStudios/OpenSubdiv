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
    #include <GL/glew.h>
#endif

#include "../far/dispatcher.h"
#include "../far/loopSubdivisionTables.h"
#include "../osd/glDrawRegistry.h"
#include "../osd/glDrawContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLDrawContext::OsdGLDrawContext() :
    patchIndexBuffer(0), ptexCoordinateTextureBuffer(0), fvarDataTextureBuffer(0),
    vertexTextureBuffer(0), vertexValenceTextureBuffer(0), quadOffsetTextureBuffer(0),
    patchLevelTextureBuffer(0)
{
}

OsdGLDrawContext::~OsdGLDrawContext()
{
    glDeleteBuffers(1, &patchIndexBuffer);
    glDeleteTextures(1, &vertexTextureBuffer);
    glDeleteTextures(1, &vertexValenceTextureBuffer);
    glDeleteTextures(1, &quadOffsetTextureBuffer);
    glDeleteTextures(1, &patchLevelTextureBuffer);
    glDeleteTextures(1, &ptexCoordinateTextureBuffer);
    glDeleteTextures(1, &fvarDataTextureBuffer);
}

bool
OsdGLDrawContext::SupportsAdaptiveTessellation()
{
// Compile-time check of GL version
#if (defined(GL_ARB_tessellation_shader) or defined(GL_VERSION_4_0)) and defined(GLEW_VERSION_4_0)
    // Run-time check of GL version with GLEW
    if (GLEW_VERSION_4_0) {
        return true;
    }
#endif
    return false;
}

bool
OsdGLDrawContext::allocate(FarMesh<OsdVertex> *farMesh,
                           GLuint vbo,
                           int numElements,
                           bool requirePtexCoordinates,
                           bool requireFVarData)
{
    FarPatchTables const * patchTables = farMesh->GetPatchTables();

    if (not patchTables) {
        // uniform patches
        isAdaptive = false;

        // XXX: farmesh should have FarDensePatchTable for dense mesh indices.
        //      instead of GetFaceVertices().
        const FarSubdivisionTables<OsdVertex> *tables = farMesh->GetSubdivisionTables();
        int level = tables->GetMaxLevel();
        const std::vector<int> &indices = farMesh->GetFaceVertices(level-1);

        // XXX: farmesh or FarSubdivisionTables should have a virtual method
        // to determine loop or not
        bool loop =
            dynamic_cast<const FarLoopSubdivisionTables<OsdVertex>*>(tables) != NULL;

        int numIndices = (int)indices.size();

        // Allocate and fill index buffer.
        glGenBuffers(1, &patchIndexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, patchIndexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     numIndices * sizeof(unsigned int), &(indices[0]), GL_STATIC_DRAW);

#if defined(GL_ES_VERSION_2_0)
        // OpenGLES 2 supports only triangle topologies for filled
        // primitives i.e. not QUADS or PATCHES or LINES_ADJACENCY
        // For the convenience of clients build build a triangles
        // index buffer by splitting quads.
        int numQuads = indices.size() / 4;
        int numTrisIndices = numQuads * 6;

        std::vector<short> trisIndices;
        trisIndices.reserve(numTrisIndices);
        for (int i=0; i<numQuads; ++i) {
            const int * quad = &indices[i*4];
            trisIndices.push_back(short(quad[0]));
            trisIndices.push_back(short(quad[1]));
            trisIndices.push_back(short(quad[2]));

            trisIndices.push_back(short(quad[2]));
            trisIndices.push_back(short(quad[3]));
            trisIndices.push_back(short(quad[0]));
        }

        // Allocate and fill triangles index buffer.
        glGenBuffers(1, &patchTrianglesIndexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, patchTrianglesIndexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     numTrisIndices * sizeof(short), &(trisIndices[0]), GL_STATIC_DRAW);
#endif

        OsdPatchArray array;
        array.desc.type = kNonPatch;
        array.patchSize = loop ? 3 : 4;
        array.firstIndex = 0;
        array.numIndices = numIndices;

        patchArrays.push_back(array);

        // Allocate ptex coordinate buffer if requested (for non-adaptive)
        if (requirePtexCoordinates) {
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
            GLuint ptexCoordinateBuffer = 0;
            glGenTextures(1, &ptexCoordinateTextureBuffer);
            glGenBuffers(1, &ptexCoordinateBuffer);
            glBindBuffer(GL_TEXTURE_BUFFER, ptexCoordinateBuffer);

            const std::vector<int> &ptexCoordinates =
                farMesh->GetPtexCoordinates(level-1);
            int size = (int)ptexCoordinates.size() * sizeof(GLint);

            glBufferData(GL_TEXTURE_BUFFER, size, &(ptexCoordinates[0]), GL_STATIC_DRAW);

            glBindTexture(GL_TEXTURE_BUFFER, ptexCoordinateTextureBuffer);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32I, ptexCoordinateBuffer);
            glBindTexture(GL_TEXTURE_BUFFER, 0);
            glDeleteBuffers(1, &ptexCoordinateBuffer);

#endif
        }

        // Allocate fvar data buffer if requested (for non-adaptive)
        if (requireFVarData) {
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
            GLuint fvarDataBuffer = 0;
            glGenTextures(1, &fvarDataTextureBuffer);
            glGenBuffers(1, &fvarDataBuffer);
            glBindBuffer(GL_TEXTURE_BUFFER, fvarDataBuffer);

            const std::vector<float> &fvarData = farMesh->GetFVarData(level-1);
            int size = (int)fvarData.size() * sizeof(float);

            glBufferData(GL_TEXTURE_BUFFER, size, &(fvarData[0]), GL_STATIC_DRAW);

            glBindTexture(GL_TEXTURE_BUFFER, fvarDataTextureBuffer);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, fvarDataBuffer);
            glDeleteBuffers(1, &fvarDataBuffer);
            glBindTexture(GL_TEXTURE_BUFFER, 0);
#endif
        }

        return true;
    }

    // adaptive patches
    isAdaptive = true;

    // Determine buffer sizes
    int totalPatchIndices = 
        patchTables->GetFullRegularPatches().GetSize() +
        patchTables->GetFullBoundaryPatches().GetSize() +
        patchTables->GetFullCornerPatches().GetSize() +
        patchTables->GetFullGregoryPatches().GetSize() +
        patchTables->GetFullBoundaryGregoryPatches().GetSize();

    int totalPatchLevels = 
        patchTables->GetFullRegularPatches().GetSize()/patchTables->GetRegularPatchRingsize() +
        patchTables->GetFullBoundaryPatches().GetSize()/patchTables->GetBoundaryPatchRingsize() +
        patchTables->GetFullCornerPatches().GetSize()/patchTables->GetCornerPatchRingsize() +
        patchTables->GetFullGregoryPatches().GetSize()/patchTables->GetGregoryPatchRingsize() +
        patchTables->GetFullBoundaryGregoryPatches().GetSize()/patchTables->GetGregoryPatchRingsize();

    for (unsigned char p=0; p<5; ++p) {
        totalPatchIndices +=
            patchTables->GetTransitionRegularPatches(p).GetSize();

        totalPatchLevels +=
            patchTables->GetTransitionRegularPatches(p).GetSize()/patchTables->GetRegularPatchRingsize();

        for (unsigned char r=0; r<4; ++r) {
            totalPatchIndices +=
                patchTables->GetTransitionBoundaryPatches(p, r).GetSize() +
                patchTables->GetTransitionCornerPatches(p, r).GetSize();

            totalPatchLevels +=
                patchTables->GetTransitionBoundaryPatches(p, r).GetSize()/patchTables->GetBoundaryPatchRingsize() +
                patchTables->GetTransitionCornerPatches(p, r).GetSize()/patchTables->GetCornerPatchRingsize();
        }
    }

    // Allocate and fill index buffer.
    glGenBuffers(1, &patchIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, patchIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        totalPatchIndices * sizeof(unsigned int), NULL, GL_STATIC_DRAW);

#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
    GLuint patchLevelBuffer = 0;
    glGenBuffers(1, &patchLevelBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, patchLevelBuffer);
    glBufferData(GL_TEXTURE_BUFFER,
        totalPatchLevels * sizeof(unsigned char), NULL, GL_STATIC_DRAW);
#endif

    // Allocate ptex coordinate buffer if requested
    GLuint ptexCoordinateBuffer = 0;
    if (requirePtexCoordinates) {
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
        glGenTextures(1, &ptexCoordinateTextureBuffer);
        glGenBuffers(1, &ptexCoordinateBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, ptexCoordinateBuffer);
        glBufferData(GL_ARRAY_BUFFER,
            totalPatchLevels * sizeof(int) * 2, NULL, GL_STATIC_DRAW);
#endif
    }

    // Allocate fvar data buffer if requested
    GLuint fvarDataBuffer = 0;
    if (requireFVarData) {
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
        glGenTextures(1, &fvarDataTextureBuffer);
        glGenBuffers(1, &fvarDataBuffer);
        glBindBuffer(GL_UNIFORM_BUFFER, fvarDataBuffer);
        glBufferData(GL_UNIFORM_BUFFER,
            totalPatchLevels * sizeof(float) * farMesh->GetTotalFVarWidth()*4, NULL, GL_STATIC_DRAW);
#endif
    }

    int indexBase = 0;
    int levelBase = 0;
    int maxValence = patchTables->GetMaxValence();

    _AppendPatchArray(&indexBase, &levelBase,
                        patchTables->GetFullRegularPatches(),
                        patchTables->GetRegularPatchRingsize(),
                        patchTables->GetFullRegularPtexCoordinates(),
                        patchTables->GetFullRegularFVarData(),
                        farMesh->GetTotalFVarWidth(),
                        OsdPatchDescriptor(kRegular, 0, 0, 0, 0), 0);
    _AppendPatchArray(&indexBase, &levelBase,
                        patchTables->GetFullBoundaryPatches(),
                        patchTables->GetBoundaryPatchRingsize(),
                        patchTables->GetFullBoundaryPtexCoordinates(),
                        patchTables->GetFullBoundaryFVarData(),
                        farMesh->GetTotalFVarWidth(),
                        OsdPatchDescriptor(kBoundary, 0, 0, 0, 0), 0);
    _AppendPatchArray(&indexBase, &levelBase,
                        patchTables->GetFullCornerPatches(),
                        patchTables->GetCornerPatchRingsize(),
                        patchTables->GetFullCornerPtexCoordinates(),
                        patchTables->GetFullCornerFVarData(),
                        farMesh->GetTotalFVarWidth(),
                        OsdPatchDescriptor(kCorner, 0, 0, 0, 0), 0);
    _AppendPatchArray(&indexBase, &levelBase,
                        patchTables->GetFullGregoryPatches(),
                        patchTables->GetGregoryPatchRingsize(),
                        patchTables->GetFullGregoryPtexCoordinates(),
                        patchTables->GetFullGregoryFVarData(),
                        farMesh->GetTotalFVarWidth(),
                        OsdPatchDescriptor(kGregory, 0, 0,
                                           static_cast<unsigned char>(maxValence), static_cast<unsigned char>(numElements)),
                        0);
    _AppendPatchArray(&indexBase, &levelBase,
                        patchTables->GetFullBoundaryGregoryPatches(),
                        patchTables->GetGregoryPatchRingsize(),
                        patchTables->GetFullBoundaryGregoryPtexCoordinates(),
                        patchTables->GetFullBoundaryGregoryFVarData(),
                        farMesh->GetTotalFVarWidth(),
                        OsdPatchDescriptor(kBoundaryGregory, 0, 0,
                                           static_cast<unsigned char>(maxValence), static_cast<unsigned char>(numElements)),
                        (int)patchTables->GetFullGregoryPatches().GetSize());

    for (unsigned char p=0; p<5; ++p) {
        _AppendPatchArray(&indexBase, &levelBase,
                        patchTables->GetTransitionRegularPatches(p),
                        patchTables->GetRegularPatchRingsize(),
                        patchTables->GetTransitionRegularPtexCoordinates(p),
                        patchTables->GetTransitionRegularFVarData(p),
                        farMesh->GetTotalFVarWidth(),
                        OsdPatchDescriptor(kTransitionRegular, p, 0, 0, 0), 0);
        for (unsigned char r=0; r<4; ++r) {
            _AppendPatchArray(&indexBase, &levelBase,
                        patchTables->GetTransitionBoundaryPatches(p, r),
                        patchTables->GetBoundaryPatchRingsize(),
                        patchTables->GetTransitionBoundaryPtexCoordinates(p, r),
                        patchTables->GetTransitionBoundaryFVarData(p, r),
                        farMesh->GetTotalFVarWidth(),
                        OsdPatchDescriptor(kTransitionBoundary, p, r, 0, 0), 0);
            _AppendPatchArray(&indexBase, &levelBase,
                        patchTables->GetTransitionCornerPatches(p, r),
                        patchTables->GetCornerPatchRingsize(),
                        patchTables->GetTransitionCornerPtexCoordinates(p, r),
                        patchTables->GetTransitionCornerFVarData(p, r),
                        farMesh->GetTotalFVarWidth(),
                        OsdPatchDescriptor(kTransitionCorner, p, r, 0, 0), 0);
        }
    }

#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
    // finalize level texture buffer
    glGenTextures(1, &patchLevelTextureBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, patchLevelTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R8I, patchLevelBuffer);
    glDeleteBuffers(1, &patchLevelBuffer);

    // finalize ptex coordinate texture buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    if (requirePtexCoordinates) {
        glBindTexture(GL_TEXTURE_BUFFER, ptexCoordinateTextureBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32I, ptexCoordinateBuffer);
        glDeleteBuffers(1, &ptexCoordinateBuffer);
    }
    // finalize fvar data texture buffer
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    if (requireFVarData) {
        glBindTexture(GL_TEXTURE_BUFFER, fvarDataTextureBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, fvarDataBuffer);
        glDeleteBuffers(1, &fvarDataBuffer);
    }
#endif

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
#endif

    // allocate and initialize additional buffer data
    FarPatchTables::VertexValenceTable const &
        valenceTable = patchTables->GetVertexValenceTable();

    if (not valenceTable.empty()) {
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
        GLuint buffer = 0;
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_TEXTURE_BUFFER, buffer);
        glBufferData(GL_TEXTURE_BUFFER,
                valenceTable.size() * sizeof(unsigned int),
                &valenceTable[0], GL_STATIC_DRAW);

        glGenTextures(1, &vertexValenceTextureBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, vertexValenceTextureBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, buffer);
        glDeleteBuffers(1, &buffer);

        glGenTextures(1, &vertexTextureBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, vertexTextureBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, vbo);

        glBindBuffer(GL_TEXTURE_BUFFER, 0);
#endif
    }

    FarPatchTables::QuadOffsetTable const &
        quadOffsetTable = patchTables->GetQuadOffsetTable();

    if (not quadOffsetTable.empty()) {
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
        GLuint buffer = 0;
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_TEXTURE_BUFFER, buffer);
        glBufferData(GL_TEXTURE_BUFFER,
                quadOffsetTable.size() * sizeof(unsigned int),
                &quadOffsetTable[0], GL_STATIC_DRAW);

        glGenTextures(1, &quadOffsetTextureBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, quadOffsetTextureBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, buffer);
        glDeleteBuffers(1, &buffer);

        glBindBuffer(GL_TEXTURE_BUFFER, 0);
#endif
    }

    return true;
}

void
OsdGLDrawContext::_AppendPatchArray(
        int *indexBase, int *levelBase,
        FarPatchTables::PTable const & ptable, int patchSize,
        FarPatchTables::PtexCoordinateTable const & ptexTable,
        FarPatchTables::FVarDataTable const & fvarTable, int fvarDataWidth,
        OsdPatchDescriptor const & desc, int gregoryQuadOffsetBase)
{
    if (ptable.IsEmpty()) {
        return;
    } 

    OsdPatchArray array;
    array.desc = desc;
    array.patchSize = patchSize;
    array.firstIndex = *indexBase;
    array.numIndices = ptable.GetSize();
    array.levelBase = *levelBase;
    array.gregoryQuadOffsetBase = gregoryQuadOffsetBase;

    int numSubPatches = 1;
    if (desc.type == OpenSubdiv::kTransitionRegular or
        desc.type == OpenSubdiv::kTransitionBoundary or
        desc.type == OpenSubdiv::kTransitionCorner) {
        int subPatchCounts[] = { 3, 4, 4, 4, 2 };
        numSubPatches = subPatchCounts[desc.pattern];
    }

    for (int subpatch = 0; subpatch < numSubPatches; ++subpatch) {
        array.desc.subpatch = subpatch;
        patchArrays.push_back(array);
    }

    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,
        array.firstIndex * sizeof(unsigned int),
        array.numIndices * sizeof(unsigned int),
        ptable[0]);
    *indexBase += array.numIndices;

    std::vector<unsigned char> levels;
    levels.reserve(ptable.GetSize());
    for (int i = 0; i < (int) ptable.GetMarkers().size()-1; ++i) {
        int numPrims = ptable.GetNumElements(i)/array.patchSize;
        for (int j = 0; j < numPrims; ++j) {
            levels.push_back(static_cast<unsigned char>(i));
        }
    }

#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
    glBufferSubData(GL_TEXTURE_BUFFER,
                    array.levelBase * sizeof(unsigned char),
                    levels.size() * sizeof(unsigned char),
                    &levels[0]);
    *levelBase += (int)levels.size();
#endif

    if (ptexCoordinateTextureBuffer) {
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
        assert(ptexTable.size()/2 == levels.size());

        // populate ptex coordinates
        glBufferSubData(GL_ARRAY_BUFFER,
                        array.levelBase * sizeof(int) * 2,
                        (int)ptexTable.size() * sizeof(int),
                        &ptexTable[0]);
#endif
    }

    if (fvarDataTextureBuffer) {
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
        assert(fvarTable.size()/(fvarDataWidth*4) == levels.size());

        // populate fvar data
        glBufferSubData(GL_UNIFORM_BUFFER,
                        array.levelBase * sizeof(float) * fvarDataWidth*4,
                        (int)fvarTable.size() * sizeof(float),
                        &fvarTable[0]);
#endif
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
