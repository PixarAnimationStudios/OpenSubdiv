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
#ifndef OSD_KERNEL_DISPATCHER_H
#define OSD_KERNEL_DISPATCHER_H

#include "vertex.h"
#include "vertexBuffer.h"

#include "../version.h"
#include "../far/dispatcher.h"

#include <map>
#include <string>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdKernelDispatcher : public FarDispatcher<OsdVertex> {

public:

    OsdKernelDispatcher (int maxLevel) : _maxLevel(maxLevel) { }
    virtual ~OsdKernelDispatcher() { }

    enum KernelType { kCPU = 0,
                      kOPENMP = 1,
                      kCUDA = 2,
                      kGLSL = 3,
                      kCL = 4,
                      kMAX };


    virtual void CopyTable(int tableIndex, size_t size, const void *ptr) = 0;

    virtual void AllocateEditTables(int n) = 0;

    virtual void UpdateEditTable(int tableIndex, const FarTable<unsigned int> &offsets, const FarTable<float> &values,
                                 int operation, int primVarOffset, int primVarWidth) = 0;


    virtual void OnKernelLaunch() = 0;

    virtual void OnKernelFinish() = 0;

    virtual OsdVertexBuffer *InitializeVertexBuffer(int numElements, int count) = 0;

    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) = 0;

    virtual void UnbindVertexBuffer() = 0;

    virtual void Synchronize() = 0;

    template<class T> void UpdateTable(int tableIndex, const T & table) {

        CopyTable(tableIndex, table.GetMemoryUsed(), table[0]);

        _tableOffsets[tableIndex].resize(_maxLevel);
        for (int i = 0; i < _maxLevel; ++i)
            _tableOffsets[tableIndex][i] = (int)(table[i] - table[0]);
    }

    static OsdKernelDispatcher *CreateKernelDispatcher( int levels, int kernel ) {
        return Factory::GetInstance().Create( levels, kernel );
    }

    static int GetNumRegisteredKernels() {
        return Factory::GetInstance().GetNumRegisteredKernels();
    }

    static bool HasKernelType(KernelType kernel) {
        return Factory::GetInstance().HasKernelType(kernel);
    }

    enum { E_IT,
           E_W,
           V_ITa,
           V_IT,
           V_W,
           F_IT,
           F_ITa,
           TABLE_MAX };

protected:

    class Factory {

    public:
        Factory();

        typedef OsdKernelDispatcher *(*Creator)( int levels );

        int Register(Creator creator) {
            _kernelCreators.push_back(creator);
            return (int)_kernelCreators.size() - 1;
        }

        OsdKernelDispatcher *Create ( int levels, int kernel ) {
            if (kernel >= (int)_kernelCreators.size())
                return NULL;

            return (_kernelCreators[kernel])(levels);
        }

        int GetNumRegisteredKernels() const {
            return (int)_kernelCreators.size();
        }

        bool HasKernelType(KernelType kernel) const {
            if (kernel >= (int)_kernelCreators.size()) {
                return false;
            }
            return (_kernelCreators[kernel] != NULL);
        }

        static Factory &GetInstance() {
            return _instance;
        }

        static Factory _instance;

    protected:
        friend class OsdCpuKernelDispatcher;
        friend class OsdGlslKernelDispatcher;
        friend class OsdCudaKernelDispatcher;
        friend class OsdClKernelDispatcher;

        int Register(Creator creator, KernelType kernel) {
            _kernelCreators.resize(kMAX, NULL);
            _kernelCreators[kernel] = creator;
            return (int)kernel;
        }

    private:
        std::vector<Creator> _kernelCreators;
    };

protected:
    int _maxLevel;
    std::vector<int> _tableOffsets[TABLE_MAX];

    struct VertexEditArrayInfo {
        std::vector<int> offsetOffsets;
        std::vector<int> valueOffsets;
        std::vector<int> numEdits;
        int operation;
        int primVarOffset;
        int primVarWidth;
    };
    std::vector<VertexEditArrayInfo> _edits;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif  /* OSD_KERNEL_DISPATCHER_H */
