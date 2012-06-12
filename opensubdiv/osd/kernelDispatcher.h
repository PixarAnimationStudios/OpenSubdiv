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

#include "../version.h"
#include "../far/dispatcher.h"

#include <GL/glew.h>
#include <map>
#include <string>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdKernelDispatcher : public FarDispatcher<OsdVertex> {

public:

    OsdKernelDispatcher (int maxLevel) : _maxLevel(maxLevel) { }
    virtual ~OsdKernelDispatcher() { }


    virtual void CopyTable(int tableIndex, size_t size, const void *ptr) = 0;


    virtual void BeginLaunchKernel() = 0;
    
    virtual void EndLaunchKernel() = 0;


    virtual void BindVertexBuffer(GLuint vertexBuffer, GLuint varyingBuffer) = 0;

    virtual void UpdateVertexBuffer(size_t size, void *ptr) = 0;

    virtual void UpdateVaryingBuffer(size_t size, void *ptr) = 0;

    virtual void MapVertexBuffer() = 0;

    virtual void MapVaryingBuffer() = 0;

    virtual void UnmapVertexBuffer() = 0;

    virtual void UnmapVaryingBuffer() = 0;

    virtual void Synchronize() = 0;

    template<class T> void UpdateTable(int tableIndex, const T & table) {
    
        CopyTable(tableIndex, table.GetMemoryUsed(), table[0]);

        _tableOffsets[tableIndex].resize(_maxLevel);
        for (int i = 0; i < _maxLevel; ++i)
            _tableOffsets[tableIndex][i] = table[i] - table[0];
    }

    static OsdKernelDispatcher *CreateKernelDispatcher( const std::string &kernel, int levels, int numVertexElements, int numVaryingElements ) {
    
        return Factory::GetInstance().Create( kernel, levels, numVertexElements, numVaryingElements );
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
        typedef OsdKernelDispatcher *(*Creator)( int levels, int numVertexElements, int numVaryingElements );
        typedef std::map<const std::string, Creator> CreatorMap;

        bool Register(const std::string &kernel, Creator creator) {
            return _creators.insert(CreatorMap::value_type(kernel, creator)).second;
        }

        bool Unregister(const std::string &kernel) {
            return _creators.erase(kernel) == 1;
        }

        OsdKernelDispatcher *Create ( const std::string &kernel, int levels, int numVertexElements, int numVaryingElements ) {
            CreatorMap::const_iterator it = _creators.find(kernel);
            if (it != _creators.end())
                return (it->second)(levels, numVertexElements, numVaryingElements);
            return NULL;
        }
	
        static Factory &GetInstance() {
            return _instance;
        }
        
	static Factory _instance;

    private:
        
	CreatorMap _creators;
    };

protected:
    int _maxLevel;
    std::vector<int> _tableOffsets[TABLE_MAX];
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif  /* OSD_KERNEL_DISPATCHER_H */
