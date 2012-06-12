#ifndef GSD_VERTEX_H
#define GSD_VERTEX_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdVertex {
public:
    OsdVertex() {}
    OsdVertex(int index) {}
    OsdVertex(const OsdVertex &src) {}

    void AddWithWeight(const OsdVertex & i, float weight, void * = 0) {}
    void AddVaryingWithWeight(const OsdVertex & i, float weight, void * = 0) {} 
    void Clear(void * = 0) {}
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // GSD_VERTEX_H
