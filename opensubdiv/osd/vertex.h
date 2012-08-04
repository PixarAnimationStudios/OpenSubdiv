#ifndef OSD_VERTEX_H
#define OSD_VERTEX_H

#include "../version.h"
#include "../hbr/face.h"
#include "../hbr/vertexEdit.h"

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
    void ApplyVertexEdit(const OpenSubdiv::HbrVertexEdit<OsdVertex> &) { }
    void ApplyMovingVertexEdit(const OpenSubdiv::HbrMovingVertexEdit<OsdVertex> &) { }
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_VERTEX_H
