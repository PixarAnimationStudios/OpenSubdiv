//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_HBRCORNEREDIT_H
#define OPENSUBDIV3_HBRCORNEREDIT_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrCornerEdit;

template <class T>
std::ostream& operator<<(std::ostream& out, const HbrCornerEdit<T>& path) {
    out << "vertex path = (" << path.faceid << ' ';
    for (int i = 0; i < path.nsubfaces; ++i) {
        out << static_cast<int>(path.subfaces[i]) << ' ';
    }
    return out << static_cast<int>(path.vertexid) << "), sharpness = " << path.sharpness;
}

template <class T>
class HbrCornerEdit : public HbrHierarchicalEdit<T> {

public:

    HbrCornerEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces, unsigned char _vertexid, typename HbrHierarchicalEdit<T>::Operation _op, float _sharpness)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), vertexid(_vertexid), op(_op), sharpness(_sharpness) {
    }

    HbrCornerEdit(int _faceid, int _nsubfaces, int *_subfaces, int _vertexid, typename HbrHierarchicalEdit<T>::Operation _op, float _sharpness)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), vertexid(static_cast<unsigned char>(_vertexid)), op(_op), sharpness(_sharpness) {
    }

    virtual ~HbrCornerEdit() {}

    friend std::ostream& operator<< <T> (std::ostream& out, const HbrCornerEdit<T>& path);

    virtual void ApplyEditToFace(HbrFace<T>* face) {
        if (HbrHierarchicalEdit<T>::GetNSubfaces() == face->GetDepth()) {
            // Modify vertex sharpness. Note that we could actually do
            // this in ApplyEditToVertex as well!
            float sharp = 0.0f;
            if (op == HbrHierarchicalEdit<T>::Set) {
                sharp = sharpness;
            } else if (op == HbrHierarchicalEdit<T>::Add) {
                sharp = face->GetVertex(vertexid)->GetSharpness() + sharpness;
            } else if (op == HbrHierarchicalEdit<T>::Subtract) {
                sharp = face->GetVertex(vertexid)->GetSharpness() - sharpness;
            }
            if (sharp < (float) HbrVertex<T>::k_Smooth) {
                sharp = HbrVertex<T>::k_Smooth;
            }
            if (sharp > (float) HbrVertex<T>::k_InfinitelySharp) {
                sharp = HbrVertex<T>::k_InfinitelySharp;
            }
            face->GetVertex(vertexid)->SetSharpness(sharp);
        }
    }

private:
    // ID of the edge (you can think of this also as the id of the
    // origin vertex of the two-vertex length edge)
    const unsigned char vertexid;
    typename HbrHierarchicalEdit<T>::Operation op;
    // sharpness of the vertex edit
    const float sharpness;
};


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_HBRCORNEREDIT_H */
