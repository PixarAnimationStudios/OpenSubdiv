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
#ifndef HBRCORNEREDIT_H
#define HBRCORNEREDIT_H

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
            if (sharp < HbrVertex<T>::k_Smooth) {
                sharp = HbrVertex<T>::k_Smooth;
            }
            if (sharp > HbrVertex<T>::k_InfinitelySharp) {
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

#endif /* HBRCORNEREDIT_H */
