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
#ifndef HBRCREASEEDIT_H
#define HBRCREASEEDIT_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrCreaseEdit;

template <class T>
std::ostream& operator<<(std::ostream& out, const HbrCreaseEdit<T>& path) {
    out << "edge path = (" << path.faceid << ' ';
    for (int i = 0; i < path.nsubfaces; ++i) {
        out << static_cast<int>(path.subfaces[i]) << ' ';
    }
    return out << static_cast<int>(path.edgeid) << "), sharpness = " << path.sharpness;
}

template <class T>
class HbrCreaseEdit : public HbrHierarchicalEdit<T> {

public:

    HbrCreaseEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces, unsigned char _edgeid, typename HbrHierarchicalEdit<T>::Operation _op, float _sharpness)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), edgeid(_edgeid), op(_op), sharpness(_sharpness) {
    }

    HbrCreaseEdit(int _faceid, int _nsubfaces, int *_subfaces, int _edgeid, typename HbrHierarchicalEdit<T>::Operation _op, float _sharpness)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), edgeid(static_cast<unsigned char>(_edgeid)), op(_op), sharpness(_sharpness) {
    }

    virtual ~HbrCreaseEdit() {}

    friend std::ostream& operator<< <T> (std::ostream& out, const HbrCreaseEdit<T>& path);

    virtual void ApplyEditToFace(HbrFace<T>* face) {
        if (HbrHierarchicalEdit<T>::GetNSubfaces() == face->GetDepth()) {
            // Modify edge sharpness
            float sharp=0.0f;
            if (op == HbrHierarchicalEdit<T>::Set) {
                sharp = sharpness;
            } else if (op == HbrHierarchicalEdit<T>::Add) {
                sharp = face->GetEdge(edgeid)->GetSharpness() + sharpness;
            } else if (op == HbrHierarchicalEdit<T>::Subtract) {
                sharp = face->GetEdge(edgeid)->GetSharpness() - sharpness;
            }
            if (sharp < HbrHalfedge<T>::k_Smooth)
                sharp = HbrHalfedge<T>::k_Smooth;
            if (sharp > HbrHalfedge<T>::k_InfinitelySharp)
                sharp = HbrHalfedge<T>::k_InfinitelySharp;
            // We have to make sure the neighbor of the edge exists at
            // this point. Otherwise, if it comes into being late, it
            // will clobber the overriden sharpness and we will lose
            // the edit.
            face->GetEdge(edgeid)->GuaranteeNeighbor();
            face->GetEdge(edgeid)->SetSharpness(sharp);
        }
    }

private:
    // ID of the edge (you can think of this also as the id of the
    // origin vertex of the two-vertex length edge)
    const unsigned char edgeid;
    typename HbrHierarchicalEdit<T>::Operation op;
    // sharpness of the edge edit
    const float sharpness;
};


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* HBRCREASEEDIT_H */
