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
#ifndef HBRFVAREDIT_H
#define HBRFVAREDIT_H

#include "../hbr/hierarchicalEdit.h"
#include "../hbr/vertexEdit.h"

template <class T> class HbrFVarEdit;

template <class T>
std::ostream& operator<<(std::ostream& out, const HbrFVarEdit<T>& path) {
    out << "vertex path = (" << path.faceid << ' ';
    for (int i = 0; i < path.nsubfaces; ++i) {
	out << static_cast<int>(path.subfaces[i]) << ' ';
    }
    return out << static_cast<int>(path.vertexid) << "), edit = (" << path.edit[0] << ',' << path.edit[1] << ',' << path.edit[2] << ')';
}

template <class T>
class HbrFVarEdit : public HbrHierarchicalEdit<T> {

public:

    HbrFVarEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces, unsigned char _vertexid, int _index, int _width, int _offset, typename HbrHierarchicalEdit<T>::Operation _op, float *_edit)
	: HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), vertexid(_vertexid), index(_index), width(_width), offset(_offset), op(_op) {
	edit = new float[width];
	memcpy(edit, _edit, width * sizeof(float));
    }
    
    HbrFVarEdit(int _faceid, int _nsubfaces, int *_subfaces, int _vertexid, int _index, int _width, int _offset, typename HbrHierarchicalEdit<T>::Operation _op, float *_edit)
	: HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), vertexid(_vertexid), index(_index), width(_width), offset(_offset), op(_op) {
	edit = new float[width];
	memcpy(edit, _edit, width * sizeof(float));
    }
    
    virtual ~HbrFVarEdit() {
	delete[] edit;
    }

    // Return the vertex id (the last element in the path)
    unsigned char GetVertexID() const { return vertexid; }
    
    friend std::ostream& operator<< <T> (std::ostream& out, const HbrFVarEdit<T>& path);

    // Return index into the facevarying data
    int GetIndex() const { return index; }

    // Return width of the data
    int GetWidth() const { return width; }

    // Return offset of the data
    int GetOffset() const { return offset; }
    
    // Get the numerical value of the edit
    const float* GetEdit() const { return edit; }

    // Get the type of operation
    typename HbrHierarchicalEdit<T>::Operation GetOperation() const { return op; }
    
    virtual void ApplyEditToFace(HbrFace<T>* face) {
	if (HbrHierarchicalEdit<T>::GetNSubfaces() == face->GetDepth()) {
            // The edit will modify the data and almost certainly
            // create a discontinuity, so allocate storage for a new
            // copy of the existing data specific to the face (or use
            // one that already exists) and modify that
            HbrFVarData<T> &fvt = face->GetVertex(vertexid)->GetFVarData(face);
            if (fvt.GetFace() != face) {
                // This is the generic fvt, allocate a new copy and edit it
                HbrFVarData<T> &newfvt = face->GetVertex(vertexid)->NewFVarData(face);
                newfvt.SetAllData(face->GetMesh()->GetTotalFVarWidth(), fvt.GetData(0));
                newfvt.ApplyFVarEdit(*const_cast<const HbrFVarEdit<T>*>(this));
            } else {
                fvt.ApplyFVarEdit(*const_cast<const HbrFVarEdit<T>*>(this));
            }
	}
    }

private:
    const unsigned char vertexid;
    const int index;
    const int width;
    const int offset;
    float* edit;
    typename HbrHierarchicalEdit<T>::Operation op;
};


#endif /* HBRFVAREDIT_H */
