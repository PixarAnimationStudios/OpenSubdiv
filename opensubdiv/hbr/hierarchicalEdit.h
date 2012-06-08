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
#ifndef HBRHIERARCHICALEDIT_H
#define HBRHIERARCHICALEDIT_H

template <class T> class HbrHierarchicalEdit;
template <class T> class HbrFace;
template <class T> class HbrVertex;

template <class T>
class HbrHierarchicalEdit {

public:
    typedef enum Operation {
	Set,
	Add,
	Subtract
    } Operation;

protected:

    HbrHierarchicalEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces)
	: faceid(_faceid), nsubfaces(_nsubfaces) {
	subfaces = new unsigned char[_nsubfaces];
	for (int i = 0; i < nsubfaces; ++i) {
	    subfaces[i] = _subfaces[i];
	}
    }

    HbrHierarchicalEdit(int _faceid, int _nsubfaces, int *_subfaces)
	: faceid(_faceid), nsubfaces(_nsubfaces) {
	subfaces = new unsigned char[_nsubfaces];
	for (int i = 0; i < nsubfaces; ++i) {
	    subfaces[i] = static_cast<unsigned char>(_subfaces[i]);
	}
    }

public:
    virtual ~HbrHierarchicalEdit() {
	delete[] subfaces;
    }

    bool operator<(const HbrHierarchicalEdit& p) const {
	if (faceid < p.faceid) return true;
	if (faceid > p.faceid) return false;
	int minlength = nsubfaces;
	if (minlength > p.nsubfaces) minlength = p.nsubfaces;
	for (int i = 0; i < minlength; ++i) {
	    if (subfaces[i] < p.subfaces[i]) return true;
	    if (subfaces[i] > p.subfaces[i]) return false;	    
	}
	return (nsubfaces < p.nsubfaces);
    }

    // Return the face id (the first element in the path)
    int GetFaceID() const { return faceid; }

    // Return the number of subfaces in the path
    int GetNSubfaces() const { return nsubfaces; }

    // Return a subface element in the path
    unsigned char GetSubface(int index) const { return subfaces[index]; }
    
    // Determines whether this hierarchical edit is relevant to the
    // face in question
    bool IsRelevantToFace(HbrFace<T>* face) const;

    // Applys edit to face. All subclasses may override this method
    virtual void ApplyEditToFace(HbrFace<T>* /* face */) {}

    // Applys edit to vertex. Subclasses may override this method.
    virtual void ApplyEditToVertex(HbrFace<T>* /* face */, HbrVertex<T>* /* vertex */) {} 

#ifdef PRMAN
    // Gets the effect of this hierarchical edit on the bounding box.
    // Subclasses may override this method
    virtual void ApplyToBound(struct bbox& /* box */, RtMatrix * /* mx */) {}
#endif

protected:
    // ID of the top most face in the mesh which begins the path
    const int faceid;

    // Number of subfaces
    const int nsubfaces;

    // IDs of the subfaces
    unsigned char *subfaces;
};

template <class T>
class HbrHierarchicalEditComparator {
public:
    bool operator() (const HbrHierarchicalEdit<T>* path1, const HbrHierarchicalEdit<T>* path2) const { 
        return (*path1 < *path2);
    }
};

#include "../hbr/face.h"
#include <string.h>

template <class T>
bool
HbrHierarchicalEdit<T>::IsRelevantToFace(HbrFace<T>* face) const {

    // Key assumption: the face's first vertex edit is relevant to
    // that face. We will then compare ourselves to that edit and if
    // the first part of our subpath is identical to the entirety of
    // that subpath, this edit is relevant.

    // Calling code is responsible for making sure we don't
    // dereference a null pointer here
    HbrHierarchicalEdit<T>* p = *face->GetHierarchicalEdits();
    if (!p) return false;

    if (this == p) return true;
    
    if (faceid != p->faceid) return false;

    // If our path length is less than the face depth, it should mean
    // that we're dealing with another face somewhere up the path, so
    // we're not relevant
    if (nsubfaces < face->GetDepth()) return false;

    if (memcmp(subfaces, p->subfaces, face->GetDepth() * sizeof(unsigned char)) != 0) {
	return false;
    }
    return true;
}

#endif /* HBRHIERARCHICALEDIT_H */
