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
#ifndef HBRFVARDATA_H
#define HBRFVARDATA_H

#include <cstring>
#include <cmath>

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrFVarEdit;
template <class T> class HbrFace;
template <class T> class HbrVertex;

// This class implements a "face varying vector item". Really it's
// just a smart wrapper around face varying data (itself just a bunch
// of floats) stored on each vertex.
template <class T> class HbrFVarData {

private:
    HbrFVarData()
        : faceid(0), initialized(0) {
    }

    ~HbrFVarData() {
        Uninitialize();
    }

    HbrFVarData(const HbrFVarData &data) {}

public:
    
    // Sets the face id
    void SetFaceID(int id) {
        faceid = id;
    }

    // Returns the id of the face to which this data is bound
    int GetFaceID() const {
        return faceid;
    }

    // Clears the initialized flag
    void Uninitialize() {
        initialized = 0;
        faceid = 0;
    }

    // Returns initialized flag
    bool IsInitialized() const {
        return initialized;
    }

    // Sets initialized flag
    void SetInitialized() {
        initialized = 1;
    }

    // Return the data from the NgpFVVector
    float* GetData(int item) { return data + item; }    

    // Clears the indicates value of this item
    void Clear(int startindex, int width) {
        memset(data + startindex, 0, width * sizeof(float));
    }

    // Clears all values of this item
    void ClearAll(int width) {
        initialized = 1;
        memset(data, 0, width * sizeof(float));
    }

    // Set values of the indicated item (with the indicated weighing)
    // on this item
    void SetWithWeight(const HbrFVarData& fvvi, int startindex, int width, float weight) {
        float *dst = data + startindex;
        const float *src = fvvi.data + startindex;
        for (int i = 0; i < width; ++i) {
            *dst++ = weight * *src++;
        }
    }

    // Add values of the indicated item (with the indicated weighing)
    // to this item
    void AddWithWeight(const HbrFVarData& fvvi, int startindex, int width, float weight) {
        float *dst = data + startindex;
        const float *src = fvvi.data + startindex;
        for (int i = 0; i < width; ++i) {
            *dst++ += weight * *src++;
        }
    }

    // Add all values of the indicated item (with the indicated
    // weighing) to this item
    void AddWithWeightAll(const HbrFVarData& fvvi, int width, float weight) {
        float *dst = data;
        const float *src = fvvi.data;
        for (int i = 0; i < width; ++i) {
            *dst++ += weight * *src++;
        }
    }

    // Compare all values item against a float buffer. Returns true
    // if all values match
    bool CompareAll(int width, const float *values, float tolerance=0.0f) const {
        if (!initialized) return false;
        for (int i = 0; i < width; ++i) {
            if (fabsf(values[i] - data[i]) > tolerance) return false;
        }
        return true;
    }

    // Initializes data
    void SetAllData(int width, const float *values) {
        initialized = 1;
        memcpy(data, values, width * sizeof(float));
    }

    // Compare this item against another item with tolerance.  Returns
    // true if it compares identical
    bool Compare(const HbrFVarData& fvvi, int startindex, int width, float tolerance=0.0f) const {
        for (int i = 0; i < width; ++i) {
            if (fabsf(data[startindex + i] - fvvi.data[startindex + i]) > tolerance) return false;
        }
        return true;
    }

    // Modify the data of the item with an edit
    void ApplyFVarEdit(const HbrFVarEdit<T>& edit);

    friend class HbrVertex<T>;
    
private:
    unsigned int faceid:31;
    unsigned int initialized:1;
    float data[1];
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#include "../hbr/fvarEdit.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T>
void
HbrFVarData<T>::ApplyFVarEdit(const HbrFVarEdit<T>& edit) {
        float *dst = data + edit.GetIndex() + edit.GetOffset();
        const float *src = edit.GetEdit();
        for (int i = 0; i < edit.GetWidth(); ++i) {
            switch(edit.GetOperation()) {
                case HbrVertexEdit<T>::Set:
                    *dst++ = *src++;
                    break;
                case HbrVertexEdit<T>::Add:
                    *dst++ += *src++;
                    break;
                case HbrVertexEdit<T>::Subtract:
                    *dst++ -= *src++;
            }
        }
        initialized = 1;
    }


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* HBRFVARDATA_H */
