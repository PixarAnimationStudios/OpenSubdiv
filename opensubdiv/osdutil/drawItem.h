//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//
#ifndef OSDUTIL_DRAW_ITEM_H
#define OSDUTIL_DRAW_ITEM_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <typename DRAW_CONTEXT> class OsdUtilMeshBatchBase;

// ----------------------------------------------------------------------------
//   (glommable) draw item
// ----------------------------------------------------------------------------
template <typename EFFECT_HANDLE, typename DRAW_CONTEXT>
class OsdUtilDrawItem {
public:
    typedef EFFECT_HANDLE EffectHandle;
    typedef DRAW_CONTEXT DrawContext;
    typedef OsdUtilMeshBatchBase<DRAW_CONTEXT> BatchBase;
    typedef std::vector<OsdUtilDrawItem<EFFECT_HANDLE, DRAW_CONTEXT> > Collection;
    
    // contructors
    // creates empty draw item
    OsdUtilDrawItem(OsdUtilMeshBatchBase<DRAW_CONTEXT> *batch,
                    EffectHandle effect) : 
        _batch(batch), _effect(effect) {}

    // creates draw item with given patcharray
    OsdUtilDrawItem(OsdUtilMeshBatchBase<DRAW_CONTEXT> *batch,
                    EffectHandle effect,
                    OsdDrawContext::PatchArrayVector const &patchArrays) : 
        _batch(batch), _effect(effect), _patchArrays(patchArrays) {}
        
    // accessors will be called by draw controller
    BatchBase * GetBatch() const { return _batch; }
    EffectHandle const & GetEffect() const { return _effect; }
    EffectHandle & GetEffect() { return _effect; }
    OsdDrawContext::PatchArrayVector const &GetPatchArrays() const { return _patchArrays; }
    OsdDrawContext::PatchArrayVector &GetPatchArrays() { return _patchArrays; }
    
private:
    // data members
    BatchBase *_batch;
    EffectHandle _effect;
    OsdDrawContext::PatchArrayVector _patchArrays;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  /* OSDUTIL_DRAW_ITEM_H */
