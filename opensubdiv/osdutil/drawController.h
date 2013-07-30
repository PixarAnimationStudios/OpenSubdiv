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
#ifndef OSDUTIL_DRAW_CONTROLLER_H
#define OSDUTIL_DRAW_CONTROLLER_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/*
  concept DrawDelegate
  {
    void Begin();
    void End();
    void Bind(OsdUtilMeshBatchBase<DRAW_CONTEXT> *batch, EffectHandle effect);
    void DrawElements(OsdDrawContext::PatchArray const &patchArray);
    bool IsCombinable(EffectHandle &a, EffectHandle &b)
  }
*/

namespace OsdUtil {

    // ------------------------------------------------------------------------
    // DrawCollection
    // ------------------------------------------------------------------------
    template <typename DRAW_ITEM_COLLECTION, typename DRAW_DELEGATE>
    void DrawCollection(DRAW_ITEM_COLLECTION const &items, DRAW_DELEGATE *delegate) {

        typedef typename DRAW_ITEM_COLLECTION::value_type DrawItem;

        delegate->Begin();
    
        // iterate over DrawItemCollection
        for (typename DRAW_ITEM_COLLECTION::const_iterator it = items.begin(); it != items.end(); ++it) {

            delegate->Bind(it->GetBatch(), it->GetEffect());

            // iterate over sub items within a draw item
            OsdDrawContext::PatchArrayVector const &patchArrays = it->GetPatchArrays();
            for (OsdDrawContext::PatchArrayVector::const_iterator pit = patchArrays.begin(); pit != patchArrays.end(); ++pit) {
                delegate->DrawElements(*pit);
            }

            delegate->Unbind(it->GetBatch(), it->GetEffect());
        }

        delegate->End();
    }

    // ------------------------------------------------------------------------
    struct PatchArrayCombiner {
        typedef std::map<OsdDrawContext::PatchDescriptor, OsdDrawContext::PatchArrayVector> Dictionary;

        struct PatchArrayComparator {
            bool operator() (OsdDrawContext::PatchArray const &a, OsdDrawContext::PatchArray const &b) {
                return a.GetDescriptor() < b.GetDescriptor() or ((a.GetDescriptor() == b.GetDescriptor()) and
                                                                 (a.GetVertIndex() < b.GetVertIndex()));
            }
        };

        // XXX: reconsider this function
        template <typename DRAW_ITEM_COLLECTION, typename BATCH, typename EFFECT_HANDLE>
        void emit(DRAW_ITEM_COLLECTION &result, BATCH *batch, EFFECT_HANDLE &effect) {

            if (dictionary.empty()) return;
            
            typename DRAW_ITEM_COLLECTION::value_type item(batch, effect);
            for (Dictionary::iterator it = dictionary.begin(); it != dictionary.end(); ++it) {
                if (it->second.empty()) continue;
                
                // expecting patchArrays is already sorted mostly.
                std::sort(it->second.begin(), it->second.end(), PatchArrayComparator());
                
                for (OsdDrawContext::PatchArrayVector::iterator pit = it->second.begin(); pit != it->second.end(); ++pit) {
                    if (not item.GetPatchArrays().empty()) {
                        OsdDrawContext::PatchArray &back = item.GetPatchArrays().back();
                        if (back.GetDescriptor() == pit->GetDescriptor() &&
                            back.GetVertIndex() + back.GetNumIndices() == pit->GetVertIndex()) {
                            // combine together
                            back.SetNumPatches(back.GetNumPatches() + pit->GetNumPatches());
                            continue;
                        }
                    }
                    // append to item
                    item.GetPatchArrays().push_back(*pit);
                }
            }

            result.push_back(item);

            for (Dictionary::iterator it = dictionary.begin(); it != dictionary.end(); ++it) {
                it->second.clear();
            }
        }

        void append(OsdDrawContext::PatchArray const &patchArray) {
            dictionary[patchArray.GetDescriptor()].push_back(patchArray);
        }
        Dictionary dictionary;
        
    };

    // ------------------------------------------------------------------------
    // OptimizeDrawItem
    // ------------------------------------------------------------------------
    template <typename DRAW_ITEM_COLLECTION, typename DRAW_DELEGATE>
    void OptimizeDrawItem(DRAW_ITEM_COLLECTION const &items,
                          DRAW_ITEM_COLLECTION &result,
                          DRAW_DELEGATE *delegate) {

        typedef typename DRAW_ITEM_COLLECTION::value_type DrawItem;

        if (items.empty()) return;
        result.reserve(items.size());

        typename DrawItem::BatchBase *currentBatch = items[0].GetBatch();
        typename DrawItem::EffectHandle const *currentEffect = &(items[0].GetEffect());

        PatchArrayCombiner combiner;
        for (typename DRAW_ITEM_COLLECTION::const_iterator it = items.begin(); it != items.end(); ++it) {
            typename DrawItem::BatchBase *batch = it->GetBatch();
            typename DrawItem::EffectHandle const &effect = it->GetEffect();

            if (currentBatch != batch or
                (not delegate->IsCombinable(*currentEffect, effect))) {

                // emit cached draw item
                combiner.emit(result, currentBatch, *currentEffect);
                
                currentBatch = batch;
                currentEffect = &effect;
            }

            // merge consecutive items if possible. This operation changes drawing order.
            // i.e.
            //      PrimA-Regular, PrimA-Transition, PrimB-Regular, PrimB-Transition
            // becomes
            //      PrimA-Regular, PrimB-Regular, PrimA-Transition, PrimB-Transition
            OsdDrawContext::PatchArrayVector const &patchArrays = it->GetPatchArrays();
            for (OsdDrawContext::PatchArrayVector::const_iterator itp = patchArrays.begin(); itp != patchArrays.end(); ++itp) {
                if (itp->GetNumPatches() == 0) continue;
                // insert patchArrays into dictionary
                combiner.append(*itp);
            }
        }
        
        // pick up after
        combiner.emit(result, currentBatch, *currentEffect);
    }
};


}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  /* OSDUTIL_DRAW_CONTROLLER_H */
