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
#ifndef OSD_UTIL_DRAW_CONTROLLER_H
#define OSD_UTIL_DRAW_CONTROLLER_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/*
  concept DrawDelegate
  {
    void BindBatch(OsdUtilMeshBatchBase<DRAW_CONTEXT> *batch);
    void BindEffect(EFFECT *effect);
    void DrawElements(EFFECT *effect, OsdDrawContext::PatchArray const &patchArray);
  }
*/

namespace OsdUtil {

    // ------------------------------------------------------------------------
    // DrawCollection
    // ------------------------------------------------------------------------
    template <typename DRAW_ITEM_COLLECTION, typename DRAW_DELEGATE>
    void DrawCollection(DRAW_ITEM_COLLECTION const &items, DRAW_DELEGATE *delegate) {

        typedef typename DRAW_ITEM_COLLECTION::value_type DrawItem;

        // XXX: Are these caches needed in this class? user delegate can do that?
        bool first = true;
        typename DrawItem::BatchBase *currentBatch = NULL;
        typename DrawItem::EffectHandle currentEffect; // XXX: initial value for pointer?
    
        // iterate over DrawItemCollection
        for (typename DRAW_ITEM_COLLECTION::const_iterator it = items.begin(); it != items.end(); ++it) {
            typename DrawItem::BatchBase *batch = it->GetBatch();
            typename DrawItem::EffectHandle effect = it->GetEffect();
            if (first || currentBatch != batch) {
                if (not first) delegate->UnbindBatch(currentBatch);
                delegate->BindBatch(batch);
                currentBatch = batch;
            }
            if (first || currentEffect != effect) {
                if (not first) delegate->UnbindEffect(currentEffect);
                delegate->BindEffect(effect);
                currentEffect = effect;
            }

            // iterate over sub items within a draw item
            OsdDrawContext::PatchArrayVector const &patchArrays = it->GetPatchArrays();
            for (OsdDrawContext::PatchArrayVector::const_iterator pit = patchArrays.begin(); pit != patchArrays.end(); ++pit) {
                delegate->DrawElements(currentEffect, *pit);
            }

            first = false;
        }
        if (not first) delegate->UnbindEffect(currentEffect);
        if (not first) delegate->UnbindBatch(currentBatch);
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
        template <typename DRAW_ITEM_COLLECTION, typename BATCH, typename EFFECT>
        void emit(DRAW_ITEM_COLLECTION &result, BATCH *batch, EFFECT effect) {

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
    template <typename DRAW_ITEM_COLLECTION>
    void OptimizeDrawItem(DRAW_ITEM_COLLECTION const &items,
                          DRAW_ITEM_COLLECTION &result) {

        typedef typename DRAW_ITEM_COLLECTION::value_type DrawItem;

        typename DrawItem::BatchBase *currentBatch = NULL;
        typename DrawItem::EffectHandle currentEffect; // XXX: initial value?

        result.reserve(items.size());

        PatchArrayCombiner combiner;
        for (typename DRAW_ITEM_COLLECTION::const_iterator it = items.begin(); it != items.end(); ++it) {
            typename DrawItem::BatchBase *batch = it->GetBatch();
            typename DrawItem::EffectHandle effect = it->GetEffect();

            if (currentBatch != batch || currentEffect != effect) {
                // emit cached draw item
                combiner.emit(result, currentBatch, currentEffect);
                
                currentBatch = batch;
                currentEffect = effect;
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
        combiner.emit(result, currentBatch, currentEffect);
    }
};


}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  /* OSD_UTIL_DRAW_CONTROLLER_H */
