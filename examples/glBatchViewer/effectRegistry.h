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
#ifndef EFFECT_REGISTRY_H
#define EFFECT_REGISTRY_H

#include <osd/glDrawRegistry.h>
#include "effect.h"

typedef std::pair<OpenSubdiv::OsdDrawContext::PatchDescriptor, EffectDesc> FullEffectDesc; // name!

class MyEffectRegistry : public OpenSubdiv::OsdGLDrawRegistry<FullEffectDesc, MyDrawConfig> {
    typedef OpenSubdiv::OsdGLDrawRegistry<FullEffectDesc, MyDrawConfig> BaseClass;

protected:
    virtual ConfigType *_CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig);
    virtual SourceConfigType *_CreateDrawSourceConfig(DescType const & desc);

public:
    ConfigType * GetDrawConfig(EffectDesc effectDesc, OpenSubdiv::OsdDrawContext::PatchDescriptor desc) {
        return BaseClass::GetDrawConfig(FullEffectDesc(desc, effectDesc));
    }
};
    

#endif  /* EFFECT_REGISTRY_H */
