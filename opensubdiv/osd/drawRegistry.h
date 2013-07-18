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
#ifndef OSD_DRAW_REGISTRY_H
#define OSD_DRAW_REGISTRY_H

#include "../version.h"

#include "../osd/drawContext.h"

#include <utility>
#include <string>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdDrawShaderSource {
    typedef std::pair< std::string, std::string > Define;
    typedef std::vector< Define > DefineVector;

    void AddDefine(std::string const & name,
                   std::string const & value = "1") {
        defines.push_back( Define(name, value) );
    }

    std::string source;
    std::string version;
    std::string target;
    std::string entry;

    DefineVector defines;
};

struct OsdDrawConfig {
    virtual ~OsdDrawConfig();
    // any base class behaviors?
};

struct OsdDrawSourceConfig {
    virtual ~OsdDrawSourceConfig();
    // any base class behaviors?
};

////////////////////////////////////////////////////////////

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_DRAW_REGISTRY_H */
