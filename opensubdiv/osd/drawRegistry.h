//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
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

namespace Osd {

struct DrawShaderSource {
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

struct DrawConfig {
    virtual ~DrawConfig();
    // any base class behaviors?
};

struct DrawSourceConfig {
    virtual ~DrawSourceConfig();
    // any base class behaviors?
};

////////////////////////////////////////////////////////////

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_DRAW_REGISTRY_H */
