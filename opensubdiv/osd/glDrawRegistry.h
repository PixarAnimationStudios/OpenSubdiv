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
#ifndef OSD_GL_DRAW_REGISTRY_H
#define OSD_GL_DRAW_REGISTRY_H

#include "../version.h"

#include "../far/mesh.h"
#include "../osd/drawRegistry.h"
#include "../osd/vertex.h"

#include "../osd/opengl.h"

#include <map>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdGLDrawConfig : public OsdDrawConfig {
    OsdGLDrawConfig() :
        program(0),
        primitiveIdBaseUniform(-1),
        gregoryQuadOffsetBaseUniform(-1) {}
    virtual ~OsdGLDrawConfig();

    GLuint program;
    GLint primitiveIdBaseUniform;
    GLint gregoryQuadOffsetBaseUniform;
};

struct OsdGLDrawSourceConfig : public OsdDrawSourceConfig {
    OsdDrawShaderSource commonShader;
    OsdDrawShaderSource vertexShader;
    OsdDrawShaderSource tessControlShader;
    OsdDrawShaderSource tessEvalShader;
    OsdDrawShaderSource geometryShader;
    OsdDrawShaderSource fragmentShader;
};

////////////////////////////////////////////////////////////

class OsdGLDrawRegistryBase {
public:
    typedef OsdDrawContext::PatchDescriptor DescType;
    typedef OsdGLDrawConfig ConfigType;
    typedef OsdGLDrawSourceConfig SourceConfigType;

    virtual ~OsdGLDrawRegistryBase();

protected:
    virtual ConfigType * _NewDrawConfig() { return new ConfigType(); }
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc,
                      SourceConfigType const * sconfig);

    virtual SourceConfigType * _NewDrawSourceConfig() { return new SourceConfigType(); }
    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc);
};

template <class DESC_TYPE = OsdDrawContext::PatchDescriptor,
          class CONFIG_TYPE = OsdGLDrawConfig,
          class SOURCE_CONFIG_TYPE = OsdGLDrawSourceConfig >
class OsdGLDrawRegistry : public OsdGLDrawRegistryBase {
public:
    typedef OsdGLDrawRegistryBase BaseRegistry;

    typedef DESC_TYPE DescType;
    typedef CONFIG_TYPE ConfigType;
    typedef SOURCE_CONFIG_TYPE SourceConfigType;

    typedef std::map<DescType, ConfigType *> ConfigMap;

public:
    virtual ~OsdGLDrawRegistry() {
        Reset();
    }

    void Reset() {
        for (typename ConfigMap::iterator
                i = _configMap.begin(); i != _configMap.end(); ++i) {
            delete i->second;
        }
        _configMap.clear();
    }

    // fetch shader config
    ConfigType *
    GetDrawConfig(DescType const & desc) {
        typename ConfigMap::iterator it = _configMap.find(desc);
        if (it != _configMap.end()) {
            return it->second;
        } else {
            ConfigType * config =
                _CreateDrawConfig(desc, _CreateDrawSourceConfig(desc));
            _configMap[desc] = config;
            return config;
        }
    }

protected:
    virtual ConfigType * _NewDrawConfig() { return new ConfigType(); }
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc,
                      SourceConfigType const * sconfig) { return NULL; }

    virtual SourceConfigType * _NewDrawSourceConfig() { return new SourceConfigType(); }
    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc) { return NULL; }

private:
    ConfigMap _configMap;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_GL_DRAW_REGISTRY_H */
