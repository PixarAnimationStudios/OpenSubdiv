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

#ifndef OSD_GL_DRAW_REGISTRY_H
#define OSD_GL_DRAW_REGISTRY_H

#include "../version.h"

#include "../osd/drawRegistry.h"
#include "../osd/vertex.h"

#include "../osd/opengl.h"

#include <map>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

struct GLDrawConfig : public DrawConfig {

    GLDrawConfig() :
        program(0) { }

    virtual ~GLDrawConfig();

    GLuint program;
};

//------------------------------------------------------------------------------

struct GLDrawSourceConfig : public DrawSourceConfig {
    DrawShaderSource commonShader;
    DrawShaderSource vertexShader;
    DrawShaderSource tessControlShader;
    DrawShaderSource tessEvalShader;
    DrawShaderSource geometryShader;
    DrawShaderSource fragmentShader;
};

//------------------------------------------------------------------------------

class GLDrawRegistryBase {

public:
    typedef DrawContext::PatchDescriptor DescType;
    typedef GLDrawConfig ConfigType;
    typedef GLDrawSourceConfig SourceConfigType;

    GLDrawRegistryBase(bool enablePtex=false) : _enablePtex(enablePtex) { }

    virtual ~GLDrawRegistryBase();

    bool IsPtexEnabled() const {
        return _enablePtex;
    }
    
    void SetPtexEnabled(bool b) {
        _enablePtex=b;
    }

protected:
    virtual ConfigType * _NewDrawConfig() {
        return new ConfigType(); 
    }

    virtual ConfigType * _CreateDrawConfig(DescType const & desc, 
                                           SourceConfigType const * sconfig);

    virtual SourceConfigType * _NewDrawSourceConfig() { 
        return new SourceConfigType(); 
    }
    
    virtual SourceConfigType * _CreateDrawSourceConfig(DescType const & desc);

private:
    bool _enablePtex;
};

//------------------------------------------------------------------------------

template <class DESC_TYPE = DrawContext::PatchDescriptor,
          class CONFIG_TYPE = GLDrawConfig,
          class SOURCE_CONFIG_TYPE = GLDrawSourceConfig >

class GLDrawRegistry : public GLDrawRegistryBase {

public:
    typedef DESC_TYPE DescType;
    typedef CONFIG_TYPE ConfigType;
    typedef SOURCE_CONFIG_TYPE SourceConfigType;

    typedef GLDrawRegistryBase BaseRegistry;

    typedef std::map<DescType, ConfigType *> ConfigMap;

    virtual ~GLDrawRegistry() {
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
    ConfigType * GetDrawConfig(DescType const & desc) {
    
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
    virtual ConfigType * _NewDrawConfig() { 
        return new ConfigType(); 
    }
    
    virtual ConfigType * _CreateDrawConfig(DescType const & /* desc */,
                                           SourceConfigType const * /* sconfig */) {
        return NULL; 
    }

    virtual SourceConfigType * _NewDrawSourceConfig() { 
        return new SourceConfigType(); 
    }
    
    virtual SourceConfigType * _CreateDrawSourceConfig(DescType const & /* desc */) { 
        return NULL; 
    }

private:
    ConfigMap _configMap;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_GL_DRAW_REGISTRY_H */
