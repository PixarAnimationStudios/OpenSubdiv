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
#ifndef OSD_D3D11_DRAW_REGISTRY_H
#define OSD_D3D11_DRAW_REGISTRY_H

#include "../version.h"

#include "../far/mesh.h"
#include "../osd/drawRegistry.h"
#include "../osd/vertex.h"

#include <map>

struct ID3D11VertexShader;
struct ID3D11HullShader;
struct ID3D11DomainShader;
struct ID3D11GeometryShader;
struct ID3D11PixelShader;

struct ID3D11Buffer;
struct ID3D11ShaderResourceView;
struct ID3D11Device;

struct ID3D11InputLayout;
struct D3D11_INPUT_ELEMENT_DESC;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdD3D11DrawConfig : public OsdDrawConfig {
    OsdD3D11DrawConfig() :
        vertexShader(0), hullShader(0), domainShader(0),
        geometryShader(0), pixelShader(0) {}
    virtual ~OsdD3D11DrawConfig();

    ID3D11VertexShader   *vertexShader;
    ID3D11HullShader     *hullShader;
    ID3D11DomainShader   *domainShader;
    ID3D11GeometryShader *geometryShader;
    ID3D11PixelShader    *pixelShader;
};

struct OsdD3D11DrawSourceConfig {
    OsdDrawShaderSource commonShader;
    OsdDrawShaderSource vertexShader;
    OsdDrawShaderSource hullShader;
    OsdDrawShaderSource domainShader;
    OsdDrawShaderSource geometryShader;
    OsdDrawShaderSource pixelShader;
};

////////////////////////////////////////////////////////////

class OsdD3D11DrawRegistryBase {
public:
    typedef OsdDrawContext::PatchDescriptor DescType;
    typedef OsdD3D11DrawConfig ConfigType;
    typedef OsdD3D11DrawSourceConfig SourceConfigType;

    virtual ~OsdD3D11DrawRegistryBase();

protected:
    virtual ConfigType * _NewDrawConfig() { return new ConfigType(); }
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc,
                      SourceConfigType const * sconfig,
                      ID3D11Device * pd3dDevice,
                      ID3D11InputLayout ** ppInputLayout,
                      D3D11_INPUT_ELEMENT_DESC const * pInputElementDescs,
                      int numInputElements);

    virtual SourceConfigType * _NewDrawSourceConfig() { return new SourceConfigType(); }
    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc, ID3D11Device * pd3dDevice);
};

template <class DESC_TYPE = OsdDrawContext::PatchDescriptor,
          class CONFIG_TYPE = OsdD3D11DrawConfig,
          class SOURCE_CONFIG_TYPE = OsdD3D11DrawSourceConfig >
class OsdD3D11DrawRegistry : public OsdD3D11DrawRegistryBase {
public:
    typedef OsdD3D11DrawRegistryBase BaseRegistry;

    typedef DESC_TYPE DescType;
    typedef CONFIG_TYPE ConfigType;
    typedef SOURCE_CONFIG_TYPE SourceConfigType;

    typedef std::map<DescType, ConfigType *> ConfigMap;

public:
    virtual ~OsdD3D11DrawRegistry() {
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
    GetDrawConfig(DescType const & desc,
                  ID3D11Device * pd3dDevice,
                  ID3D11InputLayout ** ppInputLayout = NULL,
                  D3D11_INPUT_ELEMENT_DESC const * pInputElementDescs = NULL,
                  int numInputElements = 0) {
        typename ConfigMap::iterator it = _configMap.find(desc);
        if (it != _configMap.end()) {
            return it->second;
        } else {
            ConfigType * config =
                _CreateDrawConfig(desc,
                                  _CreateDrawSourceConfig(desc, pd3dDevice),
                                  pd3dDevice,
                                  ppInputLayout,
                                  pInputElementDescs, numInputElements);
            _configMap[desc] = config;
            return config;
        }
    }

protected:
    virtual ConfigType * _NewDrawConfig() { return new ConfigType(); }
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc,
                      SourceConfigType const * sconfig,
                      ID3D11Device * pd3dDevice,
                      ID3D11InputLayout ** ppInputLayout,
                      D3D11_INPUT_ELEMENT_DESC const * pInputElementDescs,
                      int numInputElements) { return NULL; }

    virtual SourceConfigType * _NewDrawSourceConfig() { return new SourceConfigType(); }
    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc, ID3D11Device * pd3dDevice) { return NULL; }

private:
    ConfigMap _configMap;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_D3D11_DRAW_REGISTRY_H */
