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

#ifndef EXAMPLES_MAYAPTEXVIEWER_OPENSUBDIVPTEXSHADEROVERRIDE_H_
#define EXAMPLES_MAYAPTEXVIEWER_OPENSUBDIVPTEXSHADEROVERRIDE_H_

#include <maya/MPxShaderOverride.h>
#include <maya/MCallbackIdArray.h>

class OpenSubdivPtexShader;
class OsdPtexMeshData;

class OpenSubdivPtexShaderOverride : public MHWRender::MPxShaderOverride 
{
public:
    static MHWRender::MPxShaderOverride* creator(const MObject &obj);

    virtual ~OpenSubdivPtexShaderOverride();

    virtual MString initialize(const MInitContext &initContext,
                               MInitFeedback &initFeedback);

    virtual void updateDG(MObject object);

    virtual void updateDevice();

    virtual void endUpdate();

    virtual bool draw(MHWRender::MDrawContext &context,
                      const MHWRender::MRenderItemList &renderItemList) const;

    virtual bool rebuildAlways() { return false; }
    virtual MHWRender::DrawAPI supportedDrawAPIs() const { return MHWRender::kOpenGL; }
    virtual bool isTransparent() { return true; }

    static void attrChangedCB(MNodeMessage::AttributeMessage msg, MPlug& plug, MPlug& otherPlug, void* );
    void addTopologyChangedCallbacks( const MDagPath& dagPath, OsdPtexMeshData *data );

private:
    explicit OpenSubdivPtexShaderOverride(const MObject &obj);

    OpenSubdivPtexShader *_shader;

    MCallbackIdArray _callbackIds;

    int _level;
};

#endif  // EXAMPLES_MAYAPTEXVIEWER_OPENSUBDIVPTEXSHADEROVERRIDE_H_
