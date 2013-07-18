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

#ifndef EXAMPLES_MAYAVIEWER_OPENSUBDIVSHADER_H_
#define EXAMPLES_MAYAVIEWER_OPENSUBDIVSHADER_H_

// full include needed for scheme/interp enums
// class OsdMeshData;
#include "osdMeshData.h"            

#include <osd/glDrawContext.h>

#include <maya/MPxHwShaderNode.h>
#include <maya/MViewport2Renderer.h>
#include <maya/MImage.h>

class OpenSubdivShader : public MPxHwShaderNode 
{
public:
    OpenSubdivShader();
    virtual ~OpenSubdivShader();

    // MPxNode methods
    static void *creator();
    static MStatus initialize();

    // MPxNode virtuals
    virtual void    postConstructor();
    virtual MStatus compute(const MPlug &plug, MDataBlock &data);
    virtual bool    getInternalValueInContext(const MPlug &plug,       MDataHandle &handle, MDGContext &);
    virtual bool    setInternalValueInContext(const MPlug &plug, const MDataHandle &handle, MDGContext &);

    // MPxHwShaderNode virtuals
    virtual MStatus renderSwatchImage(MImage & image);

    //
    // Non-Viewport 2.0 rendering:  geometry/glGeometry, bind/glBind and unbind/glUnbind
    // are not implemented at this time.  osdMayaViewer must be run under Viewport 2.0.
    //

    void updateAttributes();
    void draw(const MHWRender::MDrawContext &context, OsdMeshData *data);

    // Accessors
    void setHbrMeshDirty(bool dirty) { _hbrMeshDirty = dirty; }
    bool getHbrMeshDirty() { return _hbrMeshDirty; }

    int getLevel() const { return _level; }
    bool isAdaptive() const { return _adaptive; }

    OsdMeshData::SchemeType getScheme() const { return _scheme; }
    OsdMeshData::KernelType getKernel() const { return _kernel; }
    OsdMeshData::InterpolateBoundaryType getInterpolateBoundary() const { return _interpolateBoundary; }
    OsdMeshData::InterpolateBoundaryType getInterpolateUVBoundary() const { return _interpolateUVBoundary; }

    const MString getUVSet() const { return _uvSet; }

    // Identifiers
    static MTypeId id;
    static MString drawRegistrantId;

    // Offsets from GL_TEXTURE0
    static const int DIFF_TEXTURE_UNIT=14;

private:

    void updateRegistry();

    GLuint bindTexture(const MString& filename, int textureUnit);
    GLuint bindProgram(const MHWRender::MDrawContext &     mDrawContext,
                             OpenSubdiv::OsdGLDrawContext *osdDrawContext,
                       const OpenSubdiv::OsdDrawContext::PatchArray &   patch);

    // OSD attributes
    static MObject aLevel;
    static MObject aTessFactor;
    static MObject aScheme;
    static MObject aKernel;
    static MObject aInterpolateBoundary;
    static MObject aAdaptive;
    static MObject aWireframe;

    // Material attributes
    static MObject aDiffuse;
    static MObject aAmbient;
    static MObject aSpecular;
    static MObject aShininess;

    // Texture attributes
    static MObject aDiffuseMapFile;
    static MObject aUVSet;
    static MObject aInterpolateUVBoundary;

    // Shader
    static MObject aShaderSource;

private:

    // OSD parameters
    int  _level;
    int  _tessFactor;
    bool _adaptive;
    bool _wireframe;

    OsdMeshData::SchemeType  _scheme;
    OsdMeshData::KernelType  _kernel;
    OsdMeshData::InterpolateBoundaryType  _interpolateBoundary;

    // Material parameters
    MColor _diffuse;
    MColor _ambient;
    MColor _specular;
    float _shininess;

    // Texture manager
    MHWRender::MTextureManager *_theTextureManager;

    // Texture parameters
    MString _diffuseMapFile;
    MString _uvSet;
    OsdMeshData::InterpolateBoundaryType  _interpolateUVBoundary;

    // Shader
    std::string _shaderSource;
    MString     _shaderSourceFilename;

    // Plugin flags
    bool _hbrMeshDirty;
    bool _adaptiveDirty;
    bool _diffuseMapDirty;
    bool _shaderSourceDirty;
};

#endif  // EXAMPLES_MAYAVIEWER_OPENSUBDIVSHADER_H_
