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
#ifndef EXAMPLES_MAYAPTEXVIEWER_OPENSUBDIVPTEXSHADER_H_
#define EXAMPLES_MAYAPTEXVIEWER_OPENSUBDIVPTEXSHADER_H_

// full include needed for scheme/interp enums
// class OsdPtexMeshData;
#include "osdPtexMeshData.h"            

#include <osd/glDrawContext.h>
#include <osd/glPtexTexture.h>

#include <maya/MPxHwShaderNode.h>
#include <maya/MViewport2Renderer.h>
#include <maya/MImage.h>

class OpenSubdivPtexShader : public MPxHwShaderNode 
{
public:
    OpenSubdivPtexShader();
    virtual ~OpenSubdivPtexShader();

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
    // are not implemented at this time.  mayaViewer must be run under Viewport 2.0.
    //

    void updateAttributes();
    void draw(const MHWRender::MDrawContext &context, OsdPtexMeshData *data);

    // Accessors
    void setHbrMeshDirty(bool dirty) { _hbrMeshDirty = dirty; }
    bool getHbrMeshDirty() { return _hbrMeshDirty; }

    int getLevel() const { return _level; }
    bool isAdaptive() const { return _adaptive; }

    OsdPtexMeshData::SchemeType getScheme() const { return _scheme; }
    OsdPtexMeshData::KernelType getKernel() const { return _kernel; }
    OsdPtexMeshData::InterpolateBoundaryType getInterpolateBoundary() const { return _interpolateBoundary; }

    // Identifiers
    static MTypeId id;
    static MString drawRegistrantId;

    // Offsets from GL_TEXTURE0
    static const int CLR_TEXTURE_UNIT  =  5;
    static const int DISP_TEXTURE_UNIT =  8;
    static const int OCC_TEXTURE_UNIT  = 11;
    static const int DIFF_TEXTURE_UNIT = 14;
    static const int ENV_TEXTURE_UNIT  = 15;

private:
    void updateRegistry();

    GLuint bindTexture(const MString& filename, int textureUnit);
    GLuint bindProgram(const MHWRender::MDrawContext &     mDrawContext,
                             OpenSubdiv::OsdGLDrawContext *osdDrawContext,
                       const OpenSubdiv::OsdPatchArray &   patch);

    OpenSubdiv::OsdGLPtexTexture * loadPtex(const MString &filename);
    bool bindPtexTexture(const MString& ptexFilename, 
                                   OpenSubdiv::OsdGLPtexTexture **osdPtexPtr,
                                   int samplerUnit);

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

    // Shader
    static MObject aShaderSource;

    // Ptex-specific attributes
    static MObject aDiffuseEnvironmentMapFile;
    static MObject aSpecularEnvironmentMapFile;
    static MObject aColorFile;
    static MObject aDisplacementFile;
    static MObject aOcclusionFile;

    static MObject aEnableDisplacement;
    static MObject aEnableColor;
    static MObject aEnableOcclusion;
    static MObject aEnableNormal;

    static MObject aFresnelBias;
    static MObject aFresnelScale;
    static MObject aFresnelPower;

private:

    // OSD parameters
    int  _level;
    int  _tessFactor;
    bool _adaptive;
    bool _wireframe;

    OsdPtexMeshData::SchemeType  _scheme;
    OsdPtexMeshData::KernelType  _kernel;
    OsdPtexMeshData::InterpolateBoundaryType  _interpolateBoundary;

    // Material parameters
    MColor _diffuse;
    MColor _ambient;
    MColor _specular;

    // Texture manager
    MHWRender::MTextureManager *_theTextureManager;

    // Ptex-specific parameters
    bool _enableColor;
    bool _enableDisplacement;
    bool _enableOcclusion;
    bool _enableNormal;

    MString _ptexColorFile;
    MColor _ptexDisplacementFile;
    MColor _ptexOcclusionFile;
    OpenSubdiv::OsdGLPtexTexture * _ptexColor;
    OpenSubdiv::OsdGLPtexTexture * _ptexDisplacement;
    OpenSubdiv::OsdGLPtexTexture * _ptexOcclusion;

    MString _colorFile;
    MString _displacementFile;
    MString _occlusionFile;

    MString _diffEnvMapFile;
    MString _specEnvMapFile;

    float _fresnelBias;
    float _fresnelScale;
    float _fresnelPower;

    // Shader
    std::string _shaderSource;
    MString     _shaderSourceFilename;

    // Plugin flags
    bool _hbrMeshDirty;
    bool _adaptiveDirty;
    bool _diffEnvMapDirty;
    bool _specEnvMapDirty;

    bool _ptexColorDirty;
    bool _ptexDisplacementDirty;
    bool _ptexOcclusionDirty;

    bool _shaderSourceDirty;

};

#endif  // EXAMPLES_MAYAPTEXVIEWER_OPENSUBDIVPTEXSHADER_H_
