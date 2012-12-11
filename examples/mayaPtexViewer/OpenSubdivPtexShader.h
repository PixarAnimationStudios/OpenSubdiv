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
