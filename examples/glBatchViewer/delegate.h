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

#ifndef DELEGATE_H
#define DELEGATE_H

#include <osd/cpuGLVertexBuffer.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>
#include <osdutil/batch.h>

#include "effect.h"
#include "effectRegistry.h"

class MyDrawContext : public OpenSubdiv::OsdGLDrawContext {
public:
    virtual ~MyDrawContext();

    static MyDrawContext *Create(OpenSubdiv::FarPatchTables const *patchTables,
                                 bool requireFVarData=false);

    GLuint GetVertexArray() const { return _vao; }

private:
    MyDrawContext();

    GLuint _vao;
};

class MyDrawDelegate {
public:
    typedef MyEffect * EffectHandle;

    void Bind(OpenSubdiv::OsdUtilMeshBatchBase<MyDrawContext> *batch, EffectHandle const &effect);
    void Unbind(OpenSubdiv::OsdUtilMeshBatchBase<MyDrawContext> *batch, EffectHandle const &effect);

    void Begin();
    void End();

    void DrawElements(OpenSubdiv::OsdDrawContext::PatchArray const &patchArray);

    bool IsCombinable(EffectHandle const &a, EffectHandle const &b) const;

    void ResetNumDrawCalls() { _numDrawCalls = 0; }
    int GetNumDrawCalls() const { return _numDrawCalls; }

private:
    MyDrawConfig *GetDrawConfig(EffectHandle &effect, OpenSubdiv::OsdDrawContext::PatchDescriptor desc);

    MyEffectRegistry _effectRegistry;
    int _numDrawCalls;
    OpenSubdiv::OsdUtilMeshBatchBase<MyDrawContext> *_currentBatch;
    EffectHandle _currentEffect;
};

#endif  /* DELEGATE_H */
