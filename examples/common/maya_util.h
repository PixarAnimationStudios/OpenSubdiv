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
#ifndef _EXAMPLE_MAYA_UTIL_H_
#define _EXAMPLE_MAYA_UTIL_H_

#include <maya/MColor.h>
#include <maya/MFloatVector.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnNumericData.h>
#include <maya/MMatrix.h>
#include <maya/MPlug.h>
#include <maya/MString.h>
#include <maya/MVector.h>


#define CHECK_GL_ERROR(...)  \
    if (GLuint err = glGetError()) {                \
        fprintf(stderr, "GL error %x :", err);      \
        fprintf(stderr, "%s", __VA_ARGS__);         \
    }


#define MERROR(status, msg)                                                                         \
    {                                                                                               \
        MGlobal::displayError(MString(msg));                                                        \
        fprintf(stderr, "%s [ %s:%d ]\n", msg, __FILE__, __LINE__);  \
    }                               

#define MCHECK_PRINT(status, msg)                                                                   \
    if (status.error())                                                                             \
    {                                                                                               \
        MGlobal::displayError(MString(msg));                                                        \
        fprintf(stderr, "%s [ %s:%d ]\n", msg, __FILE__, __LINE__);  \
    }                               

#define MCHECK_RETURN(status, msg)                                                                  \
    if (status.error())                                                                             \
    {                                                                                               \
        MGlobal::displayError(MString(msg));                                                        \
        fprintf(stderr, "%s [ %s:%d ]\n", msg, __FILE__, __LINE__);  \
        return status;                                                                              \
    }                               


// 
//      Templated funtions able to retrieve any attribute that
//      can be retrieved via the overloaded MPlug::getValue()
//      methods.  Any attributes that need MPlug::getData()
//      Need to use non-templated functions below
//
template<class T> static int
findAttribute( MFnDependencyNode &depFn, const char *attr, T *val )
{
    MStatus stat;
    MPlug plug;
    T tmp;

    // careful version - returns -1 if attribute missing
    plug = depFn.findPlug(attr, &stat);
    if (stat != MS::kSuccess) return -1;

    stat = plug.getValue(tmp);
    if ( stat != MS::kSuccess ) return -1;

    *val = tmp;
    return 0;
}


template<class T> static void
getAttribute( MObject& object, MObject& attr, T *val )
{
    // fast version - crash & burn if attribute missing
    MPlug plug(object, attr);
    plug.getValue(*val);
}


static MColor
getColor(MObject object, MObject attr) 
{
    MPlug plug(object, attr);
    MObject data;
    plug.getValue(data);
    MFnNumericData numFn(data);
    float color[3];
    numFn.getData(color[0], color[1], color[2]);
    return MColor(color[0], color[1], color[2]);
}

static MFloatVector
getVector(MObject object, MObject attr) 
{
    MPlug plug(object, attr);
    MObject data;
    plug.getValue(data);
    MFnNumericData numFn(data);
    float color[3];
    numFn.getData(color[0], color[1], color[2]);
    return MVector(color[0], color[1], color[2]);
}


static void
// reverse to dst,src ?
setMatrix(const MMatrix &mat, float *dst) 
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            dst[i*4+j] = float(mat(i, j));
}


#endif // _EXAMPLE_MAYA_UTIL_H_

