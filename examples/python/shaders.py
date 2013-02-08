#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
#

from OpenGL.GL import *

ProgramFiles = ['simple.glsl']

Programs = {
    "BareBones" : {
        "GL_VERTEX_SHADER" : "simple.xforms simple.vert",
        "GL_GEOMETRY_SHADER" : "simple.xforms simple.geom",
        "GL_FRAGMENT_SHADER" : "simple.xforms simple.frag"
    }
}

class Attribs:
    POSITION = 0
    NORMAL = 1

def load_shaders(
    glslFiles = ProgramFiles,
    attribs = Attribs,
    programMap = Programs):
    """Parse a series of simple text files, each of which is composed
    series of shader snippets. The given 'attribs' class defines an
    enumeration of attribute slots to bind (aka semantics).
        
    In each text file, lines starting with two dash characters start a new
    shader snippet with the given name.  For example:
       -- S1
       uniform float foo;
       -- Prefix
       #version 150
       -- S2
       uniform float bar;
    """

    import os.path

    # First parse the file to populate 'snippetMap'
    class ParserState:
        HEADER = 0
        SOURCE = 1
    parserState = ParserState.HEADER
    snippetMap = {}
    for glslFile in glslFiles:
        basename = os.path.basename(glslFile)
        snippetPrefix = os.path.splitext(basename)[0] + '.'
        for line in open(glslFile):
            if line.startswith('--'):
                if parserState is ParserState.HEADER:
                    parserState = ParserState.SOURCE
                elif len(source):
                    snippetMap[snippetName] = ''.join(source)
                snippetName = snippetPrefix + line[2:].strip()
                source = []
                continue
            if parserState is ParserState.HEADER:
                pass
            else:
                source.append(line)
    if len(source):
        snippetMap[snippetName] = ''.join(source)

    # Now, glue together the strings and feed them to OpenGL
    stagePrefix = "#version 150\n"
    programs = {}
    for key, val in programMap.items():
        programHandle = glCreateProgram()
        for stageName, snippetList in val.items():
            snippets = map(snippetMap.get, snippetList.split())
            stageSource = stagePrefix + ''.join(snippets)
            stage = getattr(OpenGL.GL, stageName)
            sHandle = glCreateShader(stage)
            glShaderSource(sHandle, stageSource)
            glCompileShader(sHandle)
            success = glGetShaderiv(sHandle, GL_COMPILE_STATUS)
            if not success:
                print 'Error in', stageName, snippetList
                infolog = glGetShaderInfoLog(sHandle)
                raise SystemExit(infolog)
            glAttachShader(programHandle, sHandle)
        for attrib in dir(attribs):
            if attrib.startswith('__'):
                continue
            slot = getattr(attribs, attrib)
            name = attrib[0] + attrib[1:].lower()
            glBindAttribLocation(programHandle, slot, name)
        glLinkProgram(programHandle)
        success = glGetProgramiv(programHandle, GL_LINK_STATUS);
        if not success:
            infolog = glGetProgramInfoLog(programHandle)
            raise SystemExit(infolog)
        programs[key] = programHandle
    return programs
