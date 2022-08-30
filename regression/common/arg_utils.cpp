//
//   Copyright 2019 Pixar
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

#include "arg_utils.h"

#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
parseIntArg(const char* argString, int dfltValue = 0) {
    char *argEndptr;
    int argValue = (int) strtol(argString, &argEndptr, 10);
    if (*argEndptr != 0) {
        printf("Warning: non-integer option parameter '%s' ignored\n", 
			   argString);
        argValue = dfltValue;
    }
    return argValue;
}


ArgOptions::ArgOptions() 
    : _adaptive(true)
    , _fullscreen(false)
    , _level(2)
    , _objsAreAnim(false)
    , _yup(false)
    , _repeatCount(0)
    , _defaultScheme(kCatmark)
{
}

void
ArgOptions::Parse(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {

        if (strstr(argv[i], ".obj")) {
            _objFiles.push_back(argv[i]);
        } else if (!strcmp(argv[i], "-a")) {
            _adaptive = true;
        } else if (!strcmp(argv[i], "-u")) {
            _adaptive = false;
        } else if (!strcmp(argv[i], "-l")) {
            if (++i < argc) _level = parseIntArg(argv[i], 2);
        } else if (!strcmp(argv[i], "-c")) {
            if (++i < argc) _repeatCount = parseIntArg(argv[i], 0);
        } else if (!strcmp(argv[i], "-f")) {
            _fullscreen = true;
        } else if (!strcmp(argv[i], "-yup")) {
            _yup = true;
        } else if (!strcmp(argv[i], "-anim")) {
            _objsAreAnim = true;
        } else if (!strcmp(argv[i], "-bilinear")) {
            _defaultScheme = kBilinear;
        } else if (!strcmp(argv[i], "-catmark")) {
            _defaultScheme = kCatmark;
        } else if (!strcmp(argv[i], "-loop")) {
            _defaultScheme = kLoop;
        } else {
            _remainingArgs.push_back(argv[i]);
        }

    }
}

void 
ArgOptions::PrintUnrecognizedArgWarning(const char *arg) const
{
    printf("Warning: unrecognized argument '%s' ignored\n", arg);
}

void 
ArgOptions::PrintUnrecognizedArgsWarnings() const
{
    for(size_t i = 0; i < _remainingArgs.size(); ++i) {
        PrintUnrecognizedArgWarning(_remainingArgs[i]);
    }
}

size_t
ArgOptions::AppendObjShapes(std::vector<ShapeDesc>& shapes, bool warn) const
{
    size_t originalShapesSize = shapes.size();

    for (size_t i = 0; i < GetObjFiles().size(); ++i) {
        std::ifstream ifs(GetObjFiles()[i]);
        if (ifs) {
            std::stringstream ss;
            ss << ifs.rdbuf();
            ifs.close();
            std::string str = ss.str();
            shapes.push_back(ShapeDesc(
                        GetObjFiles()[i], str.c_str(),
                        GetDefaultScheme()));
        } else if (warn) {
            printf("Warning: cannot open shape file '%s'\n",
                   GetObjFiles()[i]);
        }
    }
    return shapes.size() - originalShapesSize;
}
