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

#ifndef ARG_UTILS_H
#define ARG_UTILS_H

#include "shape_utils.h"

#include <vector>

class ArgOptions 
{
public:

    ArgOptions();

    // Uses argc and argv to initialize the members of this object.
    void Parse(int argc, char **argv);

    // Prints out unrecognized argument warnings for each argument left in
    // remainingArgs
    void PrintUnrecognizedArgsWarnings() const;

    // Print unrecognized warning for arg
    void PrintUnrecognizedArgWarning(const char *arg) const;


    // Accessors to parsed arguments
    //

    bool GetAdaptive() const { return _adaptive; }

    bool GetFullScreen() const { return _fullscreen; }

    int GetLevel() const { return _level; }

    bool GetObjsAreAnim() const { return _objsAreAnim; }

    bool GetYUp() const { return _yup; }

    int GetRepeatCount() const { return _repeatCount; }
    
    Scheme GetDefaultScheme() const { return _defaultScheme; }

    const std::vector<const char *> GetObjFiles() const { return _objFiles; }

    const std::vector<const char *> GetRemainingArgs() const {
        return _remainingArgs; }


    // Operations on parsed arguments
    //

    size_t AppendObjShapes(std::vector<ShapeDesc>& shapes,
                           bool warn = true) const;

private:

    bool _adaptive;

    bool _fullscreen;

    int _level;

    bool _objsAreAnim;

    bool _yup;

    int _repeatCount;

    Scheme _defaultScheme;

    // .obj files that we've parsed
    std::vector<const char *> _objFiles;

    // Remaining args that we have not parsed, in order that they've appeared
    std::vector<const char *> _remainingArgs;

};

#endif // COMMON_ARGS_H
