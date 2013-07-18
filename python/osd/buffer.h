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

#pragma once
#include <vector>

namespace shim {

    typedef std::vector<unsigned char> Buffer;

    enum DataType {
        invalid,
        int8,
        uint8,
        int16,
        uint16,
        int32,
        uint32,
        int64,
        uint64,
        int128,
        uint128,
        int256,
        uint256,
        float16,
        float32,
        float64,
        float80,
        float96,
        float128,
    };

    struct HomogeneousBuffer {
        shim::Buffer Buffer;
        shim::DataType Type;
    };

    typedef std::vector<shim::DataType> Layout;

    struct HeterogeneousBuffer {
        shim::Buffer Buffer;
        shim::Layout Layout;
    };
}
