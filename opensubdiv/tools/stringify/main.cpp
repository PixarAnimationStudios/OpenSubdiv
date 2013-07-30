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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


std::string stringify( std::string const & line ) {

    bool inconstant=false;

    std::stringstream s;
    for (int i=0; i<(int)line.size(); ++i) {

        // escape double quotes
        if (line[i]=='"') {
            s << '\\' ;
            inconstant = inconstant ? false : true;
        }

        if (line[i]=='\\' and line[i+1]=='\0') {
            s << "\"";
            return s.str();
        }

        // escape backslash
        if (inconstant and line[i]=='\\')
           s << '\\' ;

        s << line[i];
    }

    s << "\\n\"";

    return s.str();
}

int main(int argc, const char **argv) {

    if (argc != 3) {
        std::cerr << "Usage: quoter input-file output-file" << std::endl;
        return 1;
    }

    std::ifstream input;
    input.open(argv[1]);
    if (not input.is_open()) {
        std::cerr << "Can not read from: " << argv[1] << std::endl;
        return 1;
    }

    std::ofstream output;
    output.open(argv[2]);
    if (not output.is_open()) {
        std::cerr << "Can not write to: " << argv[2] << std::endl;
        return 1;
    }

    std::string line;

    while (not input.eof()) {
        std::getline(input, line);
        output << "\"" << stringify(line) << std::endl;
    }

    return 0;
}
