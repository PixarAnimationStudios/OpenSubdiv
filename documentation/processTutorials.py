#!/usr/bin/env python
#
#   Copyright 2013 Pixar
#
#   Licensed under the Apache License, Version 2.0 (the "Apache License")
#   with the following modification; you may not use this file except in
#   compliance with the Apache License and the following modification to it:
#   Section 6. Trademarks. is deleted and replaced with:
#
#   6. Trademarks. This License does not grant permission to use the trade
#      names, trademarks, service marks, or product names of the Licensor
#      and its affiliates, except as required to comply with Section 4(c) of
#      the License and to reproduce the content of the NOTICE file.
#
#   You may obtain a copy of the Apache License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the Apache License with the above modification is
#   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied. See the Apache License for the specific
#   language governing permissions and limitations under the Apache License.
#
from __future__ import print_function

import os
import sys
import string
import re

#-------------------------------------------------------------------------------
def ReadFile(inputfile):
    try:
        f = open( inputfile, "r")
    except IOError:
        print("Could not read file \'"+inputfile+"\'")
    content = f.read()
    f.close()
    return content

#-------------------------------------------------------------------------------
def WriteToFile(outputfile, content):
    outputPath = os.path.dirname(outputfile)
    try:
        os.makedirs(outputPath)
    except:
        pass
    try:
        f = open(outputfile, "w")
    except IOError:
        print("Could not write file \'"+outputfile+"\'")
    f.write(content)
    f.close()

#-------------------------------------------------------------------------------
# Reformats the C++ tutorial file as a ReST file for publication on the
# documentation site
def Process(srcfile, title):

    basename = os.path.basename(srcfile)

    rest = "\n"

    rest += basename+"\n"
    rest += ("-" * len(basename))+"\n\n"

    rest += "`<https://github.com/PixarAnimationStudios/OpenSubdiv/blob/release/tutorials/"+title+">`_\n"

    rest += ("\n"
             "----\n"
             "\n"
             ".. code:: c++\n"
             "\n")

    code = ReadFile(srcfile)

    lines = code.split('\n')
    # cut license header (line 24)
    for line in lines[24:]:
        rest += '    ' + line + '\n'

    #lines = [i.start() for i in re.finditer("\n", code)]
    #rest += code[lines[25]:]

    return rest

#-------------------------------------------------------------------------------
def Usage():
    print(str(sys.argv[0])+" <input file> <output file> <title>")
    exit(1)


#-------------------------------------------------------------------------------
# Main

# XXXX manuelk we should have this script traverse the tutorials folders and
#              automatically generate both the ReST from the C++ code as well
#              as the tutorials.rst file based on what is found
if (len(sys.argv)!=4):
    Usage()

srcfile = str(sys.argv[1])
title = str(sys.argv[3])
rest = Process(srcfile, title)

dstfile = str(sys.argv[2])
WriteToFile(dstfile, rest)
