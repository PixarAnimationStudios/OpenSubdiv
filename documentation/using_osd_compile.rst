..  
     Copyright 2013 Pixar
  
     Licensed under the Apache License, Version 2.0 (the "Apache License")
     with the following modification; you may not use this file except in
     compliance with the Apache License and the following modification to it:
     Section 6. Trademarks. is deleted and replaced with:
  
     6. Trademarks. This License does not grant permission to use the trade
        names, trademarks, service marks, or product names of the Licensor
        and its affiliates, except as required to comply with Section 4(c) of
        the License and to reproduce the content of the NOTICE file.
  
     You may obtain a copy of the Apache License at
  
         http://www.apache.org/licenses/LICENSE-2.0
  
     Unless required by applicable law or agreed to in writing, software
     distributed under the Apache License with the above modification is
     distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
     KIND, either express or implied. See the Apache License for the specific
     language governing permissions and limitations under the Apache License.
  

Using OpenSubdiv
----------------

.. contents::
   :local:
   :backlinks: none


Compiling & Linking
===================

Here are example commands for building an OpenSubdiv application on several architectures:

**Linux**
:: 
  
  g++ -I$OPENSUBDIV/include -c myapp.cpp
  g++ myapp.o -L$OPENSUBDIV/lib -losdCPU -losdGPU -o myapp

**Mac OS-X**   
::
  
  g++ -I$OPENSUBDIV/include -c myapp.cpp
  g++ myapp.o -L$OPENSUBDIV/lib -losdCPU -losdGPU -o myapp
  install_name_tool -add_rpath $OPENSUBDIV/lib myapp

(On 64-bit OS-X: add ``-m64`` after each ``g++``.)

**Windows**
::
  
  cl /nologo /MT /TP /DWIN32 /I"%OPENSUBDIV%\include" -c myapp.cpp
  link /nologo /out:myapp.exe /LIBPATH:"%OPENSUBDIV%\lib" libosdCPU.lib libosdGPU.lib myapp.obj 


.. container:: impnotip

    **Note:**
    
    HBR uses the offsetof macro on a templated struct, which appears to spurriously set off a 
    warning in both gcc and Clang. It is recommended to turn the warning off with the
    *-Wno-invalid-offsetof* flag.
