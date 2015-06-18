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


OSD Overview
------------

.. contents::
   :local:
   :backlinks: none

.. image:: images/api_layers_3_0.png
   :width: 100px
   :target: images/api_layers_3_0.png

OpenSubdiv (Osd)
================

**Osd** contains device dependent code that reflects *Far* structure to be
available on various backends such as TBB, CUDA, OpenCL, GLSL etc.
The main roles of **Osd** are:

 - **Refinement**
    Compute stencil-based uniform/adaptive subdivision on CPU/GPU backends
 - **Limit Stencil Evaluation**
    Compute limit surfaces by limit stencils on CPU/GPU backends
 - **Limit Evaluation with PatchTable**
    Compute limit surfaces by patch evaluation on CPU/GPU backends
 - **OpenGL/DX11 Drawing with hardware tessellation**
    Provide GLSL/HLSL tessellation functions for patch table
 - **Interleaved/Batched buffer configuration**
    Provide consistent buffer descriptor to deal with arbitrary buffer layout.
 - **Cross-Platform Implementation**
    Provide convenient classes to interop between compute and draw APIs

They are independently used by client. For example, a client can use only
the stencil table evaluation. A client can call **Osd** compute functions
on its own vertex buffers.

OpenSubdiv enforces the same results for the different computation backends with
a series of regression tests that compare the methods to each other.

----

Refinement
==========

**Osd** supports both `uniform subdivision <subdivision_surfaces.html#uniform-subdivision>`__
and `adaptive subdivision <subdivision_surfaces.html#feature-adaptive-subdivision>`__.


.. image:: images/osd_refinement.png
   :align: center

Once clients create a Far::StencilTable for the topology, then convert it into
device-specific stencil tables if necessary. The following table shows which evaluator
classes and stencil table interfaces can be used together. Note that while **Osd**
provides these stencil tables classes which can be easily constructed from Far::StencilTable,
clients aren't required to use these table classes. Clients may have their own entities
as a stencil table as long as Evaluator::EvalStencils() can access necessary interfaces.

+-----------------------------+-----------------------+-------------------------+
| Backend                     | Evaluator class       | compatible stencil table|
+=============================+=======================+=========================+
| CPU (CPU single-threaded)   | CpuEvaluator          | Far::StencilTable       |
+-----------------------------+-----------------------+-------------------------+
| TBB (CPU multi-threaded)    | TbbEvaluator          | Far::StencilTable       |
+-----------------------------+-----------------------+-------------------------+
| OpenMP (CPU multi-threaded) | OmpEvaluator          | Far::StencilTable       |
+-----------------------------+-----------------------+-------------------------+
| CUDA (GPU)                  | CudaEvaluator         | CudaStencilTable        |
+-----------------------------+-----------------------+-------------------------+
| OpenCL (CPU/GPU)            | CLEvaluator           | CLStencilTable          |
+-----------------------------+-----------------------+-------------------------+
| GL ComputeShader (GPU)      | GLComputeEvaluator    | GLStencilTableSSBO      |
+-----------------------------+-----------------------+-------------------------+
| GL Transform Feedback (GPU) | GLXFBEvaluator        | GLStencilTableTBO       |
+-----------------------------+-----------------------+-------------------------+
| DX11 ComputeShader (GPU)    | D3D11ComputeEvaluator | D3D11StencilTable       |
+-----------------------------+-----------------------+-------------------------+


Limit Stencil Evaluation
========================

Limit stencil evaluation is quite similar to refinement in **Osd**. Clients
create Far::LimitStencilTable for the locations need to evaluate. Then create
an evaluator compatible stencil table and call Evaluator::EvalStencils().

.. image:: images/osd_limitstencil.png
   :align: center

Limit Evaluation with PatchTable
================================

In **Osd**, the limit surfaces can also be evaluated by PatchTable once all
control vertices and local points are resolved by the stencil evaluation.

.. image:: images/osd_limiteval.png
   :align: center

+-----------------------------+-------------------------+-------------------------+
| Backend                     | Evaluator class         | compatible patch   table|
+=============================+=========================+=========================+
| CPU (CPU single-threaded)   | CpuEvaluator            | CpuPatchTable           |
+-----------------------------+-------------------------+-------------------------+
| TBB (CPU multi-threaded)    | TbbEvaluator            | CpuPatchTable           |
+-----------------------------+-------------------------+-------------------------+
| OpenMP (CPU multi-threaded) | OmpEvaluator            | CpuPatchTable           |
+-----------------------------+-------------------------+-------------------------+
| CUDA (GPU)                  | CudaEvaluator           | CudaPatchTable          |
+-----------------------------+-------------------------+-------------------------+
| OpenCL (CPU/GPU)            | CLEvaluator             | CLPatchTable            |
+-----------------------------+-------------------------+-------------------------+
| GL ComputeShader (GPU)      | GLComputeEvaluator      | GLPatchTable            |
+-----------------------------+-------------------------+-------------------------+
| GL Transform Feedback (GPU) | GLXFBEvaluator          | GLPatchTable            |
+-----------------------------+-------------------------+-------------------------+
| DX11 ComputeShader (GPU)    | | D3D11ComputeEvaluator | D3D11PatchTable         |
|                             | | (*)not yet supported  |                         |
+-----------------------------+-------------------------+-------------------------+

.. container:: impnotip

 **Release Notes (3.0.0)**

 * GPU limit evaluation backends (Evaluator::EvalPatches()) only supports
   BSpline patches. Clients need to specify BSpline approximation for endcap
   when creating a patch table. See `end capping <far_overview.html#endcap>`__.

OpenGL/DX11 Drawing with hardware tessellation
==============================================

One of the most interesting use cases of **Osd** layer is realtime drawing of
subdivision surfaces using hardware tessellation. This is somewhat similar to
limit evaluation with PatchTable described above. Drawing differs from limit
evaluation in that **Osd** provides shader snippets for patch evaluation and
clients will inject them into their own shader source.

.. image:: images/osd_draw.png
   :align: center

see `shader interface <osd_shader_interface.html>`__ for more detail of shader interface.

----

Interleaved/Batched buffer configuration
========================================

All **Osd** layer APIs assume that each primitive variables to be computed
(points, colors, uvs ...) are contiguous array of 32bit floating point values.
**Osd** API refers this array as "buffer". Buffer can exist on CPU memory or
GPU memory. **Osd** Evaluators typically take one source buffer and one destination
buffer, or three destination buffers if derivatives are being computed.
**Osd** Evaluators also take BufferDescriptors,
which is used to specify the layout of the source and destination buffers.
BufferDescriptor is 3 integers struct which consists of offset, length and stride.

For example:

 +-----------+-----------+-----------+
 | Vertex 0  |  Vertex 1 | ...       |
 +---+---+---+---+---+---+-----------+
 | X | Y | Z | X | Y | Z | ...       |
 +---+---+---+---+---+---+-----------+

The layout of this buffer can be described as

.. code:: c++

  Osd::BufferDescriptor desc(/*offset = */ 0, /*length = */ 3, /*stride = */ 3);

BufferDescriptor can be used for interleaved buffer too.

 +---------------------------+---------------------------+-------+
 | Vertex 0                  | Vertex 1                  | ...   |
 +---+---+---+---+---+---+---+---+---+---+---+---+---+---+-------+
 | X | Y | Z | R | G | B | A | X | Y | Z | R | G | B | A | ...   |
 +---+---+---+---+---+---+---+---+---+---+---+---+---+---+-------+

.. code:: c++

  Osd::BufferDescriptor xyzDesc(0, 3, 7);
  Osd::BufferDescriptor rgbaDesc(3, 4, 7);

Although the source and the destination buffer don't have to be a same buffer for
EvalStencils(), adaptive patch tables are constructed to index the coarse vertices
first and immediately followed by the refined vertices. In this case, the
BufferDescriptor for the destination should include the offset as the number of coarse
vertices to be skipped.

 +-----------------------------------+-----------------------------------+
 |  Coarse vertices (n) : Src        |  Refined vertices : Dst           |
 +-----------+-----------+-----------+-----------+-----------+-----------+
 | Vertex 0  | Vertex 1  | ...       | Vertex n  | Vertex n+1|           |
 +---+---+---+---+---+---+-----------+---+---+---+---+---+---+-----------+
 | X | Y | Z | X | Y | Z | ...       | X | Y | Z | X | Y | Z | ...       |
 +---+---+---+---+---+---+-----------+---+---+---+---+---+---+-----------+

.. code:: c++

  Osd::BufferDescriptor srcDesc(0, 3, 3);
  Osd::BufferDescriptor dstDesc(n*3, 3, 3);

Also note that the source descriptor doesn't have to start from offset = 0.
This is useful when a client has a big buffer multiple objects batched together.


----

Cross-Platform Implementation
=============================

One of the key goals of OpenSubdiv is to achieve as much cross-platform flexibility
as possible and leverage all optimized hardware paths where available. This can
be very challenging however, as there is a very large variety of plaftorms and
matching APIs available, with very distinct capabilities.

In **Osd**, Evaluators don't care about interops between those APIs. All Evaluators
have two kinds of APIs for both EvalStencils() and EvalPatches().

 - Explicit signatures which directly take device-specific buffer representation
   (i.e. pointer for CpuEvaluator, GLuint buffer for GLComputeEvaluator)
 - Generic signatures which take arbitrary buffer classes. The buffer class
   is required to have a certain method to return the device-specific buffer representation.

The later interface is useful if the client supports multiple backends at the same time.
The methods needs to be implemented for each Evaluators are:

+-----------------------+------------------------+------------------+
| Evaluator class       | object                 | method           |
+=======================+========================+==================+
| | CpuEvaluator        | pointer to cpu memory  | BindCpuBuffer()  |
| | TbbEvaluator        |                        |                  |
| | OmpEvaluator        |                        |                  |
+-----------------------+------------------------+------------------+
| CudaEvaluator         | pointer to cuda memory | BindCudaBuffer() |
+-----------------------+------------------------+------------------+
| CLEvaluator           | cl_mem                 | BindCLBuffer()   |
+-----------------------+------------------------+------------------+
| | GLComputeEvaluator  | GL buffer object       | BindVBO()        |
| | GLXFBEvaluator      |                        |                  |
+-----------------------+------------------------+------------------+
| D3D11ComputeEvaluator | D3D11 UAV              | BindD3D11UAV()   |
+-----------------------+------------------------+------------------+

The buffers can use these methods as a trigger of interop. **Osd** provides default
implementation of interop buffer for the most of combination of backends.
For example, if the client wants to use cuda as computation backend and use OpenGL
as drawing APIs, Osd::CudaGLVertexBuffer fits the case since it implements
BindCudaBuffer() and BindVBO(). Again, clients can implement their own buffer
class and pass it to Evaluators.


