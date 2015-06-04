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

Downloads
---------

.. contents::
   :local:
   :backlinks: none

Coupe Car
=========


Note : we can use javascripts buttons to trigger download (see button-download) or
       we can just paste regular ReST hyperlinks to the files here. Direct links
       probably more reliable though.

.. raw:: html

    <style>
        .button-download {
        }
    </style>
    <script>
        <!-- 
            note : link.click() doesn't seem to be working in all browsers ->
                   maybe just having a direct link to the file is sufficient
        -->
        function download(file, name) {
            var link = document.createElement("a");
            link.download = name;
            link.href = file;
            link.click();
        }
    </script>

    <p>Coupe car (maya) <button class="button-download" onmousedown="download('/home/mkraemer/Coupe.ma','Coupe.ma')">Download</button></p>
    <p>Coupe car (obj) <button class="button-download" onmousedown="download('file:///home/mkraemer/coupe.2.obj','Coupe.obj')">Download</button></p>

.. raw:: html

    <style>
        div#overlay {
                display: block;
                z-index: 2;
                background: #000;
                position: fixed;
                width: 100%;
                height: 100%;
                top: 0px;
                left: 0px;
                opacity: .8;
                text-align: center;
        }
        div#specialBox {
                display: block;
                position: fixed;
                left: 50%;
                top: 0%;
                transform: translate(-50%, -15%);
                z-index: 3;
                border-radius:10px;
                -moz-border-radius:10px;
                -webkit-border-radius:10px;
                margin: 150px auto 0px auto;
                width: 800px;
                height: 750px;
                background: #efefef;
                color: #333;
        }
        .button-accept {
            position: absolute;
            bottom: 5%;
            left: 45%;
        }
    </style>
    <script>
        function closeOverlay(){
            var overlay = document.getElementById('overlay');
            var specialBox = document.getElementById('specialBox');
            overlay.style.display = "none";
            specialBox.style.display = "none";
        }
    </script>

    <div id="overlay"></div>

    <div id="specialBox">
      <h2 class="articleContentTitle">EULA</h2>
      <div class="simple">
        <p>OpenSubdiv is covered by a modified Apache 2.0 license (included below), and is
        free to use for commercial or non-commercial use. All Pixar patents in the
        area of subdivision surface algorithms have also been released for public use.
        We welcome any involvement in the development or extension of this code; in
        fact, we would love it. Please contact us if you are interested.</p>
        <pre class="code literal-block">
          Copyright 2013 Pixar

          Licensed under the Apache License, Version 2.0 (the &quot;Apache License&quot;)
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
          distributed on an &quot;AS IS&quot; BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
          KIND, either express or implied. See the Apache License for the specific
          language governing permissions and limitations under the Apache License.
        </pre>
      </div>
      <p align="center">
          <button class="button-accept" onmousedown="closeOverlay()">Accept</button>
      </p>
    </div>

