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
from html.parser import HTMLParser

class HtmlToTextParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.m_text = []
        self.m_inTitle = False
        self.m_inScript = False
        self.m_inStyle = False
        self.m_title = ""
        self.m_navigation = False

    def handle_data(self, data):
        if self.m_inScript or self.m_inStyle:
            return
        text = data.strip()
        if len(text) > 0:
            text = re.sub('[\s]+', ' ', text)
            text = re.sub('[^\.,\- a-zA-Z0-9_]+', '', text)
            self.m_text.append(text + ' ')
        if self.m_inTitle:
            self.m_title = str(text)

    def handle_endtag(self, tag):
        if tag.lower() == "title": self.m_inTitle = False
        if tag.lower() == "script": self.m_inScript = False
        if tag.lower() == "style": self.m_inStyle = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title": self.m_inTitle = True
        if tag.lower() == "script": self.m_inScript = True
        if tag.lower() == "style": self.m_inStyle = True
        if tag.lower() == "div":
            for attr in attrs:
                if (len(attr)>=2 and \
                    attr[0].lower()=="class" and \
                    attr[1].lower()=="navigation"):
                    self.m_navigation = True

    def HasNavigationSection(self):
        return self.m_navigation

    def GetText(self):
        return ''.join(self.m_text).strip()

    def GetTitle(self):
        return self.m_title


#-------------------------------------------------------------------------------
def StripHTMLComments(data):
    regex = re.compile('\<![ \r\n\t]*(--([^\-]|[\r\n]|-[^\-])*--[ \r\n\t]*)\>')
    return regex.sub('',data)

#-------------------------------------------------------------------------------
def ReadNavigationTemplate( filePath ):

    navHtml = ""

    try:
        navFile = open( filePath, "r")
    except IOError:
        print("Could not open file \'"+filePath+"\'")

    with navFile:
        print("Navigation template: \'"+filePath+"\'")
        navHtml = navFile.read()
        navHtml = StripHTMLComments(navHtml)
        navFile.close()
        navHtml = StripHTMLComments(navHtml)

    return navHtml

#-------------------------------------------------------------------------------
def WriteIndexFile( outputFile, content ):
    outputPath = os.path.dirname( outputFile )

    try:
        os.makedirs( outputPath )
    except:
        pass

    print("Creating Search-Index File : \""+outputFile+"\"")

    f = open(outputFile, "w")
    f.write(content)
    f.close()

#-------------------------------------------------------------------------------
def Usage():
    print(str(sys.argv[0])+" <input directory> <output directory> <html template>")
    exit(1)


#-------------------------------------------------------------------------------
# Main
if (len(sys.argv)<3):
    Usage()

rootDir = str(sys.argv[1])

navTemplate = str(sys.argv[2])

navHtml = ReadNavigationTemplate( navTemplate )

print("Scanning : \'"+rootDir+"\'")

searchIndex = 'var tipuesearch = { "pages": [ '

# recursively scan sub-directories for HTML files
for root, dirs, files in os.walk(rootDir):

    # skip doxygen generated HTML
    if 'doxy_html' in dirs:
        dirs.remove('doxy_html')

    for f in files:

        inputFile = os.path.join(root, f)
        if inputFile.endswith(".html") or inputFile.endswith(".htm") :

            f = open(inputFile, "r+")
            html = f.read()

            # parse the ReST generated HTML
            parser = HtmlToTextParser()
            try:
                parser.feed(html)
                title = parser.GetTitle()
                text = parser.GetText()
            except HTMLParser.error:
                continue

            msg = "    \""+inputFile+"\" - "

            # index the contents of the page for search
            if (not inputFile.lower().endswith("search.html")):
                if title == "":
                    title = "untitled"
                loc = os.path.relpath(inputFile, rootDir)
                searchIndex += '{"title":"'+title+'", "text":"'+text+'", "tags": "", "loc":"'+loc+'"}, \n'
                msg += "indexed - "

            # if necessary, insert navigation html
            if (not parser.HasNavigationSection()):
                loc = html.find("<body>")
                html = html[:loc+6] + navHtml + html[loc+6:]

                msg += "added navigation"

            # replace the article title placeholder with the real title
            if title:
                html = html.replace("OSD_ARTICLE_TITLE", title)
            else:
                html = html.replace("OSD_ARTICLE_TITLE", "")

            f.seek(0)
            f.write(html)
            f.close()

            print(msg)

searchIndex = searchIndex + "]};"

WriteIndexFile( os.path.join(rootDir, "tipuesearch", "tipuesearch_content.js"), searchIndex )

