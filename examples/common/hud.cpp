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
#include <string.h>
#include <stdio.h>
#include "hud.h"
#include "font_image.h"

Hud::Hud() : _visible(true), _windowWidth(0), _windowHeight(0), _requiresRebuildStatic(true)
{
}

Hud::~Hud()
{
}

void
Hud::Init(int width, int height)
{
    _windowWidth = width;
    _windowHeight = height;
}

int
Hud::GetWidth() const
{
    return _windowWidth;
}

int
Hud::GetHeight() const
{
    return _windowHeight;
}

void
Hud::Clear()
{
    _radioButtons.clear();
    _checkBoxes.clear();
    _vboSource.clear();
    _requiresRebuildStatic = true;
}

bool
Hud::KeyDown(int key)
{
    for (std::vector<RadioButton>::iterator it = _radioButtons.begin();
        it != _radioButtons.end(); ++it) {

        if (key == it->shortcut) {
            int nextLocalIndex = it->localIndex;
            if (it->sharedShortcut) {
                // find checked radio button in this group
                int maxLocalIndex = 0;
                for (std::vector<RadioButton>::iterator it2 = _radioButtons.begin();
                     it2 != _radioButtons.end(); ++it2) {
                    
                    if (it2->group == it->group) {
                        maxLocalIndex = std::max(maxLocalIndex, it2->localIndex);
                        if (it2->checked) {
                            nextLocalIndex = it2->localIndex+1;
                        }
                    }
                }
                if(nextLocalIndex > maxLocalIndex) nextLocalIndex = 0;
            }
            for (std::vector<RadioButton>::iterator it2 = _radioButtons.begin();
                 it2 != _radioButtons.end(); ++it2) {
                if (it2->group == it->group) {
                    if (it2->localIndex == nextLocalIndex) {
                        it2->checked = true;
                        it2->callback(it2->callbackData);
                    } else {
                        it2->checked = false;
                    }
                }
            }
            _requiresRebuildStatic = true;
            return true;
        }
    }
    for (std::vector<CheckBox>::iterator it = _checkBoxes.begin();
         it != _checkBoxes.end(); ++it) {
        
        if (key == it->shortcut) {
            it->checked = !it->checked;
            it->callback(it->checked, it->callbackData);
            _requiresRebuildStatic = true;
            return true;
        }
    }
          
    return false;
}

bool
Hud::MouseClick(int x, int y)
{
    for (std::vector<RadioButton>::iterator it = _radioButtons.begin();
        it != _radioButtons.end(); ++it) {

        int bx = it->x > 0 ? it->x : _windowWidth + it->x;
        int by = it->y > 0 ? it->y : _windowHeight + it->y;

        if (x >= bx && y >= by &&
            x <= (bx + it->w) && y <= (by + it->h)) {
            for (std::vector<RadioButton>::iterator it2 = _radioButtons.begin();
                 it2 != _radioButtons.end(); ++it2) {

                if (it2->group == it->group && it != it2) it2->checked = false;
            }
            it->checked = true;
            it->callback(it->callbackData);
            _requiresRebuildStatic = true;
            return true;
        }
    }
    for (std::vector<CheckBox>::iterator it = _checkBoxes.begin();
        it != _checkBoxes.end(); ++it) {

        int bx = it->x > 0 ? it->x : _windowWidth + it->x;
        int by = it->y > 0 ? it->y : _windowHeight + it->y;

        if (x >= bx && y >= by &&
            x <= (bx + it->w) && y <= (by + it->h)) {
            it->checked = !it->checked;
            it->callback(it->checked, it->callbackData);
            _requiresRebuildStatic = true;
            return true;
        }
    }
    return false;
}

void
Hud::AddCheckBox(const char *label, bool checked, int x, int y,
                   CheckBoxCallback callback, int data, int shortcut)
{
    CheckBox cb;
    cb.label = label;
    cb.checked = checked;
    cb.x = x;
    cb.y = y;
    cb.w = (int)(strlen(label)+1) * FONT_CHAR_WIDTH;
    cb.h = FONT_CHAR_HEIGHT;
    cb.callback = callback;
    cb.callbackData = data;
    cb.shortcut = shortcut;

    _checkBoxes.push_back(cb);
    _requiresRebuildStatic = true;
}

void
Hud::AddRadioButton(int group, const char *label, bool checked, int x, int y,
                      RadioButtonCallback callback, int data, int shortcut)
{
    RadioButton rb;
    rb.group = group;
    rb.label = label;
    rb.checked = checked;
    rb.x = x;
    rb.y = y;
    rb.w = (int)(strlen(label)+1) * FONT_CHAR_WIDTH;
    rb.h = FONT_CHAR_HEIGHT;
    rb.callback = callback;
    rb.callbackData = data;
    rb.shortcut = shortcut;
    rb.sharedShortcut = false;
    rb.localIndex = 0;

    for (std::vector<RadioButton>::iterator it = _radioButtons.begin();
         it != _radioButtons.end(); ++it) {
        if (it->group == group) {
            rb.localIndex = it->localIndex+1;
            if (it->shortcut == shortcut) {
                it->sharedShortcut = true;
                rb.sharedShortcut = true;
            }
        }
    }

    _radioButtons.push_back(rb);
    _requiresRebuildStatic = true;
}

int
Hud::drawChar(std::vector<float> &vboSource, int x, int y, float r, float g, float b, char ch) const
{
    const float w = 1.0f/FONT_TEXTURE_COLUMNS;
    const float h = 1.0f/FONT_TEXTURE_ROWS;
        
    float u = (ch%FONT_TEXTURE_COLUMNS)/(float)FONT_TEXTURE_COLUMNS;
    float v = (ch/FONT_TEXTURE_COLUMNS)/(float)FONT_TEXTURE_ROWS;
        
    vboSource.push_back(float(x)); vboSource.push_back(float(y));
    vboSource.push_back(r); vboSource.push_back(g); vboSource.push_back(b);
    vboSource.push_back(u); vboSource.push_back(v);
    
    vboSource.push_back(float(x)); vboSource.push_back(float(y+FONT_CHAR_HEIGHT));
    vboSource.push_back(r); vboSource.push_back(g); vboSource.push_back(b);
    vboSource.push_back(u); vboSource.push_back(v+h);
    
    vboSource.push_back(float(x+FONT_CHAR_WIDTH)); vboSource.push_back(float(y));
    vboSource.push_back(r);                 vboSource.push_back(g); vboSource.push_back(b);
    vboSource.push_back(u+w);               vboSource.push_back(v);

    vboSource.push_back(float(x+FONT_CHAR_WIDTH)); vboSource.push_back(float(y));
    vboSource.push_back(r);                 vboSource.push_back(g); vboSource.push_back(b);
    vboSource.push_back(u+w);               vboSource.push_back(v);

    vboSource.push_back(float(x)); vboSource.push_back(float(y+FONT_CHAR_HEIGHT));
    vboSource.push_back(r); vboSource.push_back(g); vboSource.push_back(b);
    vboSource.push_back(u); vboSource.push_back(v+h);
    
    vboSource.push_back(float(x+FONT_CHAR_WIDTH)); vboSource.push_back(float(y+FONT_CHAR_HEIGHT));
    vboSource.push_back(r);                 vboSource.push_back(g); vboSource.push_back(b);
    vboSource.push_back(u+w);               vboSource.push_back(v+h);
    
    return x + FONT_CHAR_WIDTH;
}

int
Hud::drawString(std::vector<float> &vboSource, int x, int y, float r, float g, float b, const char *c) const
{
    while(*c) {
        char ch = (*c) & 0x7f;
        x = drawChar(vboSource, x, y, r, g, b, ch);
        c++;
    }
    return x;
}

void
Hud::DrawString(int x, int y, const char *fmt, ...)
{
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    x = x > 0 ? x : _windowWidth + x;
    y = y > 0 ? y : _windowHeight + y;

    drawString(_vboSource, x, y, 1, 1, 1, buf);
}

void
Hud::DrawString(int x, int y, float r, float g, float b, const char *fmt, ...)
{
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    x = x > 0 ? x : _windowWidth + x;
    y = y > 0 ? y : _windowHeight + y;

    drawString(_vboSource, x, y, r, g, b, buf);
}

void
Hud::Rebuild(int width, int height)
{
    _requiresRebuildStatic = false;
    _windowWidth = width;
    _windowHeight = height;
}

bool
Hud::Flush()
{
    if (!_visible) {
        _vboSource.clear();
        return false;
    }

    if (_requiresRebuildStatic)
        Rebuild(_windowWidth, _windowHeight);

    return true;
}

bool
Hud::IsVisible() const
{
    return _visible;
}

void
Hud::SetVisible(bool visible)
{
    _visible = visible;
}

const std::vector<Hud::RadioButton> & 
Hud::getRadioButtons() const
{
    return _radioButtons;
}
const std::vector<Hud::CheckBox> &
Hud::getCheckBoxes() const
{
    return _checkBoxes;
}

std::vector<float> &
Hud::getVboSource()
{
    return _vboSource;
}
