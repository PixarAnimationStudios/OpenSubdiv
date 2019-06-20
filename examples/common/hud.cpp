//
//   Copyright 2013 Pixar
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

#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cassert>
#include "hud.h"
#include "font_image.h"

#if _MSC_VER
#define snprintf _snprintf
#endif

Hud::Hud() : _visible(true), _windowWidth(0), _windowHeight(0),
             _framebufferWidth(0), _framebufferHeight(0),
             _requiresRebuildStatic(true)
{
    _capturedSlider = -1;
}

Hud::~Hud()
{
}

void
Hud::Init(int width, int height, int framebufferWidth, int framebufferHeight)
{
    _windowWidth = width;
    _windowHeight = height;
    _framebufferWidth = framebufferWidth;
    _framebufferHeight = framebufferHeight;
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
    _sliders.clear();
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
                if (nextLocalIndex > maxLocalIndex) nextLocalIndex = 0;
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
    for (std::vector<PullDown>::iterator it = _pulldowns.begin();
         it != _pulldowns.end(); ++it) {
        if (key==it->shortcut) {
            // cycle through selections
            ++it->selected;
            if (it->selected>=(int)it->labels.size()) {
                 it->selected=0;
            }
            it->callback(it->values[it->selected]);
            _requiresRebuildStatic = true;
            return true;
        }
    }

    return false;
}

bool
Hud::MouseClick(int x, int y)
{
    if (!IsVisible()) {
        return false;
    }

    for (std::vector<RadioButton>::iterator it = _radioButtons.begin();
        it != _radioButtons.end(); ++it) {
        if (hitTest(*it, x, y)) {
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
        if (hitTest(*it, x, y)) {
            it->checked = !it->checked;
            it->callback(it->checked, it->callbackData);
            _requiresRebuildStatic = true;
            return true;
        }
    }
    for (std::vector<Slider>::iterator it = _sliders.begin();
         it != _sliders.end(); ++it) {
        if (hitTest(*it, x, y)) {
            int bx = 0, by = 0;
            getWindowPos(*it, &bx, &by);
            it->SetValue(((x-bx-FONT_CHAR_WIDTH/2)/float(it->w))*(it->max - it->min) + it->min);
            it->callback(it->value, it->callbackData);
            _capturedSlider = (int)(it - _sliders.begin());
            _requiresRebuildStatic = true;
            return true;
        }
    }
    for (std::vector<PullDown>::iterator it = _pulldowns.begin();
         it != _pulldowns.end(); ++it) {
        if (hitTest(*it, x, y)) {
            if (! it->open) {
                it->h = FONT_CHAR_HEIGHT;
                it->h *= (int)it->labels.size();
                it->open=true;
            } else {
                int label_width = (3+(int)it->label.size()) * FONT_CHAR_WIDTH;
                int bx = 0, by = 0;
                getWindowPos(*it, &bx, &by);
                if (x > (bx+label_width)) {
                    int sel = it->selected;
                    it->SetSelected((y-by)/FONT_CHAR_HEIGHT);
                    if (it->selected!=sel) {
                        it->callback(it->values[it->selected]);
                    }
                } else {
                    it->open=! it->open;
                }
            }
            _requiresRebuildStatic = true;
            return true;
        }
    }
    return false;
}

void
Hud::MouseMotion(int x, int /* y */)
{
    if (_capturedSlider != -1) {
        std::vector<Slider>::iterator it = _sliders.begin() + _capturedSlider;

        int bx = it->x > 0 ? it->x : _windowWidth + it->x;
        it->SetValue(((x-bx-FONT_CHAR_WIDTH/2)/float(it->w))*(it->max - it->min) + it->min);
        it->callback(it->value, it->callbackData);
        _requiresRebuildStatic = true;
    }
}

bool
Hud::MouseCapture() const
{
    return _capturedSlider != -1;
}

void
Hud::MouseRelease()
{
    _capturedSlider = -1;
}

void
Hud::AddLabel(const char *label, int x, int y)
{
    Item l;
    l.label = label;
    l.x = x;
    l.y = y;

    _labels.push_back(l);
    _requiresRebuildStatic = true;
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

void
Hud::AddSlider(const char *label, float min, float max, float value,
               int x, int y, int width, bool intStep, SliderCallback callback, int data)
{
    Slider slider;
    slider.label = label;
    slider.x = x;
    slider.y = y;
    slider.w = width * FONT_CHAR_WIDTH;
    slider.h = FONT_CHAR_HEIGHT * 2;
    slider.min = min;
    slider.max = max;
    slider.value = value;
    slider.callback = callback;
    slider.callbackData = data;
    slider.intStep = intStep;

    _sliders.push_back(slider);
    _requiresRebuildStatic = true;
}

int
Hud::AddPullDown(const char *label, int x, int y, int width,
                 PullDownCallback callback, int shortcut)
{

    PullDown pd;
    pd.label = label;
    pd.x = x;
    pd.y = y;
    pd.w = width;
    pd.h =  FONT_CHAR_HEIGHT;
    pd.open = false;
    pd.selected = 0;
    pd.callback = callback;
    pd.shortcut = shortcut;

    _pulldowns.push_back(pd);
    _requiresRebuildStatic = true;

    return (int)_pulldowns.size()-1;
}

void
Hud::AddPullDownButton(int handle, const char *label, int value, bool checked)
{
    if (handle < (int)_pulldowns.size()) {

        PullDown & pulldown = _pulldowns[handle];

        pulldown.labels.push_back(label);
        pulldown.values.push_back(value);
        if (checked) {
            pulldown.selected = (int)pulldown.labels.size()-1;
        }
    }
}



int
Hud::drawChar(std::vector<float> &vboSource,
              int x, int y, float r, float g, float b, char ch)
{
    const float w = 1.0f/FONT_TEXTURE_COLUMNS;
    const float h = 1.0f/FONT_TEXTURE_ROWS;

    float u = (ch%FONT_TEXTURE_COLUMNS)/(float)FONT_TEXTURE_COLUMNS;
    float v = (ch/FONT_TEXTURE_COLUMNS)/(float)FONT_TEXTURE_ROWS;

    vboSource.push_back(float(x));
    vboSource.push_back(float(y));
    vboSource.push_back(r);
    vboSource.push_back(g);
    vboSource.push_back(b);
    vboSource.push_back(u);
    vboSource.push_back(v);

    vboSource.push_back(float(x));
    vboSource.push_back(float(y+FONT_CHAR_HEIGHT));
    vboSource.push_back(r);
    vboSource.push_back(g);
    vboSource.push_back(b);
    vboSource.push_back(u);
    vboSource.push_back(v+h);

    vboSource.push_back(float(x+FONT_CHAR_WIDTH));
    vboSource.push_back(float(y));
    vboSource.push_back(r);
    vboSource.push_back(g);
    vboSource.push_back(b);
    vboSource.push_back(u+w);
    vboSource.push_back(v);

    vboSource.push_back(float(x+FONT_CHAR_WIDTH));
    vboSource.push_back(float(y));
    vboSource.push_back(r);
    vboSource.push_back(g);
    vboSource.push_back(b);
    vboSource.push_back(u+w);
    vboSource.push_back(v);

    vboSource.push_back(float(x));
    vboSource.push_back(float(y+FONT_CHAR_HEIGHT));
    vboSource.push_back(r);
    vboSource.push_back(g);
    vboSource.push_back(b);
    vboSource.push_back(u);
    vboSource.push_back(v+h);

    vboSource.push_back(float(x+FONT_CHAR_WIDTH));
    vboSource.push_back(float(y+FONT_CHAR_HEIGHT));
    vboSource.push_back(r);
    vboSource.push_back(g);
    vboSource.push_back(b);
    vboSource.push_back(u+w);
    vboSource.push_back(v+h);

    return x + FONT_CHAR_WIDTH;
}

int
Hud::drawString(std::vector<float> &vboSource,
                int x, int y, float r, float g, float b, const char *c)
{
    while (*c) {
        char ch = (char)((*c) & 0x7f);
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
Hud::Rebuild(int width, int height, int framebufferWidth, int framebufferHeight)
{
    _requiresRebuildStatic = false;
    _windowWidth = width;
    _windowHeight = height;
    _framebufferWidth = framebufferWidth;
    _framebufferHeight = framebufferHeight;

    _staticVboSource.clear();

    int x, y;
    // add UI elements
    for (std::vector<Item>::const_iterator it = _labels.begin();
         it != _labels.end(); ++it) {
        getWindowPos(*it, &x, &y);
        drawString(_staticVboSource, x, y, 1, 1, 1, it->label.c_str());
    }
    // draw radio buttons
    for (std::vector<RadioButton>::const_iterator it = _radioButtons.begin();
         it != _radioButtons.end(); ++it) {
        getWindowPos(*it, &x, &y);
        if (it->checked) {
            x = drawChar(_staticVboSource, x, y, 1, 1, 1, FONT_RADIO_BUTTON_ON);
            drawString(_staticVboSource, x, y, 1, 1, 0, it->label.c_str());
        } else {
            x = drawChar(_staticVboSource, x, y, 1, 1, 1, FONT_RADIO_BUTTON_OFF);
            drawString(_staticVboSource, x, y, .5f, .5f, .5f, it->label.c_str());
        }
    }
    // draw checkboxes
    for (std::vector<CheckBox>::const_iterator it = _checkBoxes.begin();
         it != _checkBoxes.end(); ++it) {
        getWindowPos(*it, &x, &y);
        if (it->checked) {
            x = drawChar(_staticVboSource, x, y, 1, 1, 1, FONT_CHECK_BOX_ON);
            drawString(_staticVboSource, x, y, 1, 1, 0, it->label.c_str());
        } else {
            x = drawChar(_staticVboSource, x, y, 1, 1, 1, FONT_CHECK_BOX_OFF);
            drawString(_staticVboSource, x, y, .5f, .5f, .5f, it->label.c_str());
        }
    }
    // draw sliders
    for (std::vector<Slider>::const_iterator it = _sliders.begin();
         it != _sliders.end(); ++it) {
        getWindowPos(*it, &x, &y);
        int sx = x;
        x = drawString(_staticVboSource, x, y, 1, 1, 1, it->label.c_str());
        char value[16];
        if (it->intStep) {
            snprintf(value, 16, " : %d", (int)it->value);
        } else {
            snprintf(value, 16, " : %.2f", it->value);
        }
        drawString(_staticVboSource, x, y, 1, 1, 1, value);

        // new line
        y += FONT_CHAR_HEIGHT;
        x = sx;

        x = drawChar(_staticVboSource, x, y, 1, 1, 1, FONT_SLIDER_LEFT);
        int nw = it->w / FONT_CHAR_WIDTH;
        for (int i = 1; i < nw; ++i) {
            x = drawChar(_staticVboSource, x, y, 1, 1, 1, FONT_SLIDER_MIDDLE);
        }
        drawChar(_staticVboSource, x, y, 1, 1, 1, FONT_SLIDER_RIGHT);
        int pos = int(((it->value-it->min)/(it->max-it->min))*float(it->w));
        drawChar(_staticVboSource, sx+pos, y, 1, 1, 0, FONT_SLIDER_CURSOR);
    }
    // draw pulldowns
    for (std::vector<PullDown>::const_iterator it = _pulldowns.begin();
         it != _pulldowns.end(); ++it) {
        getWindowPos(*it, &x, &y);

        x = drawString(_staticVboSource, x, y, .5f, .5f, .5f, it->label.c_str());
        x += FONT_CHAR_WIDTH;

        if (it->open) {
            x = drawChar(_staticVboSource, x, y, 1, 1, 0, FONT_ARROW_DOWN);
            x += FONT_CHAR_WIDTH;
            for (int i=0; i<(int)it->labels.size(); ++i, y+=FONT_CHAR_HEIGHT) {
                if (i==it->selected) {
                    drawString(_staticVboSource, x, y, 1, 1, 0, it->labels[i]);
                } else {
                    drawString(_staticVboSource, x, y, 0.5f, 0.5f, 0.5f, it->labels[i]);
                }
            }
        } else {
            x = drawChar(_staticVboSource, x, y, .5f, .5f, .5f, FONT_ARROW_RIGHT);
            x += FONT_CHAR_WIDTH;
            drawString(_staticVboSource, x, y, 1, 1, 0, it->labels[it->selected]);
        }
    }

    // draw the character cells corresponding to the logo
    drawString(_staticVboSource, _windowWidth-128, _windowHeight-44, .5, .5, .5,
               "\x06\x07\x08\x09");
    drawString(_staticVboSource, _windowWidth-128, _windowHeight-28, .5, .5, .5,
               "\x16\x17\x18\x19");
    drawString(_staticVboSource, _windowWidth-92, _windowHeight-36, .5, .5, .5,
               "\x0a\x0b\x0c\x0d");
    drawString(_staticVboSource, _windowWidth-58, _windowHeight-36, .5, .5, .5,
               "\x1a\x1b\x1c\x1d\x1e\x1f");
}

bool
Hud::Flush()
{
    if (!_visible) {
        _vboSource.clear();
        return false;
    }

    if (_requiresRebuildStatic)
        this->Rebuild(_windowWidth, _windowHeight, _framebufferWidth, _framebufferHeight);

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

std::vector<float> &
Hud::getVboSource()
{
    return _vboSource;
}

std::vector<float> &
Hud::getStaticVboSource()
{
    return _staticVboSource;
}
