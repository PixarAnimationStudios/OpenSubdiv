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

#ifndef HUD_H
#define HUD_H

#include <algorithm>
#include <cstdarg>
#include <math.h>
#include <string>
#include <vector>

#define OSD_HUD_USE_FUNCTION_POINTERS (__cplusplus <= 199711L)

#if !OSD_HUD_USE_FUNCTION_POINTERS
#include <functional>
#endif

#include "hud.h"

class Hud
{
public:
#if OSD_HUD_USE_FUNCTION_POINTERS
    typedef void (*RadioButtonCallback)(int c);
    typedef void (*CheckBoxCallback)(bool checked, int data);
    typedef void (*SliderCallback)(float value, int data);
    typedef void (*PullDownCallback)(int value);
#else
    typedef std::function<void(int)> RadioButtonCallback;
    typedef std::function<void(bool,int)> CheckBoxCallback;
    typedef std::function<void(float, int)> SliderCallback;
    typedef std::function<void(int)> PullDownCallback;
#endif

    Hud();
    virtual ~Hud();

    virtual void Init(int width, int height, int framebufferWidth, int framebufferHeight);

    virtual void Rebuild(int width, int height,
                         int framebufferWidth, int framebufferHeight);

    virtual bool Flush();

    bool IsVisible() const;

    void SetVisible(bool visible);

    void DrawString(int x, int y, const char *fmt, ...);

    void DrawString(int x, int y, float r, float g, float b, const char *fmt, ...);

    void Clear();

    void AddLabel(const char *label, int x, int y);

    void AddRadioButton(int group, const char *label, bool checked, int x, int y,
                        RadioButtonCallback callback=0, int data=0, int shortcut=0);

    void AddCheckBox(const char *label, bool checked, int x, int y,
                     CheckBoxCallback callback=0, int data=0, int shortcut=0);

    void AddSlider(const char *label, float min, float max, float value,
                   int x, int y, int width, bool intStep,
                   SliderCallback callback=0, int data=0);

    int AddPullDown(const char *label, int x, int y, int width,
                    PullDownCallback callback=0, int shortcut=0);

    void AddPullDownButton(int handle, const char *label, int value, bool checked=false);

    bool KeyDown(int key);

    bool MouseClick(int x, int y);

    bool MouseCapture() const;

    void MouseRelease();

    void MouseMotion(int x, int y);

    int GetWidth() const;

    int GetHeight() const;

protected:
    struct Item {

        int x, y, w, h;
        std::string label;
    };

    struct RadioButton : public Item {

        int group;
        int localIndex;
        bool checked;
        int shortcut;
        bool sharedShortcut;
        RadioButtonCallback callback;
        int callbackData;
    };

    struct CheckBox : public Item {

        bool checked;
        CheckBoxCallback callback;
        int callbackData;
        int shortcut;
    };

    struct Slider : public Item {

        float min, max;
        float value;
        SliderCallback callback;
        int callbackData;
        bool intStep;

        void SetValue(float v) {
            v = std::max(std::min(v, max), min);
            if (intStep) {
                // MSVC 2010 does not have std::round() or roundf()
                v = v>0.0f ? floorf(v+0.5f) : ceilf(v-0.5f);
            }
            value = v;
        }
    };

    struct PullDown : public Item {

        bool open;
        int selected;
        std::vector<char const *> labels;
        std::vector<int> values;
        int shortcut;
        PullDownCallback callback;

        void SetSelected(int idx) {
            if (idx>=0 && idx<(int)labels.size()) {
                selected=idx;
            }
        }
    };

    static int drawString(std::vector<float> &vboSource, int x, int y,
                   float r, float g, float b, const char *c);

    static int drawChar(std::vector<float> &vboSource, int x, int y,
                   float r, float g, float b, char ch);

    bool hitTest(Item const &item, int x, int y) const {
        int ix = item.x > 0 ? item.x : _windowWidth + item.x;
        int iy = item.y > 0 ? item.y : _windowHeight + item.y;
        return (x >= ix &&
                y >= iy &&
                x <= (ix + item.w) &&
                y <= (iy + item.h));
    }

    void getWindowPos(Item const &item, int *x, int *y) const {
        *x = item.x > 0 ? item.x : _windowWidth + item.x;
        *y = item.y > 0 ? item.y : _windowHeight + item.y;
    }

    std::vector<float> & getVboSource();
    std::vector<float> & getStaticVboSource();

private:

    bool _visible;
    std::vector<float> _vboSource, _staticVboSource;
    int _windowWidth, _windowHeight;
    int _framebufferWidth, _framebufferHeight;
    bool _requiresRebuildStatic;
    std::vector<Item> _labels;
    std::vector<RadioButton> _radioButtons;
    std::vector<CheckBox> _checkBoxes;
    std::vector<Slider> _sliders;
    std::vector<PullDown> _pulldowns;
    int _capturedSlider;
};

#endif // HUD_H
