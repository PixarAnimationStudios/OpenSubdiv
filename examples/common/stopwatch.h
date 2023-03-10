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

#ifndef STOPWATCH_H
#define STOPWATCH_H

#if (_WIN32)
    #include <windows.h>
#else
    #include <sys/types.h>
    #include <sys/time.h>
    #include <sys/resource.h>
#endif

class Stopwatch {

public:

#ifndef _WIN32
    Stopwatch() : _totalElapsed(0) { }

    void Start() {
        struct timeval l_rtime;
        gettimeofday(&l_rtime,0);
        _elapsed = (double)l_rtime.tv_sec + (double)l_rtime.tv_usec/1000000.0;
    }

    void Stop() {
        struct timeval l_rtime;

        gettimeofday(&l_rtime,0);
        _elapsed = ((double)l_rtime.tv_sec + (double)l_rtime.tv_usec/1000000.0)
                 - _elapsed;
        _totalElapsed += _elapsed;
    }

    double GetElapsed() const {
        return _elapsed;
    }

    double GetTotalElapsed() const {
        return _totalElapsed;
    }
#else
    Stopwatch() : _totalElapsed(0) {
        QueryPerformanceFrequency(&_frequency);
    }

    void Start()
    {
        QueryPerformanceCounter(&_time);
    }

    void Stop()
    {
        LARGE_INTEGER currentTime;
        QueryPerformanceCounter(&currentTime);
        _elapsed = currentTime.QuadPart - _time.QuadPart;
        _totalElapsed+=_elapsed;
    }

    double GetElapsed() const {
        return (double) _elapsed / _frequency.QuadPart;
    }

    double GetTotalElapsed() const {
        return (double) _totalElapsed / _frequency.QuadPart;
    }
#endif

private:

#ifndef _WIN32
    double _elapsed;
    double _totalElapsed;
#else
    LARGE_INTEGER _time;
    LARGE_INTEGER _frequency;
    __int64 _elapsed;
    __int64 _totalElapsed;
#endif

};

#endif /* STOPWATCH_H */

