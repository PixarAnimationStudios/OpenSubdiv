#ifndef GSD_VERTEX_H
#define GSD_VERTEX_H

class OsdVertex {
public:
    OsdVertex() {}
    OsdVertex(int index) {}
    OsdVertex(const OsdVertex &src) {}

    void AddWithWeight(const OsdVertex & i, float weight, void * = 0) {}
    void AddVaryingWithWeight(const OsdVertex & i, float weight, void * = 0) {} 
    void Clear(void * = 0) {}
};

#endif // GSD_VERTEX_H
