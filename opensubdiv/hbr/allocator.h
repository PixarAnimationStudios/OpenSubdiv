//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//
#ifndef HBRALLOCATOR_H
#define HBRALLOCATOR_H

typedef void (*HbrMemStatFunction)(unsigned long bytes);

/**
 * HbrAllocator - derived from UtBlockAllocator.h, but embedded in
 * libhbrep.
 */
template <typename T> class HbrAllocator {

public:

    /// Constructor
    HbrAllocator(size_t *memorystat, int blocksize, void (*increment)(unsigned long bytes), void (*decrement)(unsigned long bytes), size_t elemsize = sizeof(T));

    /// Destructor
    ~HbrAllocator();

    /// Create an allocated object
    T * Allocate();

    /// Return an allocated object to the block allocator
    void Deallocate(T *);

    /// Clear the allocator, deleting all allocated objects.
    void Clear();

    void SetMemStatsIncrement(void (*increment)(unsigned long bytes)) { m_increment = increment; }

    void SetMemStatsDecrement(void (*decrement)(unsigned long bytes)) { m_decrement = decrement; }
    
private:
    size_t *m_memorystat;
    const int m_blocksize;
    int m_elemsize;
    T** m_blocks;

    // Number of actually allocated blocks
    int m_nblocks;

    // Size of the m_blocks array (which is NOT the number of actually
    // allocated blocks)
    int m_blockCapacity;

    int m_freecount;
    T * m_freelist;

    // Memory statistics tracking routines
    HbrMemStatFunction m_increment;
    HbrMemStatFunction m_decrement;
};
					  
template <typename T>
HbrAllocator<T>::HbrAllocator(size_t *memorystat, int blocksize, void (*increment)(unsigned long bytes), void (*decrement)(unsigned long bytes), size_t elemsize)
    : m_memorystat(memorystat), m_blocksize(blocksize), m_elemsize(elemsize), m_blocks(0), m_nblocks(0), m_blockCapacity(0), m_freecount(0), m_increment(increment), m_decrement(decrement) {
}

template <typename T>
HbrAllocator<T>::~HbrAllocator() {
    Clear();
}

template <typename T>
void HbrAllocator<T>::Clear() {
    for (int i = 0; i < m_nblocks; ++i) {
	// Run the destructors (placement)
	T* blockptr = m_blocks[i];
	T* startblock = blockptr;
	for (int j = 0; j < m_blocksize; ++j) {
	    blockptr->~T();
	    blockptr = (T*) ((char*) blockptr + m_elemsize);
	}
	free(startblock);
	if (m_decrement) m_decrement(m_blocksize * m_elemsize);
        *m_memorystat -= m_blocksize * m_elemsize;
    }
    free(m_blocks);
    m_blocks = 0;
    m_nblocks = 0;
    m_blockCapacity = 0;
    m_freecount = 0;
    m_freelist = NULL;
}

template <typename T>
T* 
HbrAllocator<T>::Allocate() {
    if (!m_freecount) {

	// Allocate a new block
	T* block = (T*) malloc(m_blocksize * m_elemsize);
	T* blockptr = block;
	// Run the constructors on each element using placement new
	for (int i = 0; i < m_blocksize; ++i) {
	    new (blockptr) T();
	    blockptr = (T*) ((char*) blockptr + m_elemsize);
	}
	if (m_increment) m_increment(m_blocksize * m_elemsize);
        *m_memorystat += m_blocksize * m_elemsize;
	
	// Put the block's entries on the free list
	blockptr = block;
	for (int i = 0; i < m_blocksize - 1; ++i) {
	    T* next = (T*) ((char*) blockptr + m_elemsize);
	    blockptr->GetNext() = next;
	    blockptr = next;
	}
	blockptr->GetNext() = 0;
	m_freelist = block;

	// Keep track of the newly allocated block
	if (m_nblocks + 1 >= m_blockCapacity) {
	    m_blockCapacity = m_blockCapacity * 2;
	    if (m_blockCapacity < 1) m_blockCapacity = 1;
	    m_blocks = (T**) realloc(m_blocks, m_blockCapacity * sizeof(T*));
	}
	m_blocks[m_nblocks] = block;
	m_nblocks++;
	m_freecount += m_blocksize;
    }
    T* obj = m_freelist;
    m_freelist = obj->GetNext();
    obj->GetNext() = 0;
    m_freecount--;
    return obj;
}

template <typename T>
void
HbrAllocator<T>::Deallocate(T * obj) {
    assert(!obj->GetNext());
    obj->GetNext() = m_freelist;
    m_freelist = obj;
    m_freecount++;
}

#endif /* HBRALLOCATOR_H */
