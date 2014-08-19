@ Copyright 2014 Google Inc. All rights reserved.

@ Licensed under the Apache License, Version 2.0 (the "License");
@ you may not use this file except in compliance with the License.
@ You may obtain a copy of the License at
@
@    http://www.apache.org/licenses/LICENSE-2.0
@
@ Unless required by applicable law or agreed to in writing, software
@ distributed under the License is distributed on an "AS IS" BASIS,
@ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@ See the License for the specific language governing permissions and
@ limitations under the License.

.global addWithWeight_ASM
.global OsdNeonComputeQuadFace_ASM
.global OsdNeonComputeTriQuadFace_ASM
.global OsdNeonComputeRestrictedEdge_ASM
.global OsdNeonComputeRestrictedVertexA_ASM
.global OsdNeonComputeRestrictedVertexB1_ASM
.global OsdNeonComputeRestrictedVertexB2_ASM

osdConstants:
 .float 0.25
 .float 0.3333333333
 .float 0.0
osdConstants2:
 .float 0.75
 .float 0.125
osdConstants3:
 .float 0.50
 .float 0.0625
osdConstArray: @ const float vw[8][2] = {
 .float 0.33333333, 0.11111111
 .float 0.5, 0.0625
 .float 0.6, 0.04
 .float 0.66666666, 0.027777777
 .float 0.714285714, 0.02040816
 .float 0.75, 0.015625
 .float 0.77777777, 0.012345679
 .float 0.8, 0.01

@                                          r0                    r1               r2          r3
@ call from C as void addWithWeight_ASM(float *dst, const float *srcOrigin, int srcIndex, float weight)
@
addWithWeight_ASM:
    stmfd   sp!,{r4,r14}
    vstmdb  sp!,{q4-q7}
    add r4,r1,r2,LSL #5     @ *src = srcOrigin + srcIndex * 8 * (sizeof(float))
    vdup.f32 q7,r3          @ duplicate float weight value in q7
    vld1.f32 {d0,d1,d2},[r4]    @ load 6 source floats
    vld1.f32 {d4,d5,d6},[r0]    @ load 6 dest floats
    pld [r4,#128]           @ preload next set into cache
    vmla.f32 q2,q0,d14[0]       @ multiply and accumulate 4 floats
    vmla.f32 d6,d2,d14[0]       @ mac the other 2
    vst1.f32 {d4,d5,d6},[r0]    @ store results back
    vldmia  sp!,{q4-q7}
    ldmfd   sp!,{r4,pc}

@                                                   r0              r1             r2                r3            sp[0]
@ call from C as void OsdNeonComputeQuadFace_ASM(float *vertex, const int *F_IT, int vertexOffset, int tableOffset, int batchSize)
@
OsdNeonComputeQuadFace_ASM:
 stmfd sp!,{r4-r12}
 ldr r4,[sp,#36]        @ batchSize
 vstmdb sp!,{q4-q7}
 add r1,r1,r3,LSL #2    @ &F_IT[tableOffset + (4 * i)]
 ldr r10,=osdConstants  @ point to constants
 ldr r10,[r10]          @ w = .25
 add r11,r0,r2,LSL #5   @ input/output = vertex + (8 * vertextOffset * sizeof(float))
 vdup.f32 q15,r10       @ 4 copies of w in Q15
quad_face_loop:
 ldmia r1!,{r6-r9}      @ idx0,idx1,idx2,idx3
 pld [r1,#128]          @ preload next set of index data into cache
 add r6,r0,r6,LSL #5    @ v0 = vertex + 8 * idx0
 add r7,r0,r7,LSL #5    @ v1 = vertex + 8 * idx1
 add r8,r0,r8,LSL #5    @ v2 = vertex + 8 * idx2
 add r9,r0,r9,LSL #5    @ v3 = vertex + 8 * idx3

 vld1.f32 {d0,d1,d2},[r6]       @ load v0[0]..v0[5] into Q0,D2
 vld1.f32 {d16,d17,d18},[r7]    @ load v1[0]..v1[5] into Q8,D18
 vld1.f32 {d20,d21,d22},[r8]    @ load v2[0]..v2[5] into Q10,D22
 vld1.f32 {d24,d25,d26},[r9]    @ load v3[0]..v3[5] into Q12,D26
 veor q2,q2,q2                  @ initialize dest accumulator
 veor d6,d6,d6
 vmla.f32 q2,q0,q15             @ dest[0..3] + v0[0..3] * weight
 vmla.f32 d6,d2,d30             @ dest[4..5] + v0[4..5] * weight
 vmla.f32 q2,q8,q15             @ dest[0..3] + v1[0..3] * weight
 vmla.f32 d6,d18,d30            @ dest[4..5] + v1[4..5] * weight
 vmla.f32 q2,q10,q15            @ dest[0..3] + v2[0..3] * weight
 vmla.f32 d6,d22,d30            @ dest[4..5] + v2[4..5] * weight
 vmla.f32 q2,q12,q15            @ dest[0..3] + v3[0..3] * weight
 vmla.f32 d6,d26,d30            @ dest[4..5] + v3[4..5] * weight
 vst1.f32 {d4,d5,d6},[r11]      @ store output[0..5]
 add r11,r11,#32                @ dstIndex++
 subs r4,r4,#1                  @ batchSize--
 bne quad_face_loop
 vldmia sp!,{q4-q7}
 ldmfd sp!,{r4-r12}
 bx lr

@                                                           r0              r1             r2                r3            sp[0]
@ call from C as void OsdNeonComputeTriQuadFace_ASM(float *vertex, const int *F_IT, int vertexOffset, int tableOffset, int batchSize)
@
OsdNeonComputeTriQuadFace_ASM:
 stmfd sp!,{r4-r12,lr}
 add r1,r1,r3,LSL #2    @ &F_IT[tableOffset + (4 * i)]
 ldr r3,[sp,#40]        @ batchSize
 vstmdb sp!,{q4-q7}
 ldr r4,=osdConstants   @ point to constants
 ldmia r4,{r10-r12}     @ w0 = .25, w1 = .333333, w2 = 0.00
 add r5,r0,r2,LSL #5    @ input/output = vertex + (8 * vertextOffset * sizeof(float))
triquad_face_loop:
 ldmia r1!,{r6-r9}      @ idx0,idx1,idx2,idx3
 mov r4,r10             @ assume all 4 floats (0.25)
 mov r14,r10
 pld [r1,#128]          @ preload next set of index data into cache
 add r6,r0,r6,LSL #5    @ v0 = vertex + 8 * idx0
 add r7,r0,r7,LSL #5    @ v1 = vertex + 8 * idx1
 add r8,r0,r8,LSL #5    @ v2 = vertex + 8 * idx2
 add r9,r0,r9,LSL #5    @ v3 = vertex + 8 * idx3
 vld1.f32 {d0,d1,d2},[r6]       @ load v0[0]..v0[5] into Q0,D2
 vld1.f32 {d16,d17,d18},[r7]    @ load v1[0]..v1[5] into Q8,D18
 vld1.f32 {d20,d21,d22},[r8]    @ load v2[0]..v2[5] into Q10,D22
 vld1.f32 {d24,d25,d26},[r9]    @ load v3[0]..v3[5] into Q12,D26
 cmp r8,r9              @ if (idx2 == idx3) // only compute 3 floats
 moveq r4,r11           @ use 1/3
 moveq r14,r12          @ use 0.0 for 4th weight
 vdup.f32 q15,r4
 vdup.f32 q7,r14
 veor q2,q2,q2                  @ initialize dest accumulator
 veor d6,d6,d6
 vmla.f32 q2,q0,q15             @ dest[0..3] + v0[0..3] * weight
 vmla.f32 d6,d2,d30             @ dest[4..5] + v0[4..5] * weight
 vmla.f32 q2,q8,q15             @ dest[0..3] + v1[0..3] * weight
 vmla.f32 d6,d18,d30            @ dest[4..5] + v1[4..5] * weight
 vmla.f32 q2,q10,q15            @ dest[0..3] + v2[0..3] * weight
 vmla.f32 d6,d22,d30            @ dest[4..5] + v2[4..5] * weight
 vmla.f32 q2,q12,q7             @ dest[0..3] + v3[0..3] * weight
 vmla.f32 d6,d26,d14            @ dest[4..5] + v3[4..5] * weight
 vst1.f32 {d4,d5,d6},[r5]       @ store output[0..5]
 add r5,r5,#32                  @ dstIndex++
 subs r3,r3,#1                  @ batchSize--
 bne triquad_face_loop
 vldmia sp!,{q4-q7}
 ldmfd sp!,{r4-r12,pc}

@                                                           r0              r1             r2                r3            sp[0]
@ call from C as void OsdNeonComputeRestrictedEdge_ASM(float *vertex, const int *E_IT, int vertexOffset, int tableOffset, int batchSize)
@
OsdNeonComputeRestrictedEdge_ASM:
 stmfd sp!,{r4-r12}
 ldr r4,[sp,#36]        @ batchSize
 vstmdb sp!,{q4-q7}
 add r1,r1,r3,LSL #4    @ &E_IT[4*(tableOffset+i)]
 ldr r10,=osdConstants  @ point to constants
 ldr r10,[r10]          @ w = .25
 add r11,r0,r2,LSL #5   @ input/output = vertex + (8 * vertextOffset * sizeof(float))
 vdup.f32 q15,r10       @ 4 copies of w in Q15
restricted_loop:
 ldmia r1!,{r6-r9}      @ idx0,idx1,idx2,idx3
 pld [r1,#128]          @ preload next set of index data into cache
 add r6,r0,r6,LSL #5    @ v0 = vertex + 8 * idx0
 add r7,r0,r7,LSL #5    @ v1 = vertex + 8 * idx1
 add r8,r0,r8,LSL #5    @ v2 = vertex + 8 * idx2
 add r9,r0,r9,LSL #5    @ v3 = vertex + 8 * idx3

 vld1.f32 {d0,d1,d2},[r6]       @ load v0[0]..v0[5] into Q0,D2
 vld1.f32 {d16,d17,d18},[r7]    @ load v1[0]..v1[5] into Q8,D18
 vld1.f32 {d20,d21,d22},[r8]    @ load v2[0]..v2[5] into Q10,D22
 vld1.f32 {d24,d25,d26},[r9]    @ load v3[0]..v3[5] into Q12,D26
 veor q2,q2,q2                  @ initialize dest accumulator
 veor d6,d6,d6
 vmla.f32 q2,q0,q15             @ dest[0..3] + v0[0..3] * weight
 vmla.f32 d6,d2,d30             @ dest[4..5] + v0[4..5] * weight
 vmla.f32 q2,q8,q15             @ dest[0..3] + v1[0..3] * weight
 vmla.f32 d6,d18,d30            @ dest[4..5] + v1[4..5] * weight
 vmla.f32 q2,q10,q15            @ dest[0..3] + v2[0..3] * weight
 vmla.f32 d6,d22,d30            @ dest[4..5] + v2[4..5] * weight
 vmla.f32 q2,q12,q15            @ dest[0..3] + v3[0..3] * weight
 vmla.f32 d6,d26,d30            @ dest[4..5] + v3[4..5] * weight
 vst1.f32 {d4,d5,d6},[r11]      @ store output[0..5]
 add r11,r11,#32                @ dstIndex++
 subs r4,r4,#1                  @ batchSize--
 bne restricted_loop
 vldmia sp!,{q4-q7}
 ldmfd sp!,{r4-r12}
 bx lr

@                                                             r0              r1              r2                 r3               sp[0]          sp[4]  sp[8]
@ call from C as void OsdNeonComputeRestrictedVertexB1(float *vertex, const int *V_ITa, const int *V_IT, int vertexOffset, int tableOffset, int start, int end)
@
OsdNeonComputeRestrictedVertexB1_ASM:
 stmfd sp!,{r4-r12}
 ldr r4,[sp,#36]        @ tableOffset
 ldr r5,[sp,#40]        @ start
 ldr r6,[sp,#44]        @ end
 vstmdb sp!,{q4-q7}
 add r4,r4,r5           @ i = tableOffset + start
 add r4,r4,r4,LSL #2    @ i *= 5
 add r1,r1,r4,LSL #2    @ &V_ITa[5*i]
 add r3,r3,r5           @ vertexOffset += start
 add r3,r0,r3,LSL #5    @ &vertex[(vertexOffset + start)*8]
 sub r4,r6,r5           @ length = end - start

 ldr r10,=osdConstants3 @ point to constants
 ldr r11,[r10],#4       @ w0 = 0.50f
 ldr r10,[r10]          @ w1 = 0.0625f
 vdup.f32 q15,r11       @ 4 copies of w in Q15
 vdup.f32 q14,r10
restrictedvertB1_loop:
 ldr r5,[r1]            @ h = V_ITa[5*i]
 ldr r6,[r1,#8]         @ p = V_ITa[5*i + 2]
 add r1,r1,#20          @ 5 * sizeof(int)
 pld [r1,#128]          @ preload next set of index data into cache
 add r6,r0,r6,LSL #5    @ v0 = vertex + 8 * p
 add r5,r2,r5,LSL #2    @ &V_IT[h]
 vld1.f32 {d0,d1,d2},[r6]       @ load [p]
 ldmia r5,{r5-r12}      @ V_IT[h+0]...V_IT[h+7]
 veor q2,q2,q2          @ initialize dest accumulator
 veor d3,d3,d3
 add r5,r0,r5,LSL #5    @ prepare pointers to each set of 8 indices
 add r6,r0,r6,LSL #5
 add r7,r0,r7,LSL #5
 add r8,r0,r8,LSL #5
 add r9,r0,r9,LSL #5
 add r10,r0,r10,LSL #5
 add r11,r0,r11,LSL #5
 add r12,r0,r12,LSL #5
 vmla.f32 q2,q0,q15             @ [p] * 0.5
 vmla.f32 d3,d2,d30             @ [p] * 0.5
 vld1.f32 {d0,d1,d2},[r5]       @ [h+0]
 vld1.f32 {d6,d7,d8},[r6]       @ [h+1]
 vld1.f32 {d10,d11,d12},[r7]    @ [h+2]
 vld1.f32 {d14,d15,d16},[r8]    @ [h+3]
 vld1.f32 {d18,d19,d20},[r9]    @ [h+4]
 vld1.f32 {d22,d23,d24},[r10]   @ [h+5]
 vmla.f32 q2,q0,q14             @ [h+0] * 0.0625
 vmla.f32 d3,d2,d28             @ [h+0] * 0.0625
 vmla.f32 q2,q3,q14             @ [h+1] * 0.0625
 vmla.f32 d3,d8,d28             @ [h+1] * 0.0625
 vmla.f32 q2,q5,q14             @ [h+2] * 0.0625
 vmla.f32 d3,d12,d28            @ [h+2] * 0.0625
 vmla.f32 q2,q7,q14             @ [h+3] * 0.0625
 vmla.f32 d3,d16,d28            @ [h+3] * 0.0625
 vld1.f32 {d0,d1,d2},[r11]      @ [h+6]
 vld1.f32 {d6,d7,d8},[r12]      @ [h+7]
 vmla.f32 q2,q9,q14             @ [h+4] * 0.0625
 vmla.f32 d3,d20,d28            @ [h+4] * 0.0625
 vmla.f32 q2,q11,q14            @ [h+5] * 0.0625
 vmla.f32 d3,d24,d28            @ [h+5] * 0.0625
 vmla.f32 q2,q0,q14             @ [h+6] * 0.0625
 vmla.f32 d3,d2,d28             @ [h+6] * 0.0625
 vmla.f32 q2,q3,q14             @ [h+7] * 0.0625
 vmla.f32 d3,d8,d28             @ [h+7] * 0.0625
 vmov d6,d3
 vst1.f32 {d4,d5,d6},[r3]       @ store output[0..5]
 add r3,r3,#32                  @ dstIndex++
 subs r4,r4,#1                  @ batchSize--
 bne restrictedvertB1_loop
 vldmia sp!,{q4-q7}
 ldmfd sp!,{r4-r12}
 bx lr

@                                                               r0              r1                  r2              r3              sp[0]           sp[4]   sp[8]
@ call from C as void OsdNeonComputeRestrictedVertexB2_ASM(float *vertex, const int *V_ITa, const int *V_IT, int vertexOffset, int tableOffset, int start, int end)
@
OsdNeonComputeRestrictedVertexB2_ASM:
 stmfd sp!,{r4-r12}
 ldr r4,[sp,#36]        @ tableOffset
 ldr r5,[sp,#40]        @ start
 ldr r6,[sp,#44]        @ end
 vstmdb sp!,{q4-q7}
 add r4,r4,r5           @ i = tableOffset + start
 add r4,r4,r4,LSL #2    @ i *= 5
 add r1,r1,r4,LSL #2    @ &V_ITa[5*i]
 add r3,r3,r5           @ vertexOffset += start
 add r3,r0,r3,LSL #5    @ &vertex[(vertexOffset + start)*8]
 sub r4,r6,r5           @ length = end - start
 ldr r10,=osdConstArray @ point to constants
 sub r10,r10,#24        @ instead of subtracting 3 from n each time through the loop
restrictedvertB2_loop:
 ldmia r1,{r5-r7}       @ h = V_ITa[5*i], n = V_ITa[5*i+1], p = V_ITa[5*i+2]
 add r1,r1,#20          @ i += 5 * sizeof(int)
 pld [r1,#128]          @ preload next set of index data into cache
 @ assert (n >= 3 && n <= 10)
 add r9,r10,r6,LSL #3   @ point to vw[n-3][]
 ldmia r9,{r11-r12}     @ read vw[n-3][0] and vw[n-3][1]
 add r7,r0,r7,LSL #5    @ v0 = vertex + 8 * p
 vld1.f32 {d0,d1,d2},[r7]       @ load p[0..5]
 vdup.f32 q15,r11       @ for multiplying "p"
 vdup.f32 q14,r12       @ for multiplying the "n" values
 add r5,r2,r5,LSL #2    @ &V_IT[h]
 veor q2,q2,q2          @ initialize dest accumulator
 veor d6,d6,d6
 vmla.f32 q2,q0,q15     @ p[0..3] * vw[n-3][0]
 vmla.f32 d6,d2,d30     @ p[4..5] * vw[n-3][0]
 @ process it in pairs since it's n*2
 vertB2_inner_loop:
 ldmia r5!,{r8-r9}              @ index = V_IT[h++] (x2)
 add r8,r0,r8,LSL #5
 add r9,r0,r9,LSL #5
 vld1.f32 {d0,d1,d2},[r8]       @ h[0..5]
 vld1.f32 {d8,d9,d10},[r9]      @ h[0..5]
 subs r6,r6,#1          @ for (j=0; j< n*2; j++, h++)
 vmla.f32 q2,q0,q14             @ h+0[0..3] * vw[n-3][1]
 vmla.f32 d6,d2,d28             @ h+0[4..5] * vw[n-3][1]
 vmla.f32 q2,q4,q14             @ h+1[0..3] * vw[n-3][1]
 vmla.f32 d6,d10,d28            @ h+1[4..5] * vw[n-3][1]
 bne vertB2_inner_loop
 vst1.f32 {d4,d5,d6},[r3]       @ store output[0..5]
 add r3,r3,#32                  @ dstIndex++
 subs r4,r4,#1                  @ batchSize--
 bne restrictedvertB2_loop
 vldmia sp!,{q4-q7}
 ldmfd sp!,{r4-r12}
 bx lr

@                                                             r0              r1              r2                 r3          sp[0]     sp[4]
@ call from C as void OsdNeonComputeRestrictedVertexA(float *vertex, const int *V_ITa, int vertexOffset, int tableOffset, int start, int end)
@
OsdNeonComputeRestrictedVertexA_ASM:
 stmfd sp!,{r4-r12}
 ldr r4,[sp,#36]        @ start
 ldr r5,[sp,#40]        @ end
 vstmdb sp!,{q4-q7}
 add r3,r3,r4           @ i = tableOffset + start
 add r2,r2,r4           @ vertexOffset += start
 add r2,r0,r2,LSL #5    @ &vertex[(vertexOffset + start)*8]
 sub r4,r5,r4           @ length = end - start
 add r3,r3,r3,LSL #2    @ i *= 5
 add r1,r1,r3,LSL #2    @ &V_ITa[5*i]
 add r1,r1,#8           @ + 2

 ldr r10,=osdConstants2 @ point to constants
 ldr r11,[r10],#4       @ w0 = 0.75f
 ldr r10,[r10]          @ w1 = 0.125f
 vdup.f32 q15,r11       @ 4 copies of w0 in Q15
 vdup.f32 q14,r10
restrictedvertA_loop:
 ldmia r1,{r6-r8}       @ p,eidx0,eidx1
 add r1,r1,#20          @ needed to increment in sets of 5 (i*5)
 pld [r1,#128]          @ preload next set of index data into cache
 add r6,r0,r6,LSL #5    @ v0 = vertex + 8 * p
 add r7,r0,r7,LSL #5    @ v1 = vertex + 8 * eidx0
 add r8,r0,r8,LSL #5    @ v2 = vertex + 8 * eidx1

 vld1.f32 {d0,d1,d2},[r6]       @ load v0[0..5] into Q0,D2
 vld1.f32 {d16,d17,d18},[r7]    @ load v1[0..5] into Q8,D18
 vld1.f32 {d20,d21,d22},[r8]    @ load v2[0..5] into Q10,D22
 veor q2,q2,q2                  @ initialize dest accumulator
 veor d6,d6,d6
 vmla.f32 q2,q0,q15             @ p[0..3] * 0.75
 vmla.f32 d6,d2,d30             @ p[4..5] * 0.75
 vmla.f32 q2,q8,q14             @ eidx0[0..3] * 0.125
 vmla.f32 d6,d18,d28            @ eidx0[4..5] * 0.125
 vmla.f32 q2,q10,q14            @ eidx1[0..3] * 0.125
 vmla.f32 d6,d22,d28            @ eidx1[4..5] * 0.125
 vst1.f32 {d4,d5,d6},[r2]       @ store output[0..5]
 add r2,r2,#32                  @ dstIndex++
 subs r4,r4,#1                  @ batchSize--
 bne restrictedvertA_loop
 vldmia sp!,{q4-q7}
 ldmfd sp!,{r4-r12}
 bx lr

  .end
