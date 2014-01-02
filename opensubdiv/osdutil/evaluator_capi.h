#ifndef __OPENSUBDIV_OSDUTIL_EVALUATOR_CAPI_H__
#define __OPENSUBDIV_OSDUTIL_EVALUATOR_CAPI_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Types declaration. */
struct OpenSubdiv_EvaluationDescr;

/* Methods to create and delete evaluators. */
struct OpenSubdiv_EvaluatorDescr *openSubdiv_createEvaluatorDescr(int numVertices);
void openSubdiv_deleteEvaluatorDescr(struct OpenSubdiv_EvaluatorDescr *evaluator_descr);
void openSubdiv_createEvaluatorDescrFace(struct OpenSubdiv_EvaluatorDescr *evaluator_descr, int num_vertices, int *indices);
void openSubdiv_finishEvaluatorDescr(struct OpenSubdiv_EvaluatorDescr *evaluator_descr, int refinementLevel);

/* Set the positions of points on the coarse mesh and refine. This method    */
/* will perform catmull/clark refinement on the CVs to be ready to call      */
/* openSubdiv_evaluateDescr.  Points here are 3 floats/point.                */
int openSubdiv_setEvaluatorCoarsePositions(
    struct OpenSubdiv_EvaluatorDescr *evaluator_descr,
    const float *positions, int numVertices);
    
/* Evaluate the subdivision limit surface at the given ptex face and u/v,    */
/* return position and derivative information.  Derivative pointers can be   */
/* NULL.  Note that face index here is the ptex index, or the index into     */
/* faces after turning all the faces on the original mesh into quads.        */
void openSubdiv_evaluateLimit(
    struct OpenSubdiv_EvaluatorDescr *evaluation_descr,
    int face_id, float u, float v,
    float P[3], float dPdu[3], float dPdv[3]);
    
    

#ifdef __cplusplus
}
#endif

#endif /* __OPENSUBDIV_CAPI_H__ */
