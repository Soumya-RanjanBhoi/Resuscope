import os
import uvicorn
from typing import Dict, Any, Union, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from contextlib import asynccontextmanager

from models.matcher import SemanticModel

from models.extractor import extract_text_from_docx, extract_text_from_pdf, extract_all_skills


from models.rewriter import (
    get_required_skills, 
    get_content_score,
    get_structure_score,
    optimize_skills_section,
    optimize_structure, 
    check_content, 
    check_tone_and_style,
    get_professional_summary_suggestions
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 App is starting up...")
    print("📦 Loading semantic model...")

    try:
        SemanticModel.get_instance()
        print("✅ Semantic Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load semantic model: {e}")
        raise RuntimeError("Critical startup error: Semantic model failed to load.")

    yield

    print("🛑 App is shutting down...")



app = FastAPI(title="AI Resume Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_uploaded_file(file: UploadFile = File(...)) -> str:
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Unsupported File Format. Please upload .docx or .pdf")

    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file.file)
        else:
            text = extract_text_from_docx(file.file)

        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from file or file is empty.")
        return text

    except HTTPException:
        raise
    except Exception as e:
        print(f"File processing error: {e}")
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@app.get('/')
def start():
    return {"message": "AI Resume Analyzer API is running successfully"}

@app.get('/health')
def health_check():
    try:
        SemanticModel.get_instance()
        model_ready = True
    except:
        model_ready = False
        
    return {"status": "healthy", "model_ready": model_ready, "message": "Service is running"}

@app.post('/analyze_resume', summary="Comprehensive Resume Analysis")
async def analyze_resume(
    job_description: str,
    job_title:str,
    resume_text: str = Depends(process_uploaded_file)
):
    try:
        semantic_scorer = SemanticModel.get_instance()
        
        res_skill = extract_all_skills(resume_text)
        req_skill = get_required_skills(job_description) 

        req_tech_skill: List[str] = req_skill.get('technical_skills', [])
        req_soft_skill: List[str] = req_skill.get('soft_skills', [])

        res_tech_skill = [index.get('skills',[]) for index in res_skill.get('skill_sets', []) if index.get('category') == "TECHNICAL"]
        res_soft_skill = [index.get('skills',[]) for index in res_skill.get('skill_sets', []) if index.get('category') == "SOFT"]
        
        tech_input: List[str] = res_tech_skill[0] if res_tech_skill and res_tech_skill[0] else []
        soft_input: List[str] = res_soft_skill[0] if res_soft_skill and res_soft_skill[0] else []


        tech_score = semantic_scorer.score(req_tech_skill, tech_input)
        soft_score = semantic_scorer.score(req_soft_skill, soft_input)

        skill_match_score = 0.75 * tech_score + 0.25 * soft_score 

        ats_match_score = semantic_scorer.score(job_description, resume_text) 
        
        content_analysis = get_content_score(resume_text, job_title)
        structure_analysis = get_structure_score(resume_text)

        content_score = content_analysis.get('score', 0) if isinstance(content_analysis, dict) else 0
        structure_score = structure_analysis.get('score', 0) if isinstance(structure_analysis, dict) else 0


        overall_score = round( (ats_match_score * 0.5) + (skill_match_score * 0.3) + (content_score * 0.1) + (structure_score * 0.1) )
        improvements = get_professional_summary_suggestions(round(overall_score),resume_text,job_title)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "overall_score": overall_score,
                "scores": {
                    "ats_compatibility_score": round(ats_match_score),
                    "skill_match_score": round(skill_match_score),
                    "content_quality_score": content_score,
                    "structure_score": structure_score,
                },
                "improvements_suggestion":improvements
            }
        )

    except Exception as e:
        print(f"Error in analyze_resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {str(e)}")


@app.post("/optimize-skills", summary="Suggests skills to add/remove")
async def optimize_skills_endpoint(
    job_description: str,
    job_title: str, 
    resume_text: str = Depends(process_uploaded_file)
):
    try:
        resume_skills_raw = extract_all_skills(resume_text)
        req_skills_data = get_required_skills(job_description)
        
        skills_details = optimize_skills_section(resume_skills_raw, req_skills_data, job_title)

        return JSONResponse(status_code=200, content={"success": True, "skills_optimization": skills_details})
    
    except Exception as e:
        print(f"Error in optimize-skills: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




@app.post('/optimize_structure_feedback', summary="Get actionable structure feedback")
async def get_structure_feedback_endpoint(resume_text: str = Depends(process_uploaded_file)):
    try:
        feedback = optimize_structure(resume_text)
        return JSONResponse(status_code=200, content={"success": True, "structure_feedback": feedback})
    
    except Exception as e:
        print(f"Error in optimize_structure_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/optimize_content_feedback", summary="Get actionable content feedback")
async def get_content_feedback_endpoint(resume_text: str = Depends(process_uploaded_file)):
    try:
        feedback = check_content(resume_text)
        return JSONResponse(status_code=200, content={'success': True, 'content_feedback': feedback})
    
    except Exception as e:
        print(f"Error in optimize_content_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post('/optimize_tone_style_feedback', summary="Get actionable tone and style feedback")
async def get_tone_style_feedback_endpoint(resume_text: str = Depends(process_uploaded_file)):
    try:
        feedback = check_tone_and_style(resume_text)
        return JSONResponse(status_code=200, content={"success": True, "tone_style_feedback": feedback})
    
    except Exception as e:
        print(f"Error in optimize_tone_style_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)