import os
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable

load_dotenv()


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0, 
        max_retries=2
    )

def create_chain(pydantic_model: BaseModel, template_str: str, input_vars: List[str]) -> Runnable:
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    prompt = PromptTemplate(
        template=template_str,
        input_variables=input_vars, 
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt | llm | parser


class RequiredSkillsResponse(BaseModel):
    technical_skills: List[str] = Field(description="List of specific technical hard skills required.")
    soft_skills: List[str] = Field(description="List of essential soft skills required.")

class ContentScoreResponse(BaseModel):
    score: int = Field(description="A score from 0-100 based on the quality of the resume content.")
    reasoning: str = Field(description="A brief explanation of why this score was given.")
    missing_keywords: List[str] = Field(description="List of important keywords or skills found in the job title context that are missing from the text.")
    improvement_tips: List[str] = Field(description="Exactly 3 actionable tips to improve the resume content.")

class StructureScoreResponse(BaseModel):
    score: int = Field(..., description="Integer score between 0-100 representing structure quality.")
    reasoning: str = Field(..., description="Brief explanation of why this score was given.")


class SkillOptimizationResponse(BaseModel):
    skills_to_add: List[str] = Field(..., description="List of relevant skills missing from the resume, highly relevant to the job title.")

class SectionSuggestionResponse(BaseModel):
    missing_sections: List[str] = Field(..., description="List of standard resume sections that are missing but highly recommended for the target job.")
    explanation: str = Field(..., description="Why these sections are recommended and what content they should contain.")

class FeedbackResponse(BaseModel):
    key_points: List[str] = Field(..., description="Exactly 4 key actionable points for improvement (e.g., style, tone, clarity).")
    has_issues: bool = Field(..., description="True if significant issues were found, otherwise False.")

class ProfessionalSummaryResponse(BaseModel):
    key_points: List[str] = Field(...,description="List of 4 distinct professional summary options, each 3-4 Phrases.")

def get_required_skills(job_description: str) -> Dict[str, Any]:
    try:
        template = """
            Act as an expert technical recruiter and resume parser. 
            Your task is to STRICTLY extract and categorize the skills mentioned in the provided job description.
            DO NOT add any skill yourself that is not explicitly present in the job description.

            Job Description:
            ---
            {job_description}
            ---

            Identify the essential skills and separate them into 'technical_skills' and 'soft_skills'.
            {format_instructions}
        """

        input_vars = ['job_description']
        chain = create_chain(RequiredSkillsResponse, template, input_vars)
        
        result = chain.invoke({'job_description': job_description})
        return result.model_dump()
    
    except Exception as e:
        return {"technical_skills": [], "soft_skills": [], "error": f"Error generating skills: {e}"}


def get_content_score(text: str, job_title: str) -> Dict[str, Any]:
    try:
        template = """
            Act as a strict Resume ATS (Applicant Tracking System).
            Analyze the resume text provided below for the role of: {job_title}.

            Evaluate the **content quality** based on:
            1. Relevance to the target job ({job_title})
            2. Presence of quantified achievements (metrics, numbers)
            3. Clarity and use of action verbs
            4. Absence of spelling/grammar errors (Score lower if found)

            Resume Text:
            ---
            {text}
            ---

            Based on this analysis, provide a score (0-100), the reasoning, missing keywords, and exactly 3 actionable tips for improvement.
            {format_instructions}
            """
        input_vars = ['job_title', 'text']
        chain = create_chain(ContentScoreResponse, template, input_vars)

        result = chain.invoke({'job_title': job_title, 'text': text})
        return result.model_dump()
    
    except Exception as e:
        return {"score": 0, "reasoning": f"Failed to analyze resume: {e}", "missing_keywords": [], "improvement_tips": []}


def get_structure_score(text: str) -> Dict[str, Any]:
    try:
        template = """
        You are an expert Resume Structure Evaluator.
        
        Analyze the provided resume text for visual hierarchy and logical flow, focusing on the text output, not the original PDF/DOCX layout.
        Criteria:
        - Logical section ordering (Expected: Contact -> Summary -> Experience -> Projects -> Skills -> Education)
        - Clear distinction between sections (implied by proper use of newlines)
        - Overall readability and flow.
        
        Resume Text:
        ---
        {text}
        ---
        
        Assign a score (0-100) and provide brief reasoning based on the above criteria.
        {format_instructions}
        """
        input_vars = ['text']
        chain = create_chain(StructureScoreResponse, template, input_vars)
        result = chain.invoke({"text": text})
        return result.model_dump()

    except Exception as e:
        return {"score": 0, "reasoning": f"Error during structure scoring: {str(e)}"}




def optimize_skills_section(current_skills, job_requirements, job_title: str) -> Dict[str, Any]:
    try:
        template = """
            Act as an experienced Technical Recruiter and Hiring Manager specializing in the role of {job_title}.

            Your task is to evaluate the candidate’s profile by comparing their current skills with the job requirements.

            Objectives:
            Identify **5 key skills the candidate should ADD** to better align with the role.
            - Include a balanced mix of **technical skills** and **soft skills**.
            - Focus on high-impact, industry-relevant, and role-specific skills.

            Context:
            - Current Skills: {current_skills}
            - Job Requirements: {job_requirements}

            Guidelines:
            - Be concise, realistic, and role-specific.
            - Avoid generic or obvious recommendations.
            - Justify each recommendation with a short, clear rationale.

            {format_instructions}
            """

        input_vars = ["current_skills", "job_requirements", "job_title"]
        chain = create_chain(SkillOptimizationResponse, template, input_vars)
        
        result = chain.invoke({
            "current_skills": current_skills,
            "job_requirements": job_requirements, 
            "job_title": job_title
        })
        return result.model_dump()

    except Exception as e:
        return {"skills_to_add": [], "error": f"Error optimizing skills: {str(e)}"}




def optimize_structure(resume_text: str) -> Dict[str, Any]:
    try:
        template = """
        Act as an expert Resume Structure Evaluator. Your sole focus is the **overall logical organization** and flow of the document based on the extracted text.

        **STRICT GUIDELINES:**
        1.  **IGNORE EXTRACTION ARTIFACTS:** Do not comment on specific formatting issues like missing spaces, hyphenation, or incorrect link formats. Assume these are errors caused by the PDF/DOCX text extraction process, not the candidate's fault.
        2.  **Focus Scope:** Evaluate only the following structural components:
            * **Section Order:** Is the order logical (e.g., Contact -> Summary -> Experience -> Skills -> Education)?
            * **Section Separation:** Are the sections clearly delineated?
            * **Consistency:** Is the visual flow consistent (e.g., proper use of capitalization for headings, date format consistency)?

        **CRITERIA:**
        * **A.** Logical and professional ordering of major sections.
        * **B.** Clear hierarchy and separation between major sections.
        * **C.** Appropriate depth and placement of the Skills and Education sections.

        Resume Text:
        ---
        {resume_text}
        ---

        Provide exactly 4 actionable key points for improvement based ONLY on the structural criteria above.
        If the structure is perfect, all 4 points must reflect positive feedback. Don't write long sentences Give answer is short Phrases.
        Set 'has_issues' to False only if no significant structural improvements are needed (A, B, C criteria met).

        {format_instructions}
        """
        input_vars = ["resume_text"]
        chain = create_chain(FeedbackResponse, template, input_vars)
        result = chain.invoke({"resume_text": resume_text})
        return result.model_dump()

    except Exception as e:
        return {"key_points": [f"Error optimizing structure: {str(e)}"], "has_issues": True}


def check_content(text: str) -> Dict[str, Any]:
    try:
        template = """
        Act as a Senior Resume Editor. Review the resume content strictly for **Quality, Clarity, and Professional Relevance**.

        **STRICT GUIDELINES:**
        1.  **Scope Focus:** Analyze the substance of the experience and project descriptions. Do NOT comment on Tone/Style, Grammar/Spelling (unless severe), or overall structure/layout.
        2.  **Prioritize Quantification (STAR/CAR):** Focus heavily on whether the bullet points describe accomplishments using metrics, numbers, and impact, rather than just listing duties. 
        3.  **Ignore Artifacts:** Ignore minor spacing, punctuation, or hyphenation errors; assume they are due to text extraction and focus on the *meaning of the content*.

        **Focus Criteria:**
        1.  **Clarity and Action:** Are the ideas clear, and does the text consistently use strong action verbs at the start of bullet points?
        2.  **Quantification (The "How Much"):** Does the candidate quantify their achievements with numbers, percentages, or scale?
        3.  **Relevance and Depth:** Is the content relevant to the implied target roles, and is there enough detail in the descriptions?
        4.  **Conciseness and Flow:** Is the content free from unnecessary jargon, redundancies, and passive language, ensuring a logical flow within and between bullet points?

        Don't Give answer in long sentences , Give answer in short Phrases.

        Resume Text:
        ---
        {text}
        ---

        Provide exactly 4 actionable key points for improvement based ONLY on the Content Quality criteria above.
        {format_instructions}
        """
        input_vars = ["text"]
        chain = create_chain(FeedbackResponse, template, input_vars)
        result = chain.invoke({"text": text})
        

        return result.model_dump()

    except Exception as e:
        return {"key_points": [f"Error checking content: {str(e)}"], "has_issues": True}


def check_tone_and_style(text: str) -> Dict[str, Any]:
    try:
        template = """
        Act as an expert copywriter and career coach. Review the resume text specifically for **Tone and Style** only.

        **STRICT GUIDELINES:**
        1.  **Scope Focus:** Analyze the use of language, formality, and voice. Do NOT comment on structural ordering, missing sections, or job-specific skills.
        2.  **Ignore Artifacts:** Ignore minor spacing, punctuation, or hyphenation errors; assume they are due to text extraction and focus on the *writing style*.

        **Focus Criteria:**
        1.  **Professionalism and Formality:** Is the language appropriate for a workplace setting?
        2.  **Active Voice Usage (vs Passive Voice):** Is the text driven by action verbs (e.g., "Led a team") rather than passive constructions (e.g., "The team was led by me")? 
        3.  **Impactful Language:** Does the writing use strong, engaging vocabulary (e.g., "Pioneered," "Accelerated") instead of weak adjectives (e.g., "Good," "Nice")?
        4.  **Consistency in Style:** Is the style (e.g., use of present/past tense, bullet format) uniform throughout the document?

        Don't Give answer in long sentences , Give answer in short Phrases.

        Resume Text:
        ---
        {text}
        ---

        Provide exactly 4 actionable key points for improvement based ONLY on the Tone and Style criteria above.
        {format_instructions}
        """
        input_vars = ["text"]
        chain = create_chain(FeedbackResponse, template, input_vars)
        result = chain.invoke({"text": text})
        
        return result.model_dump()

    except Exception as e:
        return {"key_points": [f"Error checking tone/style: {str(e)}"], "has_issues": True}
    
def get_professional_summary_suggestions(score:Union[int,float] ,resume_text: str, job_title: str) -> Dict[str, Any]:
    try:
        template = """
            You are an ATS Score Report Generator. Analyze the provided resume text against the requirements for a {job_title}.

            **STRICT GUIDELINES:**
            Ignore minor spacing, punctuation, or hyphenation errors; assume they are due to text extraction and focus on the *writing style*.

            Given the Score of the resume from 0-100 based on keyword match, skill relevance, and standard formatting adherence.
            Provide EXACTLY 4 highly critical, actionable points for improvement. Categorize these points by Technical Skills, Education/Training Gaps, Formatting/Structure Issues, and Missing Key Sections.
            
             ---
            Resume  Score: {score}
            Job Title: {job_title}
            Resume Text: 
            {resume_text}
            ---

            Your response must be concise and formatted for a critical summary.
            Don't Give answer in long sentences , Give answer in short Phrases.
            {format_instructions} (Using a Pydantic model with 'score' and 'key_points: List[str]')
            """
        input_vars = ["score","resume_text", "job_title", "level"]
        chain = create_chain(ProfessionalSummaryResponse, template, input_vars)
        result = chain.invoke({"score":score,"resume_text": resume_text, "job_title": job_title})
        return result.model_dump()
    except Exception as e:
        return {"options": [], "error": f"Error generating summary: {str(e)}"}