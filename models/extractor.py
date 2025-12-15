import os
import io
from typing import List, Union, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import pdfplumber
import docx


load_dotenv()

def extract_text_from_pdf(file) -> str:

    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_docx(file) -> str:
    try:
        doc = docx.Document(file)
        return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""


class SkillCategory(BaseModel):
    category: str = Field(..., description="Skill category (e.g., TECHNICAL, SOFT, LANGUAGE).")
    skills: List[str] = Field(..., description="List of specific skills relevant to this category.")


class CandidateSkills(BaseModel):
    skill_sets: List[SkillCategory] = Field(..., description="List of all skill categories extracted from the resume text.")


def extract_all_skills(text: str) -> Union[Dict[str, Any], str]:
    if not text or len(text) < 50:
        return {"skill_sets": [], "error": "Resume text is too short for skill extraction."}
        
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        
        parser = PydanticOutputParser(pydantic_object=CandidateSkills)

        prompt = PromptTemplate(
            template="""
            You are an expert resume parser. Your task is to extract all skills from the provided text.
            Group the skills logically into categories like TECHNICAL, SOFT in the Specified Section of the Text , Do not go into the Project section to extract.
            
            Resume Text:
            ---
            {text}
            ---
            
            {format_instructions}
            """,
            input_variables=['text'],
            validate_template=True, 
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )

        chain = prompt | model | parser

        result = chain.invoke({'text': text})
        
        return result.model_dump() 

    except Exception as e:
        print(f"Error during AI skill extraction: {e}")
        return {"skill_sets": [], "error": f"AI extraction failed: {str(e)}"}