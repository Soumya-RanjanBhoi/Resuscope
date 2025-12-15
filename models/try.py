
import pdfplumber
file = "D:/project-1/models/CV.pdf"
from rewriter import get_required_skills
from extractor import extract_all_skills
from matcher import score

descr= """
We are seeking a highly skilled Data Scientist to turn raw data into meaningful insights that drive business decisions and optimize operations. The ideal candidate will have a strong foundation in statistical analysis, machine learning, and programming. You will work closely with cross-functional teams to identify opportunities, process large structured and unstructured datasets, and build predictive models that solve complex business problems.​

Key Responsibilities
Data Collection & Cleaning: Identify relevant data sources and automate collection processes. Preprocess, cleanse, and validate the integrity of data to ensure it is ready for analysis.​

Modeling & Analysis: Develop and implement predictive models, machine learning algorithms, and statistical techniques to mine large datasets for trends and patterns.​

Experimentation: Design and execute analytical experiments (A/B testing) to test hypotheses and improve model performance or product features.​

Collaboration: Partner with product, engineering, and business teams to understand requirements and integrate data-driven insights into production systems.​

Visualization & Reporting: Present complex findings to both technical and non-technical stakeholders using clear data visualizations, dashboards, and reports.​

Continuous Improvement: Stay updated with emerging technologies and methodologies in AI/ML to continuously enhance data collection procedures and model accuracy.​

Required Skills & Qualifications
Education: Bachelor’s or Master’s degree in Computer Science, Statistics, Mathematics, Engineering, or a related quantitative field.​

Programming: Proficiency in Python or R for statistical analysis and modeling. Strong command of SQL for data extraction and management.​

Machine Learning: specific experience with building and deploying models using frameworks such as Scikit-Learn, TensorFlow, or PyTorch.​

Statistical Analysis: Deep understanding of probability distributions, regression analysis, and hypothesis testing.​

Data Manipulation: Experience with data manipulation libraries (e.g., Pandas, NumPy) and working with big data platforms (e.g., Spark, Hadoop).​

Preferred Qualifications
Deep Learning & NLP: Familiarity with advanced techniques such as Natural Language Processing (NLP) and transformer architectures (e.g., BERT, GPT) is highly desirable for complex text analysis tasks.​

Cloud Platforms: Hands-on experience with cloud computing services like AWS, Google Cloud Platform (GCP), or Azure for training and deploying models.​

Visualization Tools: Expertise in business intelligence tools like Tableau, Power BI, or Matplotlib/Seaborn for creating interactive dashboards.​

Deployment: Experience with containerization (Docker) and API development (FastAPI/Flask) to serve models in production environments.
"""
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
    

text = extract_text_from_pdf(file)
res_skill = extract_all_skills(text)
req_skill = get_required_skills(descr)

# Normalize outputs in case extractor/rewriter return unexpected types
if not isinstance(res_skill, dict):
    print("Warning: extract_all_skills returned unexpected type; normalizing to empty skill structure")
    res_skill = {'skill_sets': [], 'technical_skills': [], 'soft_skills': []}

if not isinstance(req_skill, dict):
    print("Warning: get_required_skills returned unexpected type; normalizing to empty requirement structure")
    req_skill = {'technical_skills': [], 'soft_skills': []}


print(text)
print()
print(req_skill)



print()

print(res_skill)

print()
req_tech_skill = req_skill['technical_skills']
req_soft_skill = req_skill['soft_skills']

res_tech_skill = [ index.get('skills',[]) for index in res_skill.get('skill_sets', []) if index.get('category') == "TECHNICAL" ]
res_soft_skill = [ index.get('skills',[]) for index in res_skill.get('skill_sets', []) if index.get('category') == "SOFT" ]


print("Technical Req Skill:",req_tech_skill)
print("Soft req skill:", req_soft_skill)

print()

print("Resume tech skill:", res_tech_skill)
print('resume soft skill:', res_soft_skill)

print()

tech_input = res_tech_skill[0] if res_tech_skill and len(res_tech_skill) > 0 else []
soft_input = res_soft_skill[0] if res_soft_skill and len(res_soft_skill) > 0 else []

tech_score = score(req_tech_skill, tech_input)
soft_score = score(req_soft_skill, soft_input)

total_score = 0.85*tech_score + 0.15* soft_score
print("Tech score:", tech_score)
print('soft score:',soft_score)
print("Total Score:", total_score)