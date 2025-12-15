from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import torch
from typing import Union, List, Optional

load_dotenv()

class SemanticModel:
    _instance: Optional['SemanticModel'] = None 
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: str):
        if SemanticModel._instance is not None:
            raise Exception("This class is a Singleton! Use SemanticModel.get_instance() instead.")
        
        self.model_name = model_name
        
        print(f"Loading Sentence Transformer Model: {self.model_name}...")
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            SemanticModel._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            raise RuntimeError(f"Could not load Sentence Transformer model. Check model name and internet connection.")
        
    @staticmethod
    def get_instance() -> 'SemanticModel':
        if SemanticModel._instance is None:
            model_name = os.environ.get("MODEL_NAME")
            if not model_name:
                raise EnvironmentError("MODEL_NAME environment variable is not set. Cannot initialize the scoring model.")
            
            SemanticModel._instance = SemanticModel(model_name)
            
        return SemanticModel._instance

    def score(self, job_text: Union[str, List[str]], resume_text: Union[str, List[str]]) -> float:
        try:
            model = SemanticModel._model 
            if model is None:
                 raise RuntimeError("Semantic Model is not loaded.")
            
            
            if isinstance(job_text, str) and isinstance(resume_text, str):
                embeddings = model.encode([job_text, resume_text], convert_to_tensor=True, show_progress_bar=False)
                emb1, emb2 = embeddings[0], embeddings[1]
            
            elif isinstance(job_text, list) and isinstance(resume_text, list):
                if not job_text or not resume_text: return 0.0
                
              
                emb1 = model.encode(job_text, convert_to_tensor=True, show_progress_bar=False).mean(dim=0)
                emb2 = model.encode(resume_text, convert_to_tensor=True, show_progress_bar=False).mean(dim=0)
            
            else:
                return 0.0

            similarity = util.cos_sim(emb1, emb2).item()
            

            score_value = round(similarity * 100, 2)
            return max(0.0, min(100.0, score_value))
            
        except RuntimeError as e:
            print(f"Semantic Model Internal Error: {e}")
            return 0.0
        except Exception as e:
            print(f"Scoring Error: {e}")
            return 0.0