import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json

class LLM():
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                             load_in_8bit=True)
        
        self.task = """
        Given the following HTML query, extract the following information in a JSON format regarding a news article if the fields exist otherwise use "NULL" to show a missing field. The information may not be explicitly present as text in all cases and might be part of an html element's attributes. For example the headline or summary might be contained with the alt attribute of an img tag or the publication date might have to be extrapolated from the url. Do not including any opening or closing text, only include the JSON object. Do not make anything up! 
        Fields: ```json
        {
            news headline: // the title or headline of a news story, 
            url to article: // url that links to the article, 
            news summary: // a short description or summary of the news article, 
            date/time of publication: // when the article was published, 
            estimated read time: // estimated time to read this article, 
            author: // person or persons who write the article
        }
        ```
        """.strip()
        self.system = "You are a web extraction bot that extracts information from html in a JSON format"
        
    def update_system(self, system: str):
        """Update the System Prompt of extraction prompt template

        Args:
            system (str): New system prompt to be used
        """
        self.system = system.strip()
    
    def update_task(self, task: str):
        """Update the task description of the extraction prompt template

        Args:
            task (str): New task description of the extraction prompt
        """
        self.system = task.strip()
        
    def generate_prompt(self, html: str)-> str:
        """Generate a prompt in accordance with the specific LLM's prompt template using the System Prompt, Task Description, and HTML snipper from which information must be extracted 

        Args:
            html (str): HTML snippet from which information should be extracted
        """
        # Define in subclass
        pass
    
    def extract(self, html: "list[str]", max_new_tokens=256):
        """Extracts news headlines, article url, news summary, publication date/time, estimated read time, author from each html snippet with the list html
        Args:
            html (List[str]): list of html snippets from which extraction must be performed
            max_new_tokens (int): Maximum new tokens to be generated from LLM (max size of JSON as string). Defaults to 256.

        Returns:
            List[dict or str]: Returns a list of dictionaries with extracted information, or a string error message if extraction failed or JSON not present
        """
        prompts = []
        for tag in html:
            prompts.append(self.generate_prompt(tag))
            
        
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            # outputs = self.tokenizer.decode(output)
            out = []
            for o in outputs:
                output = self.tokenizer.decode(o)
                matches = re.findall(r'\{.*?\}', output, re.DOTALL)
                if matches:
                    try:
                        out.append(json.loads(matches[-1]))
                    except json.JSONDecodeError:
                        print("The extracted text is not a valid JSON object.")
                        out.append("ERROR: JSON DECODE FAILED")
                else:
                    out.append("ERROR: NO JSON")
            return out

class Llama2(LLM, ):
    def __init__(self, model_name='meta-llama/Llama-2-7b-chat-hf'):
        super().__init__(model_name=model_name) 
    
    def generate_prompt(self, html):
        return f'[INST]<<SYS>>{self.system}<</SYS>>\n###Instruction:\n{self.task}\n\n###Query:\n{html}[/INST]'
    
class Mistral(LLM):
    def __init__(self, model_name='cognitivecomputations/dolphin-2.6-mistral-7b'):
        # 'cognitivecomputations/dolphin-2.6-mistral-7b'
        # Open-Orca/Mistral-7B-OpenOrca
        super().__init__(model_name=model_name) 
    
    def generate_prompt(self, html):
        return f'<|im_start|>system\n{self.system}<|im_end|>\n<|im_start|>user\n###Instruction:\n{self.task}\n\n###Query:\n{html}<|im_end|>\n<|im_start|>assistant'
    
class StableBeluga(LLM):
    def __init__(self, model_name='stabilityai/StableBeluga-7B'):
        super().__init__(model_name=model_name) 
    
    def generate_prompt(self, html):
        return f"{self.system}### User: Instruction:\n{self.task}\n\nQuery:\n{html}\n\n### Assistant:\n"
    
class OpenChat(LLM):
    def __init__(self, model_name='openchat/openchat-3.5-1210'):
        super().__init__(model_name=model_name) 
    
    def generate_prompt(self, html):
        return f"GPT4 Correct User: ###Instruction:\n{self.task}\n\n###Query:\n{html}<|end_of_turn|>GPT4 Correct Assistant:"
    
class Falcon(LLM):
    def __init__(self, model_name='tiiuae/falcon-7b'):
        super().__init__(model_name=model_name) 
    
    def generate_prompt(self, html):
        return f"{self.system}### User: Instruction:\n{self.task}\n\nQuery:\n{html}\n\n### Assistant:\n"