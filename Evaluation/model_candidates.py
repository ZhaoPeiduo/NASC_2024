import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class JapaneseLLMTemplate:
    def __init__(self):
        raise NotImplementedError("This is an interface class for further Model subclasses and shall not be initialized.")
    
    def build_prompt(self, question, options):
        raise NotImplementedError("This is an interface class for further Model subclasses and method is not implemented.")
    
    def generate_answer(self, question, options):
        raise NotImplementedError("This is an interface class for further Model subclasses and method is not implemented.")

class StableLM_Gamma_7b_Half(JapaneseLLMTemplate):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/japanese-stablelm-instruct-gamma-7b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/japanese-stablelm-instruct-gamma-7b",
            torch_dtype=torch.half
        )
        self.model.eval()
        self.device= 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.model.to(self.device)

    def build_prompt(self, user_query, inputs="", sep="\n\n### "):
        sys_msg = "あなたは国語教師です。以下の問題を考えて、要求を適切に満たす応答を書きなさい。"
        p = sys_msg
        roles = ["指示", "応答"]
        msgs = [": \n" + user_query, ": \n"]
        if inputs:
            roles.insert(1, "入力")
            msgs.insert(1, ": \n" + inputs)
        for role, msg in zip(roles, msgs):
            p += sep + role + msg
        return p

    def generate_answer(self, question, options):
        # question = question.replace("___", "[MASK]")
        user_inputs = {
            "user_query":'文法を基づいて、最もよい選択肢のアルファベットを一つ選びなさい。',
            "inputs": f"問題：{question} 選択肢：{options}"
        }

        prompt = self.build_prompt(**user_inputs)

        # print(prompt)
        input_ids = self.tokenizer.encode(
            prompt, 
            add_special_tokens=True, 
            return_tensors="pt"
        )

        attention_mask = torch.ones_like(input_ids).to(self.device)  # Create attention mask with all 1s for non-padding tokens

        tokens = self.model.generate(
            input_ids.to(device=self.model.device),
            attention_mask=attention_mask, 
            max_new_tokens=16,
            # temperature=0.1,
            repetition_penalty=1.1,
            # top_p=0.95,
            do_sample=False,
            pad_token_id=2,
            eos_token_id=2
        )

        answer = self.tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return answer
        
    def generate_explanation(self, question, options, answer):
        user_inputs = {
            "user_query":f'どうして{answer}選びましたか？',
            "inputs": f"問題：{question} 選択肢：{options}"
        }

        prompt = self.build_prompt(**user_inputs)

        input_ids = self.tokenizer.encode(
            prompt, 
            add_special_tokens=True, 
            return_tensors="pt"
        )

        attention_mask = torch.ones_like(input_ids).to(self.device)  # Create attention mask with all 1s for non-padding tokens
        
        tokens = self.model.generate(
            input_ids.to(device=self.model.device),
            attention_mask=attention_mask, 
            max_new_tokens=80,
            # temperature=0.1,
            repetition_penalty=1.1,
            # top_p=0.95,
            do_sample=False,
            pad_token_id=2,
            eos_token_id=2
        )

        explanation = self.tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return f'{answer}　説明：{explanation}'

class ELYZA_Llama_7b_Half(JapaneseLLMTemplate):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-instruct")
        self.model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-instruct", torch_dtype=torch.half)
        self.model.eval()
        self.device= 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.model.to(self.device)

    def build_prompt(self, question, options):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = "あなたは国語教師です。文法を基づいて、最もよい選択肢のアルファベットを一つだけ選びなさい"
        text = f"問題：{question} 選択肢：{options}"
        prompt = f"{self.tokenizer.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{text}{E_INST}"
        return prompt
    
    def generate_answer(self, question, options):
        prompt = self.build_prompt(question, options)
        
        with torch.no_grad():
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=64,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        return output
        
class GPT_NeoX_4b(JapaneseLLMTemplate):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft")
        self.model.eval()
        self.device= 'cpu'
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
    
    def build_prompt(self, question, options):
        prompt = [
        {
            "speaker": "ユーザー",
            "text": "あなたは国語教師です。以下の問題を考えて、要求を適切に満たす応答を書きなさい。"
        },
        {
            "speaker": "システム",
            "text": "わかりました。どのような問題ですか？"
        },
        {
            "speaker": "ユーザー",
            "text": f"文法を基づいて、最もよい選択肢のアルファベットを一つ選びなさい。問題：{question} 選択肢：{options}"
        }]
        prompt = [
            f"{uttr['speaker']}: {uttr['text']}"
            for uttr in prompt
        ]
        prompt = "<NL>".join(prompt)
        prompt = (
            prompt
            + "<NL>"
            + "システム: "
        )
        return prompt
    
    def generate_answer(self, question, options):
        prompt = self.build_prompt(question, options)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                do_sample=False,
                max_new_tokens=32,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
        output = output.replace("<NL>", "\n")
        return output
