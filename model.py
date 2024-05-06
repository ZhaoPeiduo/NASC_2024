import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from progress_manager import ProgressManager


class JapaneseLLM:
    def __init__(self, manager:ProgressManager):
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
        self.manager = manager

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

    async def generate_answer(self, question, options):
        print("Generating answers...")
        # question = question.replace("___", "[MASK]")
        user_inputs = {
            "user_query":'文法を基づいて、最もよい選択肢のアルファベットを一つ選びなさい。',
            "inputs": f"問題：{question} 選択肢：{options}"
        }

        prompt = self.build_prompt(**user_inputs)
        await self.manager.update_progress_and_send(5)

        # print(prompt)
        input_ids = self.tokenizer.encode(
            prompt, 
            add_special_tokens=True, 
            return_tensors="pt"
        )

        attention_mask = torch.ones_like(input_ids).to(self.device)  # Create attention mask with all 1s for non-padding tokens
        await self.manager.update_progress_and_send(5)
        max_length = max([len(x) for x in options])

        tokens = self.model.generate(
            input_ids.to(device=self.model.device),
            attention_mask=attention_mask, 
            max_new_tokens=max_length+4,
            # temperature=0.1,
            repetition_penalty=1.1,
            # top_p=0.95,
            do_sample=False,
            pad_token_id=2,
            eos_token_id=2
        )
        await self.manager.update_progress_and_send(20)

        answer = self.tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return answer
        
    async def generate_explanation(self, question, options, answer):
        print("Generating explanations...")
        user_inputs = {
            "user_query":f'どうして{answer}選びましたか？',
            "inputs": f"問題：{question} 選択肢：{options}"
        }

        prompt = self.build_prompt(**user_inputs)
        await self.manager.update_progress_and_send(20)

        input_ids = self.tokenizer.encode(
            prompt, 
            add_special_tokens=True, 
            return_tensors="pt"
        )

        attention_mask = torch.ones_like(input_ids).to(self.device)  # Create attention mask with all 1s for non-padding tokens
        await self.manager.update_progress_and_send(20)
        
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
        await self.manager.update_progress_and_send(20)
        return f'説明：{explanation}'

