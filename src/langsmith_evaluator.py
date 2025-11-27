from typing import Any, Dict, Optional
from src.logger import logger
from src.config import Config
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_groq import ChatGroq


class LangSmithEvaluator:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.api_key = Config.LANGSMITH_API_KEY
        self.project_name = Config.LANGSMITH_PROJECT
        self.endpoint = Config.LANGSMITH_ENDPOINT
        
        if self.enabled:
            try:
                import os
                os.environ["LANGSMITH_API_KEY"] = self.api_key
                os.environ["LANGSMITH_PROJECT"] = self.project_name
                os.environ["LANGSMITH_ENDPOINT"] = self.endpoint
            except Exception as e:
                self.enabled = False
    
    def create_dataset(self, 
                        name: str, 
                        description: str,
                        dataset_id: str,
                        input_output_data: list,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False
        
        try:
            client = Client()
            dataset = client.create_dataset(dataset_name=name, 
                                            description=description)
            
            dataset_id = dataset.id
            for input_prompt, output_answer in input_output_data:
                client.create_example(dataset_id=dataset_id,
                                        inputs={"question": input_prompt},
                                        outputs={"answer": output_answer},
                                        metadata=metadata or {}
                                    )
            return True
        
        except Exception as e:
            print(f"Error adding example to LangSmith dataset: {e}")
            return False

    @staticmethod
    def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        # Use the model name string for Groq models as required by openevals
        evaluator = create_llm_as_judge(
            prompt=CORRECTNESS_PROMPT,
            model="groq:llama-3.3-70b-versatile",  # Pass model name string, not ChatGroq object
            feedback_key="correctness",
        )
        eval_result = evaluator(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs
        )
        return eval_result
        
    def evaluate_response(self, target):
        client = Client()
        experiment_results = client.evaluate(
                                            target,
                                            data="transformer_qa_dataset",
                                            evaluators=[LangSmithEvaluator.correctness_evaluator],
                                            experiment_prefix="experiment-quickstart-fixed-ant-71",
                                            max_concurrency=2,
                                        )
        return experiment_results


evaluator = LangSmithEvaluator(enabled=True)



question = [
            "What is the core architectural innovation proposed in the paper?",
            "How many layers are used in the encoder and decoder stacks of the base Transformer model?",
            "What is the dimensionality of the model embeddings in the base Transformer?",
            "What are the two sub-layers present in each encoder layer?",
            "What additional sub-layer is present in each decoder layer compared to the encoder?",
            "What is the purpose of masking in the decoder self-attention?",
            "How is Scaled Dot-Product Attention computed?",
            "Why is the dot-product scaled by the square root of dk in the attention formula?",
            "How many attention heads are used in the base Transformer model?",
            "What are the dimensions of each attention head in the base model?",
            "What is the structure of the position-wise feed-forward network?",
            "Why are positional encodings required in the Transformer?",
            "What type of positional encoding is used in the final model?",
            "On which datasets was the Transformer trained for machine translation?",
            "What optimizer was used to train the Transformer models?",
            "How many training steps and how long did the base model take to train?",
            "What BLEU score did the Transformer (big) achieve on WMT 2014 English–German?",
            "What BLEU score did the Transformer (big) achieve on WMT 2014 English–French?",
            "What other NLP task was used to demonstrate the Transformer’s generalization ability?",
            "Does the Transformer model rely entirely on attention mechanisms instead of recurrent or convolutional layers?"
            ]

correct_answers = ["The Transformer overcomes the inherently sequential computation of recurrent neural networks, which prevents parallelization and slows down training on long sequences.",
                    "The core innovation is the Transformer architecture, which relies entirely on attention mechanisms and removes both recurrence and convolution.",
                    "Both the encoder and the decoder use 6 identical layers.",
                    "The embedding dimensionality d_model in the base Transformer is 512.",
                    "Each encoder layer consists of a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.",
                    "Each decoder layer adds an encoder–decoder multi-head attention sub-layer in addition to self-attention and feed-forward layers.",
                    "Masking prevents positions from attending to future positions so that predictions depend only on past tokens during auto-regressive decoding.",
                    "Scaled Dot-Product Attention is computed as softmax(QK^T / sqrt(dk)) multiplied by V.",
                    "The scaling by sqrt(dk) prevents large dot-product values from pushing the softmax into regions with very small gradients.",
                    "The base Transformer model uses 8 attention heads.",
                    "Each attention head has dk = 64 and dv = 64 dimensions.",
                    "The feed-forward network consists of two linear layers with a ReLU activation in between and an inner dimension of 2048.",
                    "Positional encodings are required because the Transformer has no recurrence or convolution to capture token order.",
                    "The model uses fixed sinusoidal positional encodings based on sine and cosine functions.",
                    "The Transformer was trained on WMT 2014 English–German and WMT 2014 English–French datasets.",
                    "The Adam optimizer with β1 = 0.9, β2 = 0.98, and ε = 10^-9 was used for training.",
                    "The base model was trained for 100,000 steps and took about 12 hours on 8 NVIDIA P100 GPUs.",
                    "The Transformer (big) achieved a BLEU score of 28.4 on WMT 2014 English–German.",
                    "The Transformer (big) achieved a BLEU score of 41.8 on WMT 2014 English–French.",
                    "Yes, the Transformer model rely on attention mechanisms instead of recurrent or convolutional layers."
                ]


input_output_data = list(zip(question, correct_answers))


def evaluate_live_question_and_log(user_question: str, llm_response: str):

    reference_answer = None
    for q, a in input_output_data:
        if q.strip().lower() == user_question.strip().lower():
            reference_answer = a
            break
    if reference_answer:
        inputs = {"question": user_question}
        outputs = {"answer": llm_response}
        reference_outputs = {"answer": reference_answer}
        eval_result = LangSmithEvaluator.correctness_evaluator(inputs, outputs, reference_outputs)
        print("User Question:", user_question)
        print("LLM Response:", llm_response)
        print("Reference Answer:", reference_answer)
        print("LangSmith Evaluation Result:", eval_result)
        return eval_result
    else:
        print("No reference answer found for this question. Evaluation not required.")
        return None











