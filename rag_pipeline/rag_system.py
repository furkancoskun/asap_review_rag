import ollama 
from ollama_tools.check_ollama import check_ollama_availability

OLLAMA_MODEL_NAME = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama API endpoint

class RAGSystem:
    def __init__(self, retriever_instance):
        self.retriever = retriever_instance
        self.ollama_model_name = OLLAMA_MODEL_NAME
        try:
            self.llm_client = ollama.Client(host=OLLAMA_BASE_URL)
            # Perform a quick check during initialization
            if not check_ollama_availability(self.ollama_model_name, OLLAMA_BASE_URL):
                 raise ConnectionError(f"Ollama model '{self.ollama_model_name}' not available or Ollama not running properly.")
            print(f"RAG System initialized with Ollama model: {self.ollama_model_name}")
        except Exception as e:
            print(f"Failed to initialize Ollama client for RAG system: {e}")
            self.llm_client = None


    def _construct_prompt(self, query, contexts, prompt_template):
        context_str = "\n---\n".join([f"Context Snippet {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        # For some models, explicitly stating the role of context might be helpful
        return prompt_template.format(query=query, contexts=context_str)

    def generate_answer(self, paper_id, original_query, prompt_template, top_k_retrieval=5):
        """
        Retrieves context and generates an answer for a given paper_id and query using Ollama.
        Args:
            paper_id (str): The ID of the paper.
            original_query (str): The high-level query/question from the user/evaluation.
            prompt_template (str): A string template for the LLM prompt.
            top_k_retrieval (int): Number of chunks to retrieve.
        Returns:
            tuple: (generated_answer_text, retrieved_contexts_list, full_llm_prompt)
                   Returns (None, [], "") if retrieval fails or generation fails.
        """
        if not self.llm_client:
            print("Ollama client not initialized. Cannot generate answer.")
            return None, [], ""

        retrieved_df = self.retriever.retrieve_from_paper(original_query, paper_id, top_k=top_k_retrieval)

        if retrieved_df.empty:
            return None, [], ""

        retrieved_contexts_list = retrieved_df['text'].tolist()

        full_llm_prompt = self._construct_prompt(original_query, retrieved_contexts_list, prompt_template)

        try:
            response = self.llm_client.chat(
                model=self.ollama_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing academic peer reviews."},
                    {"role": "user", "content": full_llm_prompt}
                ],
                # stream=False,
                options={ 
                    "temperature": 0.3,
                }
            )
            generated_answer_text = response['message']['content'].strip()
            return generated_answer_text, retrieved_contexts_list, full_llm_prompt
        except Exception as e:
            print(f"Error during Ollama LLM generation for paper {paper_id}: {e}")
            return None, retrieved_contexts_list, full_llm_prompt
