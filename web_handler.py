from typing import List, Dict, Any
import logging
import requests
from bs4 import BeautifulSoup
from model_manager import ModelManager
from googlesearch import search # Import specific function

class WebHandler:
    def __init__(self, model_manager: ModelManager, max_search_results: int = 2, context_token_limit: int = 4000, 
                 summary_max_tokens: int = 200, recency_filter: str = 'month'):
        """Initializes the WebHandler.

        Args:
            model_manager (ModelManager): The ModelManager instance.
            max_search_results (int): Maximum number of search result pages to process.
            context_token_limit (int): Token limit for context sent to LLM for summarization.
            summary_max_tokens (int): Max tokens for the generated summary.
            recency_filter (str): Filter for search results by recency. Options:
                                 'day' - past day, 'week' - past week, 'month' - past month, 
                                 'year' - past year, None - no filter
        """
        self.model_manager = model_manager
        self.max_search_results = max_search_results
        self.context_token_limit = context_token_limit
        self.summary_max_tokens = summary_max_tokens
        
        # Set up recency filter mapping
        self._recency_map = {
            'day': 'qdr:d',
            'week': 'qdr:w',
            'month': 'qdr:m',
            'year': 'qdr:y',
            None: None
        }
        self.recency_filter = self._recency_map.get(recency_filter, 'qdr:m')  # Default to month if invalid
        
        logging.info(f"WebHandler initialized with recency filter: {recency_filter if recency_filter else 'none'}")

    def _fetch_web_content(self, query: str) -> List[str]:
        """Performs internet search and returns cleaned text content from results."""
        logging.info(f"Web search attempt: {query}")
        search_results_text = []
        urls_processed = set() # Use set for faster lookup

        try:
            # Fetch search results (limit number fetched initially)
            num_to_fetch = self.max_search_results + 2 # Fetch slightly more to account for potential failures
            
            # Log with recency information
            recency_info = ""
            if self.recency_filter:
                for period, tbs in self._recency_map.items():
                    if tbs == self.recency_filter and period:
                        recency_info = f" (from past {period})"
                        break
            
            logging.info(f"Google search for '{query}'{recency_info} (fetching up to {num_to_fetch})...")
            
            # Simplified googlesearch call with recency filter
            try:
                # Use the recency filter if set
                search_generator = search(query, pause=2.0, tbs=self.recency_filter) 
                # Convert generator to list cautiously, limiting the initial fetch
                fetched_urls = []
                # Try fetching a bit more than needed, handle potential StopIteration
                try:
                     for i, url in enumerate(search_generator):
                          fetched_urls.append(url)
                          if i >= num_to_fetch - 1: 
                               break
                except Exception as search_iter_err:
                     # Log error during iteration, but proceed if we got some URLs
                     logging.warning(f"Error during googlesearch iteration: {search_iter_err}")

            except Exception as google_search_err:
                 # Catch errors during the initial search() call itself
                 logging.error(f"Error calling googlesearch library for '{query}': {google_search_err}", exc_info=True)
                 # If the search call itself fails, return empty list immediately
                 return []
                    
            # If we have no URLs after trying, gracefully handle it
            if not fetched_urls:
                logging.warning(f"Search for '{query}' returned no results.")
                return []

            # Process the fetched URLs
            count_success = 0
            for url in fetched_urls:
                if url in urls_processed:
                    continue # Skip already processed URLs
                if count_success >= self.max_search_results:
                    break # Stop once we have enough successful results

                urls_processed.add(url)
                try:
                    logging.info(f"Processing URL: {url}")
                    response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                    response.raise_for_status() # Check for HTTP errors
                    response.encoding = response.apparent_encoding # Detect encoding
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Remove unwanted tags
                    for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                        element.decompose()

                    # Extract and clean text
                    text = soup.get_text(separator=' ', strip=True)
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    cleaned_text = ' '.join(chunk for chunk in chunks if chunk)

                    if cleaned_text:
                        # Append a reasonable amount of text
                        search_results_text.append(cleaned_text[:2000])
                        count_success += 1 # Increment success count here
                        logging.info(f"URL {url} processed successfully (Content length: {len(cleaned_text)}). Got {count_success} results so far.")
                    else:
                         logging.warning(f"URL {url} yielded no text content after cleaning.")

                except requests.exceptions.Timeout:
                     logging.warning(f"Timeout fetching URL {url}")
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Request error for URL {url}: {str(e)}")
                except Exception as e:
                    logging.warning(f"Error processing URL {url}: {str(e)}", exc_info=True)
                # Ensure loop continues even if one URL fails

        # This outer exception catch is now primarily for unexpected issues
        # outside the specific library calls handled above.
        except Exception as e:
            logging.error(f"Unexpected error during _fetch_web_content for '{query}': {e}", exc_info=True)
            # Return empty list in case of unexpected errors as well
            return []

        if not search_results_text:
             logging.warning(f"Web search for '{query}' returned no usable content after processing URLs.")

        return search_results_text

    def _summarize_text(self, query: str, text_content: List[str], language_hint: str = 'en') -> str:
        """Summarizes text content using an LLM to answer the original query in the specified language."""
        if not text_content:
            return "관련 웹 정보를 찾을 수 없습니다."

        # Combine search results, respecting token limit
        context = "\n\n---\n\n".join(text_content)
        # Simple truncation based on character count as a proxy for tokens
        # Adjust multiplier if needed, e.g., 4 chars/token average
        context_for_llm = context[:self.context_token_limit * 4]
        logging.info(f"Attempting text summarization (Context length: {len(context_for_llm)} chars, Language: {language_hint})...")

        # Determine language for the prompt
        language_name = "Korean" if language_hint == 'ko' else "English"
        system_prompt = (
            f"You are a helpful assistant. Summarize the following text context to directly answer the user's original question. "
            f"Provide a concise and relevant answer based *only* on the provided text. "
            f"If the text doesn't answer the question, state that. "
            f"Please respond in {language_name}."
        )

        try:
            llm_result = self.model_manager.call_llm(
                task_type='summarization',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original Question: {query}\n\nContext:\n{context_for_llm}"}
                ],
                temperature=0.2,
                max_tokens=self.summary_max_tokens,
            )

            if not llm_result["success"]:
                # Log error, handled in call_llm
                summary = None
            else:
                summary = llm_result["content"]

            if not summary:
                logging.warning("LLM summary result is empty or failed.")
                # Fallback: provide first part of the first result
                fallback_message = "검색 결과를 요약하는 데 실패했습니다." if language_hint == 'ko' else "Failed to summarize search results."
                return f"{fallback_message} 첫 번째 결과 일부: \n{text_content[0][:300]}..."

            logging.info("Text summarization successful.")
            return summary

        except Exception as e:
            # This catch might be less likely now as call_llm handles API errors,
            # but keep it for unexpected issues during message preparation etc.
            logging.error(f"Unexpected error during summarization preparation or fallback: {e}", exc_info=True)
            # Fallback: provide first part of the first result
            fallback_message = "텍스트 요약 중 오류가 발생했습니다." if language_hint == 'ko' else "An error occurred during text summarization."
            return f"{fallback_message} 첫 번째 결과 일부: \n{text_content[0][:300]}..."

    def perform_web_search_and_summarize(self, query: str, language_hint: str = 'en') -> Dict[str, Any]:
        """Performs web search and summarizes the results in the specified language."""
        # 1. Fetch web content
        search_results = self._fetch_web_content(query)

        if not search_results:
             fallback_message = "웹 검색 중 오류가 발생했거나 관련 정보를 찾을 수 없습니다." if language_hint == 'ko' else "An error occurred during web search or no relevant information was found."
             return {
                "success": False,
                "result": fallback_message,
                "raw_content": []
            }

        # 2. Summarize the fetched content, passing the hint
        summary = self._summarize_text(query, search_results, language_hint=language_hint)

        return {
            "success": True,
            "result": summary,
            "raw_content": search_results # Return raw content for potential later use
        } 