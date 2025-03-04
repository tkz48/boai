use either::Either;
use llm_client::clients::types::{LLMClientCompletionRequest, LLMClientCompletionStringRequest};

use super::types::{FillInMiddleFormatter, FillInMiddleRequest};

pub struct CodeLlamaFillInMiddleFormatter;

impl CodeLlamaFillInMiddleFormatter {
    pub fn new() -> Self {
        Self
    }
}

impl FillInMiddleFormatter for CodeLlamaFillInMiddleFormatter {
    fn fill_in_middle(
        &self,
        request: FillInMiddleRequest,
    ) -> Either<LLMClientCompletionRequest, LLMClientCompletionStringRequest> {
        // format is
        // <PRE> {prefix} <SUF>{suffix} <MID>
        // https://ollama.ai/library/codellama
        let prefix = request.prefix();
        let suffix = request.suffix();
        let response = format!("<PRE> {prefix} <SUF>{suffix} <MID>");
        // log the response here
        let string_request =
            LLMClientCompletionStringRequest::new(request.llm().clone(), response, 0.0, None)
                .set_stop_words(request.stop_words())
                .set_max_tokens(512);
        Either::Right(string_request)
    }
}
