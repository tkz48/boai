use super::types::{FillInMiddleFormatter, FillInMiddleRequest};
use either::Either;
use llm_client::clients::types::{LLMClientCompletionRequest, LLMClientCompletionStringRequest};

pub struct DeepSeekFillInMiddleFormatter;

impl DeepSeekFillInMiddleFormatter {
    pub fn new() -> Self {
        Self
    }
}

impl FillInMiddleFormatter for DeepSeekFillInMiddleFormatter {
    fn fill_in_middle(
        &self,
        request: FillInMiddleRequest,
    ) -> Either<LLMClientCompletionRequest, LLMClientCompletionStringRequest> {
        // format is
        // <｜fim▁begin｜>{{{prefix}}}<｜fim▁hole｜>{{{suffix}}}<｜fim▁end｜>
        // https://ollama.ai/library/deepseek
        let prefix = request.prefix();
        let suffix = request.suffix();
        let response = format!("<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>");
        let string_request =
            LLMClientCompletionStringRequest::new(request.llm().clone(), response, 0.0, None)
                .set_stop_words(request.stop_words())
                .set_max_tokens(512);
        Either::Right(string_request)
    }
}
