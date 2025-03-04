//! We are going to run the various tokenizers here, we also make sure to run
//! the tokenizer in a different thread here, because its important that we
//! don't block the main thread from working

use std::collections::HashMap;
use std::str::FromStr;

use thiserror::Error;
use tiktoken_rs::ChatCompletionRequestMessage;
use tokenizers::Tokenizer;

use crate::{
    clients::types::{LLMClientMessage, LLMClientRole, LLMType},
    format::{
        claude::ClaudeFormatting,
        deepseekcoder::DeepSeekCoderFormatting,
        llama70b::CodeLLama70BInstructFormatting,
        mistral::MistralInstructFormatting,
        mixtral::MixtralInstructFormatting,
        types::{LLMFormatting, TokenizerError},
    },
};

pub struct LLMTokenizer {
    pub tokenizers: HashMap<LLMType, Tokenizer>,
    pub formatters: HashMap<LLMType, Box<dyn LLMFormatting + Send + Sync>>,
}

#[derive(Error, Debug)]
pub enum LLMTokenizerError {
    #[error("Tokenizer not found for model {0}")]
    TokenizerNotFound(LLMType),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("error from tokenizer crate: {0}")]
    TokenizerCrateError(#[from] tokenizers::Error),

    #[error("anyhow error: {0}")]
    AnyhowError(#[from] anyhow::Error),

    #[error("tokenizer error: {0}")]
    TokenizerErrorInternal(#[from] TokenizerError),

    #[error("fast count not supported for messages")]
    FastCountNotSupportedForMessages,
}

pub enum LLMTokenizerInput {
    Prompt(String),
    Messages(Vec<LLMClientMessage>),
}

impl LLMTokenizer {
    pub fn new() -> Result<Self, LLMTokenizerError> {
        let tokenizer = Self {
            tokenizers: HashMap::new(),
            formatters: HashMap::new(),
        };
        let updated_tokenizer = tokenizer
            .add_llm_type(
                LLMType::Mixtral,
                Box::new(MixtralInstructFormatting::new()?),
            )
            .add_llm_type(
                LLMType::MistralInstruct,
                Box::new(MistralInstructFormatting::new()?),
            )
            .add_llm_type(
                LLMType::DeepSeekCoder1_3BInstruct,
                Box::new(DeepSeekCoderFormatting::new()),
            )
            .add_llm_type(
                LLMType::CodeLLama70BInstruct,
                Box::new(CodeLLama70BInstructFormatting::new()?),
            )
            .add_llm_type(
                LLMType::CodeLlama13BInstruct,
                Box::new(CodeLLama70BInstructFormatting::new()?),
            )
            .add_llm_type(
                LLMType::CodeLlama7BInstruct,
                Box::new(CodeLLama70BInstructFormatting::new()?),
            )
            .add_llm_type(
                LLMType::DeepSeekCoder6BInstruct,
                Box::new(DeepSeekCoderFormatting::new()),
            )
            .add_llm_type(
                LLMType::DeepSeekCoder33BInstruct,
                Box::new(DeepSeekCoderFormatting::new()),
            )
            .add_llm_type(LLMType::ClaudeOpus, Box::new(ClaudeFormatting::new()))
            .add_llm_type(LLMType::ClaudeSonnet, Box::new(ClaudeFormatting::new()))
            .add_llm_type(LLMType::ClaudeHaiku, Box::new(ClaudeFormatting::new()))
            .add_llm_type(LLMType::GeminiPro, Box::new(ClaudeFormatting::new()));
        Ok(updated_tokenizer)
    }

    fn add_llm_type(
        mut self,
        llm_type: LLMType,
        formatter: Box<dyn LLMFormatting + Send + Sync>,
    ) -> Self {
        // This can be falliable, since soe llms might have formatting support
        // and if they don't thats fine
        let _ = self.load_tokenizer(&llm_type);
        self.formatters.insert(llm_type, formatter);
        self
    }

    fn to_openai_tokenizer(&self, model: &LLMType) -> Option<String> {
        match model {
            LLMType::GPT3_5_16k => Some("gpt-3.5-turbo-16k-0613".to_owned()),
            LLMType::Gpt4 => Some("gpt-4-0613".to_owned()),
            LLMType::Gpt4Turbo => Some("gpt-4-1106-preview".to_owned()),
            LLMType::Gpt4_32k => Some("gpt-4-32k-0613".to_owned()),
            // TODO(skcd): This is the wrong tokenizer we really want to use
            // the new o200k here, but tiktoken needs to upgrade first
            LLMType::Gpt4O => Some("gpt-4-32k-0613".to_owned()),
            LLMType::Gpt4OMini => Some("gpt-4-32k-0613".to_owned()),
            _ => None,
        }
    }

    pub fn count_tokens_approx(
        &self,
        _: &LLMType,
        input: LLMTokenizerInput,
    ) -> Result<usize, LLMTokenizerError> {
        match input {
            LLMTokenizerInput::Prompt(prompt) => {
                let words = prompt.split_whitespace().count();
                let new_line_count = prompt.lines().count();
                // the approx algorithm is (words + new_line_count) * 4/3
                // each token is approx 3/4th of a word
                Ok(((words + new_line_count) * 4) / 3)
            }
            LLMTokenizerInput::Messages(_) => {
                Err(LLMTokenizerError::FastCountNotSupportedForMessages)
            }
        }
    }

    pub fn count_tokens(
        &self,
        model: &LLMType,
        input: LLMTokenizerInput,
    ) -> Result<usize, LLMTokenizerError> {
        match input {
            LLMTokenizerInput::Prompt(prompt) => self.count_tokens_using_tokenizer(model, &prompt),
            LLMTokenizerInput::Messages(messages) => {
                // we can't send messages directly to the tokenizer, we have to
                // either make it a message or its an openai prompt in which case
                // its fine
                // so we are going to return an error if its not openai
                if model.is_openai() {
                    // we can use the openai tokenizer
                    let model = self.to_openai_tokenizer(model);
                    match model {
                        Some(model) => Ok(tiktoken_rs::num_tokens_from_messages(
                            &model,
                            messages
                                .into_iter()
                                .map(|message| {
                                    let role = message.role();
                                    let content = message.content();
                                    match role {
                                        LLMClientRole::User => ChatCompletionRequestMessage {
                                            role: "user".to_owned(),
                                            content: Some(content.to_owned()),
                                            name: None,
                                            function_call: None,
                                        },
                                        LLMClientRole::Assistant => ChatCompletionRequestMessage {
                                            role: "assistant".to_owned(),
                                            content: Some(content.to_owned()),
                                            name: None,
                                            function_call: None,
                                        },
                                        LLMClientRole::System => ChatCompletionRequestMessage {
                                            role: "system".to_owned(),
                                            content: Some(content.to_owned()),
                                            name: None,
                                            function_call: None,
                                        },
                                        LLMClientRole::Function => ChatCompletionRequestMessage {
                                            role: "function".to_owned(),
                                            content: Some(content.to_owned()),
                                            name: None,
                                            function_call: None,
                                        },
                                    }
                                })
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )?),
                        None => Err(LLMTokenizerError::TokenizerError(
                            "Only openai models are supported for messages".to_owned(),
                        )),
                    }
                } else {
                    let prompt = self
                        .formatters
                        .get(model)
                        .map(|formatter| formatter.to_prompt(messages.to_vec()))
                        .unwrap_or(ClaudeFormatting::new().to_prompt(messages));
                    let prompt_length = prompt.len();
                    let num_tokens = self
                        .tokenizers
                        .get(model)
                        .map(|tokenizer| {
                            tokenizer
                                .encode(prompt, false)
                                .map(|encoding| encoding.len())
                                .unwrap_or(prompt_length)
                        })
                        .unwrap_or(prompt_length);
                    Ok(num_tokens)
                }
            }
        }
    }

    pub fn count_tokens_using_tokenizer(
        &self,
        model: &LLMType,
        prompt: &str,
    ) -> Result<usize, LLMTokenizerError> {
        // we have the custom tokenizers already loaded, if this is not the openai loop
        if !model.is_openai() {
            let tokenizer = self.tokenizers.get(model);
            match tokenizer {
                Some(tokenizer) => {
                    // Now over here we will try to figure out how to pass the
                    // values around
                    let result = tokenizer.encode(prompt, false);
                    match result {
                        Ok(encoding) => Ok(encoding.len()),
                        Err(e) => Err(LLMTokenizerError::TokenizerError(format!(
                            "Failed to encode prompt: {}",
                            e
                        ))),
                    }
                }
                None => Ok(prompt.len()),
            }
        } else {
            // If we are using openai model, then we have to use the bpe config
            // and count the number of tokens
            let model = self.to_openai_tokenizer(model);
            if let None = model {
                return Err(LLMTokenizerError::TokenizerError(
                    "OpenAI model not found".to_owned(),
                ));
            }
            let model = model.expect("if let None to hold");
            let bpe = tiktoken_rs::get_bpe_from_model(&model)?;
            Ok(bpe.encode_ordinary(prompt).len())
        }
    }

    pub fn load_tokenizer(&mut self, model: &LLMType) -> Result<(), LLMTokenizerError> {
        let tokenizer = match model {
            LLMType::MistralInstruct => {
                let config = include_str!("configs/mistral.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::Mixtral => {
                let config = include_str!("configs/mixtral.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::DeepSeekCoder1_3BInstruct => {
                let config = include_str!("configs/deepseekcoder.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::CodeLLama70BInstruct => {
                let config = include_str!("configs/mistral.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::CodeLlama13BInstruct => {
                let config = include_str!("configs/mistral.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::CodeLlama7BInstruct => {
                let config = include_str!("configs/mistral.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::DeepSeekCoder6BInstruct => {
                let config = include_str!("configs/deepseekcoder.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::DeepSeekCoder33BInstruct => {
                let config = include_str!("configs/deepseekcoder.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::ClaudeOpus | LLMType::ClaudeSonnet | LLMType::ClaudeHaiku => {
                let config = include_str!("configs/claude.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::GeminiPro => {
                let config = include_str!("configs/deepseekcoder.json");
                Some(Tokenizer::from_str(config)?)
            }
            _ => None,
        };
        if let Some(tokenizer) = tokenizer {
            self.tokenizers.insert(model.clone(), tokenizer);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use tokenizers::Tokenizer;

    #[test]
    fn test_loading_deepseek_tokenizer_works() {
        let tokenizer_file = include_str!("configs/deepseekcoder.json");
        let _ = Tokenizer::from_str(tokenizer_file).unwrap();
    }
}
