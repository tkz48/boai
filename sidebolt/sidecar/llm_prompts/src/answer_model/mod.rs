//! We define all the properties for the model configuration related to answering
//! a user question in the chat format here

use std::collections::HashMap;

use lazy_static::lazy_static;
use llm_client::clients::types::LLMType;

lazy_static! {
    static ref CODE_LLAMA_STOP_WORDS: Vec<String> = vec![
        "<PRE>".to_owned(),
        "<SUF>".to_owned(),
        "<MID>".to_owned(),
        "<EOT>".to_owned(),
    ];
    static ref DEEPSEEK_STOP_WORDS: Vec<String> = vec![
        "<｜end▁of▁sentence｜>".to_owned(),
        "<｜begin▁of▁sentence｜>".to_owned(),
        "<｜fim▁end｜>".to_owned(),
    ];
}

#[derive(Debug, Clone)]
pub struct AnswerModel {
    pub llm_type: LLMType,
    /// The number of tokens reserved for the answer
    pub answer_tokens: i64,

    /// The number of tokens reserved for the prompt
    pub prompt_tokens_limit: i64,

    /// The number of tokens reserved for history
    pub history_tokens_limit: i64,

    /// The total number of tokens reserved for the model
    pub total_tokens: i64,

    /// Inline completion tokens, how many are we willing to generate
    pub inline_completion_tokens: Option<i64>,
}

impl AnswerModel {
    pub fn get_stop_words_inline_completion(&self) -> Option<Vec<String>> {
        match self.llm_type {
            LLMType::CodeLLama70BInstruct => Some(CODE_LLAMA_STOP_WORDS.to_vec()),
            LLMType::CodeLlama13BInstruct => Some(CODE_LLAMA_STOP_WORDS.to_vec()),
            LLMType::CodeLlama7BInstruct => Some(CODE_LLAMA_STOP_WORDS.to_vec()),
            LLMType::DeepSeekCoder1_3BInstruct => Some(DEEPSEEK_STOP_WORDS.to_vec()),
            LLMType::DeepSeekCoder6BInstruct => Some(DEEPSEEK_STOP_WORDS.to_vec()),
            LLMType::DeepSeekCoder33BInstruct => Some(DEEPSEEK_STOP_WORDS.to_vec()),
            _ => None,
        }
    }
}

// GPT-3.5-16k Turbo has 16,385 tokens
pub const GPT_3_5_TURBO_16K: AnswerModel = AnswerModel {
    llm_type: LLMType::GPT3_5_16k,
    answer_tokens: 1024 * 2,
    prompt_tokens_limit: 2500 * 2,
    history_tokens_limit: 2048 * 2,
    total_tokens: 16385,
    inline_completion_tokens: None,
};

// GPT-4 has 8,192 tokens
pub const GPT_4: AnswerModel = AnswerModel {
    llm_type: LLMType::Gpt4,
    answer_tokens: 1024,
    // The prompt tokens limit for gpt4 are a bit higher so we can get more context
    // when required
    prompt_tokens_limit: 4500 + 610,
    history_tokens_limit: 2048,
    total_tokens: 8192,
    inline_completion_tokens: None,
};

// GPT4-32k has 32,769 tokens
pub const GPT_4_32K: AnswerModel = AnswerModel {
    llm_type: LLMType::Gpt4_32k,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 32769,
    inline_completion_tokens: None,
};

// GPT4-Turbo has 128k tokens as input, but let's keep it capped at 32k tokens
// as LLMs exhibit LIM issues which has been frequently documented
pub const GPT_4_TURBO_128K: AnswerModel = AnswerModel {
    llm_type: LLMType::Gpt4Turbo,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 32769,
    inline_completion_tokens: None,
};

pub const GPT4_O_128K: AnswerModel = AnswerModel {
    llm_type: LLMType::Gpt4O,
    answer_tokens: 1024 * 2,
    prompt_tokens_limit: 2500 * 8,
    history_tokens_limit: 2048 * 8,
    total_tokens: 128000,
    inline_completion_tokens: None,
};

pub const GPT4_O_MINI: AnswerModel = AnswerModel {
    llm_type: LLMType::Gpt4OMini,
    answer_tokens: 8069,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 128000,
    inline_completion_tokens: None,
};

// MistralInstruct has 8k tokens in total
pub const MISTRAL_INSTRUCT: AnswerModel = AnswerModel {
    llm_type: LLMType::MistralInstruct,
    answer_tokens: 1024,
    prompt_tokens_limit: 4500,
    history_tokens_limit: 2048,
    total_tokens: 8000,
    inline_completion_tokens: None,
};

// Mixtral has 32k tokens in total
pub const MIXTRAL: AnswerModel = AnswerModel {
    llm_type: LLMType::Mixtral,
    answer_tokens: 1024,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 1024 * 4,
    total_tokens: 32000,
    inline_completion_tokens: None,
};

// LLAMA 3.1 8B Instruct
pub const LLAMA_3_1_8B_INSTRUCT: AnswerModel = AnswerModel {
    llm_type: LLMType::Llama3_1_8bInstruct,
    answer_tokens: 128_000,
    prompt_tokens_limit: 32_000,
    history_tokens_limit: 32_000,
    total_tokens: 128_000,
    inline_completion_tokens: None,
};

// CodeLLaMA70B has 100k tokens in total
pub const CODE_LLAMA_70B: AnswerModel = AnswerModel {
    llm_type: LLMType::CodeLLama70BInstruct,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 32769,
    inline_completion_tokens: None,
};

pub const CODE_LLAMA_13B: AnswerModel = AnswerModel {
    llm_type: LLMType::CodeLlama13BInstruct,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 16_000,
    // we run this very hot, so keep the context length on the lower end here
    // by default, only give out around 2056 tokens
    // another option is providing hosted version of this via togetherAI or
    // vllm hosted by us
    inline_completion_tokens: Some(2056),
};

pub const CODE_LLAMA_7B: AnswerModel = AnswerModel {
    llm_type: LLMType::CodeLlama7BInstruct,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 16_000,
    // we run this very hot, so keep the context length on the lower end here
    // by default, only give out around 2056 tokens
    // another option is providing hosted version of this via togetherAI or
    // vllm hosted by us
    inline_completion_tokens: Some(2056),
};

pub const DEEPSEEK_CODER_1_3B_INSTRUCT: AnswerModel = AnswerModel {
    llm_type: LLMType::DeepSeekCoder1_3BInstruct,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 16_000,
    inline_completion_tokens: Some(2056),
};

pub const DEEPSEEK_CODER_6B: AnswerModel = AnswerModel {
    llm_type: LLMType::DeepSeekCoder6BInstruct,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 16_000,
    inline_completion_tokens: Some(2056),
};

pub const DEEPSEEK_CODER_33B: AnswerModel = AnswerModel {
    llm_type: LLMType::DeepSeekCoder33BInstruct,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 16_000,
    inline_completion_tokens: Some(2056),
};

pub const CLAUDE_OPUS: AnswerModel = AnswerModel {
    llm_type: LLMType::ClaudeOpus,
    // https://arc.net/l/quote/wjrntwlo
    answer_tokens: 4096,
    prompt_tokens_limit: 150 * 1000,
    history_tokens_limit: 50 * 1000,
    total_tokens: 200 * 1000,
    inline_completion_tokens: Some(2056),
};

pub const CLAUDE_SONNET: AnswerModel = AnswerModel {
    llm_type: LLMType::ClaudeSonnet,
    // https://arc.net/l/quote/wjrntwlo
    answer_tokens: 4096,
    prompt_tokens_limit: 150 * 1000,
    history_tokens_limit: 50 * 1000,
    total_tokens: 200 * 1000,
    inline_completion_tokens: Some(2056),
};

pub const CLAUDE_HAIKU: AnswerModel = AnswerModel {
    llm_type: LLMType::ClaudeHaiku,
    // https://arc.net/l/quote/wjrntwlo
    answer_tokens: 4096,
    prompt_tokens_limit: 150 * 1000,
    history_tokens_limit: 50 * 1000,
    total_tokens: 200 * 1000,
    inline_completion_tokens: Some(2056),
};

pub const GEMINI_PRO: AnswerModel = AnswerModel {
    llm_type: LLMType::GeminiPro,
    answer_tokens: 8192,
    prompt_tokens_limit: 950 * 1000,
    history_tokens_limit: 50 * 1000,
    total_tokens: 1 * 1000 * 1000,
    inline_completion_tokens: None,
};

pub struct LLMAnswerModelBroker {
    pub models: HashMap<LLMType, AnswerModel>,
}

impl LLMAnswerModelBroker {
    pub fn new() -> Self {
        let broker = Self {
            models: Default::default(),
        };
        broker
            .add_answer_model(GPT_3_5_TURBO_16K)
            .add_answer_model(GPT_4)
            .add_answer_model(GPT_4_32K)
            .add_answer_model(GPT_4_TURBO_128K)
            .add_answer_model(GPT4_O_128K)
            .add_answer_model(MISTRAL_INSTRUCT)
            .add_answer_model(MIXTRAL)
            .add_answer_model(CODE_LLAMA_13B)
            .add_answer_model(CODE_LLAMA_70B)
            .add_answer_model(CODE_LLAMA_7B)
            .add_answer_model(DEEPSEEK_CODER_1_3B_INSTRUCT)
            .add_answer_model(DEEPSEEK_CODER_6B)
            .add_answer_model(DEEPSEEK_CODER_33B)
            .add_answer_model(CLAUDE_OPUS)
            .add_answer_model(CLAUDE_SONNET)
            .add_answer_model(CLAUDE_HAIKU)
            .add_answer_model(GEMINI_PRO)
            .add_answer_model(GPT4_O_MINI)
            .add_answer_model(LLAMA_3_1_8B_INSTRUCT)
    }

    fn add_answer_model(mut self, model: AnswerModel) -> Self {
        self.models.insert(model.llm_type.clone(), model);
        self
    }

    pub fn inline_completion_tokens(&self, llm_type: &LLMType) -> Option<i64> {
        self.models
            .get(llm_type)
            .and_then(|model| model.inline_completion_tokens)
    }

    pub fn get_answer_model(&self, llm_type: &LLMType) -> Option<AnswerModel> {
        let default_answer_model = AnswerModel {
            llm_type: llm_type.clone(),
            answer_tokens: 8192,
            prompt_tokens_limit: 2500 * 8,
            history_tokens_limit: 2048 * 8,
            total_tokens: 128000,
            inline_completion_tokens: None,
        };
        // return the default answer model over here
        Some(
            self.models
                .get(llm_type)
                .map(|answer_model| answer_model.clone())
                .unwrap_or(default_answer_model),
        )
    }
}
