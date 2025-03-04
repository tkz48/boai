use std::collections::HashMap;

use llm_client::clients::types::LLMType;

use super::{
    anthropic::AnthropicLineEditPrompt,
    codellama::CodeLlamaLineEditPrompt,
    deepseekcoder::DeepSeekCoderLinEditPrompt,
    mistral::MistralLineEditPrompt,
    openai::OpenAILineEditPrompt,
    types::{
        InLineDocRequest, InLineEditPrompt, InLineEditPromptError, InLineEditRequest,
        InLineFixRequest, InLinePromptResponse,
    },
};

pub struct InLineEditPromptBroker {
    prompt_generators: HashMap<LLMType, Box<dyn InLineEditPrompt + Send + Sync>>,
}

impl InLineEditPromptBroker {
    pub fn new() -> Self {
        let broker = Self {
            prompt_generators: HashMap::new(),
        };
        broker
            .insert_prompt_generator(LLMType::GPT3_5_16k, Box::new(OpenAILineEditPrompt::new()))
            .insert_prompt_generator(LLMType::Gpt4, Box::new(OpenAILineEditPrompt::new()))
            .insert_prompt_generator(LLMType::Gpt4_32k, Box::new(OpenAILineEditPrompt::new()))
            .insert_prompt_generator(LLMType::Gpt4O, Box::new(OpenAILineEditPrompt::new()))
            .insert_prompt_generator(
                LLMType::MistralInstruct,
                Box::new(MistralLineEditPrompt::new()),
            )
            .insert_prompt_generator(LLMType::Mixtral, Box::new(MistralLineEditPrompt::new()))
            .insert_prompt_generator(
                LLMType::CodeLlama7BInstruct,
                Box::new(CodeLlamaLineEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::CodeLlama13BInstruct,
                Box::new(CodeLlamaLineEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::Llama3_8bInstruct,
                Box::new(CodeLlamaLineEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::CodeLLama70BInstruct,
                Box::new(CodeLlamaLineEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::DeepSeekCoder6BInstruct,
                Box::new(DeepSeekCoderLinEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::DeepSeekCoder1_3BInstruct,
                Box::new(DeepSeekCoderLinEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::DeepSeekCoder33BInstruct,
                Box::new(DeepSeekCoderLinEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::ClaudeOpus,
                Box::new(AnthropicLineEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::ClaudeSonnet,
                Box::new(AnthropicLineEditPrompt::new()),
            )
            .insert_prompt_generator(
                LLMType::ClaudeHaiku,
                Box::new(AnthropicLineEditPrompt::new()),
            )
    }

    pub fn insert_prompt_generator(
        mut self,
        llm_type: LLMType,
        prompt_generator: Box<dyn InLineEditPrompt + Send + Sync>,
    ) -> Self {
        self.prompt_generators.insert(llm_type, prompt_generator);
        self
    }

    fn get_prompt_generator(
        &self,
        llm_type: &LLMType,
    ) -> Result<&Box<dyn InLineEditPrompt + Send + Sync>, InLineEditPromptError> {
        self.prompt_generators
            .get(llm_type)
            .ok_or(InLineEditPromptError::ModelNotSupported)
    }

    pub fn get_prompt(
        &self,
        llm_type: &LLMType,
        request: InLineEditRequest,
    ) -> Result<InLinePromptResponse, InLineEditPromptError> {
        let prompt_generator = self.get_prompt_generator(llm_type)?;
        Ok(prompt_generator.inline_edit(request))
    }

    pub fn get_fix_prompt(
        &self,
        llm_type: &LLMType,
        request: InLineFixRequest,
    ) -> Result<InLinePromptResponse, InLineEditPromptError> {
        let prompt_generator = self.get_prompt_generator(llm_type)?;
        Ok(prompt_generator.inline_fix(request))
    }

    pub fn get_doc_prompt(
        &self,
        llm_type: &LLMType,
        request: InLineDocRequest,
    ) -> Result<InLinePromptResponse, InLineEditPromptError> {
        let prompt_generator = self.get_prompt_generator(llm_type)?;
        Ok(prompt_generator.inline_doc(request))
    }
}
