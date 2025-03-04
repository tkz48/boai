use crate::clients::types::LLMClientMessage;

use super::types::{LLMFormatting, TokenizerConfig, TokenizerError};

pub struct CodeLLama70BInstructFormatting {
    _tokenizer_config: TokenizerConfig,
}

impl CodeLLama70BInstructFormatting {
    pub fn new() -> Result<Self, TokenizerError> {
        let config = include_str!("tokenizer_config/codellama.json");
        let tokenizer_config = serde_json::from_str::<TokenizerConfig>(config)?;
        Ok(Self {
            _tokenizer_config: tokenizer_config,
        })
    }
}

impl LLMFormatting for CodeLLama70BInstructFormatting {
    fn to_prompt(&self, messages: Vec<LLMClientMessage>) -> String {
        // we want to convert the message to codellama format
        // persent here: https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf/blob/main/tokenizer_config.json#L4
        // {% if messages[0]['role'] == 'system' %}
        // {% set user_index = 1 %}
        // {% else %}
        // {% set user_index = 0 %}
        // {% endif %}
        // {% for message in messages %}
        // {% if (message['role'] == 'user') != ((loop.index0 + user_index) % 2 == 0) %}
        // {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        // {% endif %}
        // {% if loop.index0 == 0 %}
        // {{ '<s>' }}
        // {% endif %}
        // {% set content = 'Source: ' + message['role'] + '\n\n ' + message['content'].strip() %}
        // {{ content + ' <step> ' }}
        // {% endfor %}
        // {{'Source: assistant\nDestination: user\n\n '}}
        println!("{:?}", &messages);
        let formatted_message = messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| {
                let content = message.content().trim();
                let role = match message.role() {
                    role if role.is_assistant() => "assistant",
                    role if role.is_user() => "user",
                    _ => "system",
                };
                let prefix = if index == 0 { "<s>" } else { "" };
                format!("{}Source: {}\n\n {} <step> ", prefix, role, content.trim())
            })
            .collect::<Vec<_>>()
            .join("");
        let response = format!(
            "{}Source: assistant\nDestination: user\n\n ",
            formatted_message
        );
        println!("formatted message for llm");
        println!("{}", &response);
        response
    }
}

#[cfg(test)]
mod tests {

    use crate::clients::types::LLMClientMessage;

    use super::CodeLLama70BInstructFormatting;
    use super::LLMFormatting;

    #[test]
    fn test_formatting_works() {
        let messages = vec![
            LLMClientMessage::system("System prompt    ".to_owned()),
            LLMClientMessage::user("First user query".to_owned()),
            LLMClientMessage::assistant("Model response to first query".to_owned()),
            LLMClientMessage::user("Second user query".to_owned()),
        ];
        let codellama_formatting = CodeLLama70BInstructFormatting::new().unwrap();
        assert_eq!(
            codellama_formatting.to_prompt(messages),
            "<s>Source: system\n\n System prompt <step> Source: user\n\n First user query <step> Source: assistant\n\n Model response to first query <step> Source: user\n\n Second user query <step> Source: assistant\nDestination: user\n\n "
        );
    }
}
