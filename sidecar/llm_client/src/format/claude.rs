use crate::clients::types::{LLMClientMessage, LLMClientRole};

use super::types::LLMFormatting;

pub struct ClaudeFormatting {}

impl ClaudeFormatting {
    pub fn new() -> Self {
        Self {}
    }
}

impl LLMFormatting for ClaudeFormatting {
    fn to_prompt(&self, messages: Vec<LLMClientMessage>) -> String {
        // well claude does not tell us how to make this work, there is no config
        // and the counting of tokens is not exposed so we use a huresitc here to count it
        // we use System Message: {system_message}
        // Human: {human_message}
        // Assistant: {assistant_message}
        let formatted_message = messages.into_iter().map(|message| {
            let role = message.role();
            match role {
                LLMClientRole::System => format!("System Message: {}\n", message.content()),
                LLMClientRole::User => format!("Human: {}\n", message.content()),
                LLMClientRole::Assistant => format!("Assistant: {}\n", message.content()),
                LLMClientRole::Function => format!("Function: {}\n", message.content()),
            }
        });
        formatted_message.collect::<Vec<_>>().join("\n")
    }
}
