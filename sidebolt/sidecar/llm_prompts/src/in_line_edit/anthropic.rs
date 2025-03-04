use llm_client::clients::types::{LLMClientMessage, LLMClientRole};

use super::{
    openai::OpenAILineEditPrompt,
    types::{
        InLineDocRequest, InLineEditPrompt, InLineEditRequest, InLineFixRequest,
        InLinePromptResponse,
    },
};

pub struct AnthropicLineEditPrompt {
    openai_line_edit: OpenAILineEditPrompt,
}

impl AnthropicLineEditPrompt {
    pub fn new() -> Self {
        Self {
            openai_line_edit: OpenAILineEditPrompt::new(),
        }
    }

    fn system_message_inline_edit(&self, language: &str) -> String {
        format!(
            r#"You are an AI programming assistant.
When asked for your name, you must respond with "Aide".
Follow the user's requirements carefully & to the letter.
- Output the edited code in a single code block.
- Minimize any other prose.
- Each code block starts with ```{language} and // FILEPATH.
- If you suggest to run a terminal command, use a code block that starts with ```bash.
- You always answer with {language} code.
- Modify the code or create new code.
- Unless directed otherwise, the user is expecting for you to edit their selected code.
- Make sure to ALWAYS INCLUDE the BEGIN and END markers in your generated code with // BEGIN and then // END which is present in the code selection given by the user and start your answer with ```{language} codeblock
- The user will also provide extra context for you to use in <extra_data>{{extra_data}}</extra_data>, use them as instructed by the user"#
        )
    }

    fn _system_message_fix(&self, language: &str) -> String {
        format!(
            r#"You are an AI programming assistant.
When asked for your name, you must respond with "Aide".
Follow the user's requirements carefully & to the letter.
- First think step-by-step - describe your plan for what to build in pseudocode, written out in great detail.
- Then output the code in a single code block.
- Each code block starts with ``` and // FILEPATH.
- If you suggest to run a terminal command, use a code block that starts with ```bash.
- You always answer with {language} code.
- Modify the code or create new code.
- Unless directed otherwise, the user is expecting for you to edit their selected code."#
        )
    }

    fn _documentation_system_prompt(&self, language: &str, is_identifier_node: bool) -> String {
        if is_identifier_node {
            let system_prompt = format!(
                r#"You are an AI programming assistant.
When asked for your name, you must respond with "Aide".
Follow the user's requirements carefully & to the letter.
- Each code block must ALWAYS STARTS and include ```{language} and // FILEPATH
- You always answer with {language} code.
- When the user asks you to document something, you must answer in the form of a {language} code block.
- Your documentation should not include just the name of the function, think about what the function is really doing.
- When generating the documentation, be sure to understand what the function is doing and include that as part of the documentation and then generate the documentation.
- DO NOT modify the code which you will be generating"#
            );
            system_prompt.to_owned()
        } else {
            let system_prompt = format!(
                r#"You are an AI programming assistant.
When asked for your name, you must respond with "Aide".
Follow the user's requirements carefully & to the letter.
- Each code block must ALWAYS STARTS and include ```{language} and // FILEPATH
- You always answer with {language} code.
- When the user asks you to document something, you must answer in the form of a {language} code block.
- Your documentation should not include just the code selection, think about what the selection is really doing.
- When generating the documentation, be sure to understand what the selection is doing and include that as part of the documentation and then generate the documentation.
- DO NOT modify the code which you will be generating"#
            );
            system_prompt.to_owned()
        }
    }

    fn above_selection(&self, above_context: Option<&String>) -> Option<String> {
        if let Some(above_context) = above_context {
            Some(format!(
                r#"I have the following code above:
<code_above>
{above_context}
</code_above>"#
            ))
        } else {
            None
        }
    }

    fn below_selection(&self, below_context: Option<&String>) -> Option<String> {
        if let Some(below_context) = below_context {
            Some(format!(
                r#"I have the following code below:
<code_below>
{below_context}
</code_below>"#
            ))
        } else {
            None
        }
    }

    fn extra_data(&self, extra_data: &[String]) -> Option<String> {
        if extra_data.is_empty() {
            None
        } else {
            let data = extra_data
                .into_iter()
                .map(|data| {
                    format!(
                        r#"<data>
{data}
</data>"#
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(format!(
                r#"<extra_data>
{data}
</extra_data>"#
            ))
        }
    }

    #[allow(unused_assignments)]
    fn fix_inline_prompt_response(&self, response: InLinePromptResponse) -> InLinePromptResponse {
        match response {
            InLinePromptResponse::Completion(completion) => {
                InLinePromptResponse::Completion(completion)
            }
            InLinePromptResponse::Chat(chat_messages) => {
                let mut final_chat_messages = vec![];
                // the limitation we have here is that we have to concatenate all the consecutive
                // user and assistant messages together
                let mut previous_role: Option<LLMClientRole> = None;
                let mut pending_message: Option<LLMClientMessage> = None;
                for chat_message in chat_messages.into_iter() {
                    let role = chat_message.role().clone();
                    // if roles match, then we just append this to the our ongoing message
                    if previous_role == Some(role) {
                        if pending_message.is_some() {
                            pending_message = pending_message
                                .map(|pending_message| pending_message.concat(chat_message));
                        }
                    } else {
                        // if we have some previous message we should flush it
                        if let Some(pending_message_value) = pending_message {
                            final_chat_messages.push(pending_message_value);
                            pending_message = None;
                            previous_role = None;
                        }
                        // set the previous message and the role over here
                        previous_role = Some(chat_message.role().clone());
                        pending_message = Some(chat_message);
                    }
                }
                // if we still have some value remaining we push it to our chat messages
                if let Some(pending_message_value) = pending_message {
                    final_chat_messages.push(pending_message_value);
                }
                InLinePromptResponse::Chat(final_chat_messages)
            }
        }
    }
}

impl InLineEditPrompt for AnthropicLineEditPrompt {
    fn inline_edit(&self, request: InLineEditRequest) -> InLinePromptResponse {
        // Here we create the messages for the openai, since we have flexibility
        // and the llms are in general smart we can just send the chat messages
        // instead of the completion(which has been deprecated)
        let above = request.above();
        let below = request.below();
        let in_range = request.in_range();
        let language = request.language();
        let extra_data = request.extra_data();

        let mut messages = vec![];
        messages.push(LLMClientMessage::system(
            self.system_message_inline_edit(language),
        ));
        if let Some(extra_data) = self.extra_data(extra_data) {
            messages.push(LLMClientMessage::user(extra_data));
        }
        if let Some(above) = self.above_selection(above) {
            messages.push(LLMClientMessage::user(above));
        }
        if let Some(below) = self.below_selection(below) {
            messages.push(LLMClientMessage::user(below));
        }
        if let Some(in_range) = in_range {
            messages.push(LLMClientMessage::user(format!(
                r#"<code_to_modify>
{in_range}
</code_to_modify>"#
            )));
        }
        let user_query = request.user_query().to_owned();
        messages.push(LLMClientMessage::user(format!(
            r#"Only edit code in the <code_to_modify> section, my instructions are
{user_query}"#
        )));
        let inline_prompt_response = InLinePromptResponse::Chat(messages);
        self.fix_inline_prompt_response(inline_prompt_response)
    }

    fn inline_fix(&self, request: InLineFixRequest) -> InLinePromptResponse {
        let inline_prompt_response = self.openai_line_edit.inline_fix(request);
        self.fix_inline_prompt_response(inline_prompt_response)
    }

    fn inline_doc(&self, request: InLineDocRequest) -> InLinePromptResponse {
        let inline_prompt_response = self.openai_line_edit.inline_doc(request);
        self.fix_inline_prompt_response(inline_prompt_response)
    }
}

#[cfg(test)]
mod tests {
    use llm_client::clients::types::LLMClientMessage;

    use crate::in_line_edit::types::InLinePromptResponse;

    use super::AnthropicLineEditPrompt;

    #[test]
    fn test_merging_of_messages_works() {
        let messages = vec![
            LLMClientMessage::system("s1".to_owned()),
            LLMClientMessage::user("u1".to_owned()),
            LLMClientMessage::user("u2".to_owned()),
            LLMClientMessage::user("u3".to_owned()),
        ];
        let prompt = AnthropicLineEditPrompt::new();
        let fixed_messages =
            prompt.fix_inline_prompt_response(InLinePromptResponse::Chat(messages));
        assert!(matches!(fixed_messages, InLinePromptResponse::Chat(_)));
        let messages = fixed_messages.messages();
        assert!(messages.is_some());
        let messages = messages.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content(), "s1");
        assert_eq!(messages[1].content(), "u1\nu2\nu3");
    }
}
