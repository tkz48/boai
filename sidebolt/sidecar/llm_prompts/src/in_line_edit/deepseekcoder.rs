use crate::in_line_edit::doc_helpers::selection_type;

use super::{
    doc_helpers::documentation_type,
    types::{
        InLineDocRequest, InLineEditPrompt, InLineEditRequest, InLineFixRequest,
        InLinePromptResponse,
    },
};

pub struct DeepSeekCoderLinEditPrompt {
    system_message: String,
}

impl DeepSeekCoderLinEditPrompt {
    pub fn new() -> Self {
        Self {
            system_message: "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n".to_owned(),
        }
    }

    fn extra_code_context(&self, extra_data: &[String]) -> String {
        if extra_data.is_empty() {
            String::new()
        } else {
            let extra_data_str = extra_data.join("\n");
            let extra_data_prompt = format!(
                "The following code context has been provided to you:
{extra_data_str}\n"
            );
            extra_data_prompt
        }
    }

    /// We try to get the inline edit prompt for the code here, so we can ask
    /// the LLM to generate the prompt
    fn code_context(&self, above: Option<&String>, below: Option<&String>) -> String {
        // Do we have some code context above?
        let above = if let Some(above) = above {
            format!(
                r#"Code Context above the selection:
{above}
"#
            )
        } else {
            String::new()
        };

        // Do we have some code context below?
        let below = if let Some(below) = below {
            format!(
                r#"Code Context below the selection:
{below}
"#
            )
        } else {
            String::new()
        };

        // We send this context to the LLM
        let code_context = format!(r#"{above}{below}"#,);
        code_context
    }
}

impl InLineEditPrompt for DeepSeekCoderLinEditPrompt {
    fn inline_doc(&self, request: InLineDocRequest) -> InLinePromptResponse {
        let comment_type = documentation_type(&request);
        let selection_type = selection_type(&request);
        let in_range = request.in_range();
        let language = request.language();
        let file_path = request.file_path();
        // deepseek follows the following format for the prompt:
        // {system_prompt}{instruction}
        // and then get the response with <|EOT|> at the end of it
        let instruction = format!("You are an expert software engineer. You have to generate {comment_type} for {selection_type}, the {selection_type} is given below:
{in_range}

Add {comment_type} and generate the selected code, do not forget the // END marker");
        let response = format!(
            r#"### Response:
```{language}
// FILEPATH: {file_path}
// BEGIN: ed8c6549bwf9
"#
        );
        let system_message = self.system_message.to_owned();
        let instruction = format!("### Instruction:\n{}\n", instruction);
        let prompt = format!("{system_message}{instruction}{response}");
        InLinePromptResponse::Completion(prompt)
    }

    fn inline_edit(&self, request: InLineEditRequest) -> InLinePromptResponse {
        let extra_data_context = self.extra_code_context(request.extra_data());
        let code_context = self.code_context(request.above(), request.below());
        let user_query = request.user_query();
        let language = request.language();
        let file_path = request.file_path();
        let (selection_context, extra_instruction) = if let Some(in_range_code_context) =
            request.in_range()
        {
            (
                format!(
                    r#"Your task is to rewrite the code below following the instruction: {user_query}
Code you have to edit:
{in_range_code_context}"#
                ),
                "Rewrite the code",
            )
        } else {
            (
                format!(r#"Follow the user instruction and generate code: {user_query}"#),
                "Generate the code",
            )
        };
        let instruction = format!(
            r#"You are an expert software engineer. You have been given some code context below:
{extra_data_context}
{code_context}
{selection_context}

{extra_instruction}"#
        );
        let response = format!(
            r#"```{language}
            // FILEPATH: {file_path}
            // BEGIN: ed8c6549bwf9
"#
        );
        let system_message = self.system_message.to_owned();
        let instruction = format!("### Instruction:\n{}\n", instruction);
        let response = format!("### Response:\n{}", response);
        let prompt = format!("{system_message}{instruction}{response}");
        InLinePromptResponse::Completion(prompt)
    }

    fn inline_fix(&self, request: InLineFixRequest) -> InLinePromptResponse {
        let code_context = self.code_context(request.above(), request.below());
        let language = request.language();
        let errors = request.diagnostics_prompts().join("\n");
        let in_range_code_context = request.in_range();
        let file_path = request.file_path();
        let selection_context = format!(
            r#"Your task is to fix the errors in the code using the errors provided
{errors}

Code you have to edit:
{in_range_code_context}"#
        );
        let instruction = format!(
            r#"You are an expert software engineer. You have to fix the errors present in the code, the context is given below:
{code_context}

{selection_context}

You have to fix the code below, generate the code without any explanation"#
        );
        let response = format!(
            r#"```{language}
            // FILEPATH: {file_path}
            // BEGIN: ed8c6549bwf9
"#
        );
        let system_message = self.system_message.to_owned();
        let instruction = format!("### Instruction:\n{}\n", instruction);
        let response = format!("### Response:\n{}", response);
        let prompt = format!("{system_message}{instruction}{response}");
        InLinePromptResponse::Completion(prompt)
    }
}
