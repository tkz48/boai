//! Call the endpoints of codestory endpoint

use llm_client::{
    clients::{
        codestory::CodeStoryClient,
        types::{LLMClient, LLMClientCompletionRequest, LLMClientMessage, LLMType},
    },
    provider::{CodestoryAccessToken, LLMProviderAPIKeys},
};

#[tokio::main]
async fn main() {
    let codestory_client = CodeStoryClient::new("https://0d19-80-209-142-211.ngrok-free.app");
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let request = LLMClientCompletionRequest::new(
        LLMType::ClaudeHaiku,
        vec![
            LLMClientMessage::system("you are a python expert".to_owned()),
            LLMClientMessage::user("Can you write 1 to 300 in a new line for me".to_owned()),
        ],
        1.0,
        None,
    )
    .set_max_tokens(2000);

    // fill this
    let codestory_access_token = "".to_owned();

    let response = codestory_client
        .stream_completion(
            LLMProviderAPIKeys::CodeStory(CodestoryAccessToken {
                access_token: codestory_access_token,
            }),
            request,
            sender,
        )
        .await;
    println!("{:?}", response);
}

#[cfg(test)]
mod tests {
    use llm_client::{
        clients::{
            codestory::CodeStoryClient,
            types::{LLMClient, LLMClientCompletionRequest, LLMClientMessage, LLMType},
        },
        provider::{CodestoryAccessToken, LLMProviderAPIKeys},
    };

    #[test]
    fn test_llm_types() {
        let llm_types = vec![
            LLMType::ClaudeHaiku,
            LLMType::CodeLlama13BInstruct,
            LLMType::Gpt4Turbo,
            LLMType::Gpt4,
            LLMType::GPT3_5_16k,
            LLMType::GeminiProFlash,
        ];

        let api_base = "https://0d19-80-209-142-211.ngrok-free.app";

        for llm_type in llm_types {
            // Test logic for each LLMType
            tracing::info!("Testing LLM type: {:?}", llm_type);
            test_llm_type(llm_type, &api_base);
        }
    }

    fn test_llm_type(llm_type: LLMType, api_base: &str) {
        // Your test logic here
        // For example:
        let codestory_client = CodeStoryClient::new(api_base);
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let request = LLMClientCompletionRequest::new(
            llm_type,
            vec![
                LLMClientMessage::system("you are a python expert".to_owned()),
                LLMClientMessage::user("Can you write 1 to 300 in a new line for me".to_owned()),
            ],
            1.0,
            None,
        )
        .set_max_tokens(2000);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let codestory_access_token = "".to_owned();

        let response = runtime.block_on(async {
            codestory_client
                .stream_completion(
                    LLMProviderAPIKeys::CodeStory(CodestoryAccessToken {
                        access_token: codestory_access_token,
                    }),
                    request,
                    sender,
                )
                .await
        });

        assert!(response.is_ok());
    }
}
