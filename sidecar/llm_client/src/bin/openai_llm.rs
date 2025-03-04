use futures::StreamExt;
use llm_client::{
    clients::{
        openai::OpenAIClient,
        types::{LLMClient, LLMClientCompletionRequest, LLMClientMessage},
    },
    provider::OpenAIProvider,
};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[tokio::main]
async fn main() {
    let openai_client = OpenAIClient::new();
    let api_key =
        llm_client::provider::LLMProviderAPIKeys::OpenAI(OpenAIProvider::new("".to_owned()));
    // let api_key =
    //     llm_client::provider::LLMProviderAPIKeys::OpenAIAzureConfig(ProviderAzureConfig {
    //         deployment_id: "gpt35-turbo-access".to_string(),
    //         api_base: "https://codestory-gpt4.openai.azure.com".to_owned(),
    //         api_key: "89ca8a49a33344c9b794b3dabcbbc5d0".to_owned(),
    //         api_version: "2023-08-01-preview".to_owned(),
    //     });
    let request = LLMClientCompletionRequest::new(
        llm_client::clients::types::LLMType::O1Preview,
        vec![LLMClientMessage::user(
            "tell me how to add 2 numbers in rust".to_owned(),
            ),
            LLMClientMessage::assistant(
            "Sure I will help you what, what kind of numbers are you interested in? is it usize or is it i64".to_owned()
            ),
            LLMClientMessage::user("Help me out with usize".to_owned())
        ],
        1.0,
        None,
    );
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let response = openai_client
        .stream_completion(api_key, request, sender)
        .await;
    let mut receiver = UnboundedReceiverStream::new(receiver);
    while let Some(delta) = receiver.next().await {
        println!("{:?}", delta);
    }
    dbg!(&response);
}
