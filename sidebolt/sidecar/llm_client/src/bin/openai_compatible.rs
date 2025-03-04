use llm_client::clients::{
    openai_compatible::OpenAICompatibleClient,
    types::{LLMClient, LLMClientCompletionRequest, LLMClientMessage},
};
use llm_client::provider::OpenAICompatibleConfig;

#[tokio::main]
async fn main() {
    let api_key = "".to_owned();
    // put your api base over here
    let api_base = "".to_owned();
    // put the model name over here, not sure if this is really required for LMStudio?
    let model_name = "".to_owned();

    let openai_client = OpenAICompatibleClient::new();
    let api_key =
        llm_client::provider::LLMProviderAPIKeys::OpenAICompatible(OpenAICompatibleConfig {
            api_key,
            api_base,
        });
    let request = LLMClientCompletionRequest::new(
        llm_client::clients::types::LLMType::Custom(model_name.to_owned()),
        vec![LLMClientMessage::system(
            "tell me how to add 2 numbers in rust".to_owned(),
        )],
        1.0,
        None,
    );
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let response = openai_client
        .stream_completion(api_key, request, sender)
        .await;

    // wait for the magic to show up in your stdout
    dbg!(&response);
}
