use llm_client::clients::togetherai::TogetherAIClient;
use llm_client::clients::types::LLMClient;
use llm_client::clients::types::LLMClientCompletionStringRequest;
use llm_client::provider::LLMProviderAPIKeys;
use llm_client::provider::TogetherAIProvider;

#[tokio::main]
async fn main() {
    let api_key = LLMProviderAPIKeys::TogetherAI(TogetherAIProvider::new(
        "cc10d6774e67efef2004b85efdb81a3c9ba0b7682cc33d59c30834183502208d".to_owned(),
    ));
    // let api_key = LLMProviderAPIKeys::Ollama(OllamaProvider {});
    let client = TogetherAIClient::new();
    // when we add comments to the prompt it still works
    let prompt =
        "<｜fim▁begin｜>// Clipboard: function add(a: number, b: number) {\n\t#We are going to add 2 numbers\n\treturn a + b;\n}\n// Path: testing.ts\nfunction subtract(a<｜fim▁hole｜>)<｜fim▁end｜>";
    // let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    // let response = client
    //     .stream_prompt_completion(api_key, request, sender)
    //     .await;
    // println!("{}", response.expect("to work"));
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let request = LLMClientCompletionStringRequest::new(
        llm_client::clients::types::LLMType::DeepSeekCoder33BInstruct,
        prompt.to_owned(),
        0.2,
        None,
    )
    .set_max_tokens(100);
    let response = client
        .stream_prompt_completion(api_key, request, sender)
        .await;
    println!("{}", response.expect("to work"));
}
