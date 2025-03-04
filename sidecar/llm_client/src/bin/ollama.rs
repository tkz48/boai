use llm_client::clients::types::LLMClientCompletionStringRequest;
use llm_client::{
    clients::{ollama::OllamaClient, types::LLMClient},
    provider::OllamaProvider,
};

#[tokio::main]
async fn main() {
    let ollam_client = OllamaClient::new();
    let api_key = llm_client::provider::LLMProviderAPIKeys::Ollama(OllamaProvider {});
    let prompt = r#"<|im_start|>system
You have to take the code which is provided to you in ## Code Context and apply the changes made by a junior engineer to it, which is provided in the ## Export to codebase.
The junior engineer is lazy and sometimes forgets to write the whole code and leaves `// rest of code ..` or `// ..` in the code, so you have to make sure that you complete the code completely from the original code context when generating the final code.
Make sure the code which you generate is complete with the changes applied from the ## Export to codebase section and do not be lazy and leave comments like `// rest of code ..` or `// ..`
The code needs to be generated in typescript.<|im_end|>
<|im_start|>user
## Code Context:
```typescript
// FILEPATH: src/providers/providerwrapper.ts
// BEGIN: abpxx6d04wxr
async function streamTokens(
    request: CohereGenerationTypes.Request,
    options: CohereGenerationTypes.RequestOptions,
  ): Promise<ReadableStream<string>> {
    const byteStream = await streamBytes(request, options);
    return byteStream.pipeThrough(new CohereGenerationDecoderStream(chunkToToken));
  }
  
  export class CohereGeneration {
    static run = run;
    static stream = stream;
    static streamBytes = streamBytes;
    static streamTokens = streamTokens;
  }
// END: abpxx6d04wxr
```

## Export to codebase
Sure, here are the docstrings for the selected code:

```typescript
/**
 * Asynchronously streams tokens from a Cohere generation request.
 *
 * @param {CohereGenerationTypes.Request} request - The generation request to be processed.
 * @param {CohereGenerationTypes.RequestOptions} options - Options for the request.
 * @returns {Promise<ReadableStream<string>>} - A promise that resolves to a readable stream of tokens.
 */
async function streamTokens(
  request: CohereGenerationTypes.Request,
  options: CohereGenerationTypes.RequestOptions,
): Promise<ReadableStream<string>> {
  const byteStream = await streamBytes(request, options);
  return byteStream.pipeThrough(new CohereGenerationDecoderStream(chunkToToken));
}
```

Now you have to generate the code after applying the edits mentioned in the ## Export to codebase section making sure that we complete the whole code from the ## Code Context and make sure not to leave any `// rest of code..` or `// ..` comments.
Just generate the final code starting with a single code block enclosed in ```typescript and ending with ```
Remember to APPLY THE EDITS from the ## Export to codebase section and make sure to complete the code from the ## Code Context section.
## Final Output:<|im_end|>
<|im_start|>assistant"#;
    let request = LLMClientCompletionStringRequest::new(
        llm_client::clients::types::LLMType::Custom(
            "codestory-finetune-export-to-codebase:latest".to_owned(),
        ),
        prompt.to_owned(),
        0.7,
        None,
    );
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let response = ollam_client
        .stream_prompt_completion(api_key, request, sender)
        .await;
    dbg!(&response);
}
