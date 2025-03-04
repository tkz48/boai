use llm_client::{
    clients::{
        fireworks::FireworksAIClient,
        types::{LLMClient, LLMClientCompletionStringRequest, LLMType},
    },
    provider::{FireworksAPIKey, LLMProviderAPIKeys},
};

#[tokio::main]
async fn main() {
    let api_key = LLMProviderAPIKeys::FireworksAI(FireworksAPIKey {
        api_key: "s8Y7yIXdL0lMeHHgvbZXS77oGtBAHAsfsLviL2AKnzuGpg1n".to_owned(),
    });
    // let api_key = LLMProviderAPIKeys::OpenAICompatible(OpenAIComptaibleConfig {
    //     api_key: "some_key".to_owned(),
    //     api_base: "https://y2ukqtf6jeai9x-42424.proxy.runpod.net/v1".to_owned(),
    // });
    // let client = OpenAICompatibleClient::new();
    // let client = TogetherAIClient::new();
    let client = FireworksAIClient::new();
    let prompt = r#"<PRE> // Path: /Users/skcd/scratch/sidecar/sidecar/src/inline_completion/context/codebase_context.rs
//
//
// /// Creates the codebase context which we want to use
// /// for generating inline-completions
// pub struct CodeBaseContext {
//     tokenizer: Arc<LLMTokenizer>,
//     llm_type: LLMType,
//     file_path: String,
//     file_content: String,
//     cursor_position: Position,
//     symbol_tracker: Arc<SymbolTrackerInline>,
//     editor_parsing: Arc<EditorParsing>,
//     llm_response:
// }
//
// pub enum CodebaseContextString {
//     TruncatedToLimit(String, i64),
//     UnableToTruncate,
// }
//
// impl CodebaseContextString {
//     pub fn get_prefix_with_tokens(self) -> Option<(String, i64)> {
//         match self {
//             CodebaseContextString::TruncatedToLimit(prefix, used_tokens) => {
//                 Some((prefix, used_tokens))
//             }
//             CodebaseContextString::UnableToTruncate => None,
//         }
//     }
// }
//
// impl CodeBaseContext {
//     pub fn new(
//         tokenizer: Arc<LLMTokenizer>,
//         llm_type: LLMType,
//         file_path: String,
//         file_content: String,
//         cursor_position: Position,
//         symbol_tracker: Arc<SymbolTrackerInline>,
//         editor_parsing: Arc<EditorParsing>,
//     ) -> Self {
//         Self {
//             tokenizer,
//             llm_type,
//             file_path,
//             file_content,
//             cursor_position,
//             symbol_tracker,
//             editor_parsing,
//         }
//
// Path: /Users/skcd/scratch/sidecar/sidecar/src/inline_completion/context/codebase_context.rs
//
// /// Creates the codebase context which we want to use
// /// for generating inline-completions
// pub struct CodeBaseContext {
//     tokenizer: Arc<LLMTokenizer>,
//     llm_type: LLMType,
//     file_path: String,
//     file_content: String,
//     cursor_position: Position,
//     symbol_tracker: Arc<SymbolTrackerInline>,
//     editor_parsing: Arc<EditorParsing>,
//     llm_response:
// }
//
// pub enum CodebaseContextString {
//     TruncatedToLimit(String, i64),
//     UnableToTruncate,
// }
//
// impl CodebaseContextString {
//     pub fn get_prefix_with_tokens(self) -> Option<(String, i64)> {
//         match self {
//             CodebaseContextString::TruncatedToLimit(prefix, used_tokens) => {
//                 Some((prefix, used_tokens))
//             }
//             CodebaseContextString::UnableToTruncate => None,
//         }
//     }
// }
//
// impl CodeBaseContext {
//     pub fn new(
//         tokenizer: Arc<LLMTokenizer>,
//         llm_type: LLMType,
//         file_path: String,
//         file_content: String,
//         cursor_position: Position,
//         symbol_tracker: Arc<SymbolTrackerInline>,
//         editor_parsing: Arc<EditorParsing>,
//     ) -> Self {
//         Self {
//             tokenizer,
//             llm_type,
//             file_path,
//             file_content,
//             cursor_position,
//             symbol_tracker,
//             editor_parsing,
//         }
//     }
//
// Path: /Users/skcd/scratch/sidecar/sidecar/src/inline_completion/context/codebase_context.rs
// /// Creates the codebase context which we want to use
// /// for generating inline-completions
// pub struct CodeBaseContext {
//     tokenizer: Arc<LLMTokenizer>,
//     llm_type: LLMType,
//     file_path: String,
//     file_content: String,
//     cursor_position: Position,
//     symbol_tracker: Arc<SymbolTrackerInline>,
//     editor_parsing: Arc<EditorParsing>,
//     llm_response:
// }
//
// pub enum CodebaseContextString {
//     TruncatedToLimit(String, i64),
//     UnableToTruncate,
// }
//
// impl CodebaseContextString {
//     pub fn get_prefix_with_tokens(self) -> Option<(String, i64)> {
//         match self {
//             CodebaseContextString::TruncatedToLimit(prefix, used_tokens) => {
//                 Some((prefix, used_tokens))
//             }
//             CodebaseContextString::UnableToTruncate => None,
//         }
//     }
// }
//
// impl CodeBaseContext {
//     pub fn new(
//         tokenizer: Arc<LLMTokenizer>,
//         llm_type: LLMType,
//         file_path: String,
//         file_content: String,
//         cursor_position: Position,
//         symbol_tracker: Arc<SymbolTrackerInline>,
//         editor_parsing: Arc<EditorParsing>,
//     ) -> Self {
//         Self {
//             tokenizer,
//             llm_type,
//             file_path,
//             file_content,
//             cursor_position,
//             symbol_tracker,
//             editor_parsing,
//         }
//     }
//
//
// Path: /Users/skcd/scratch/sidecar/sidecar/src/inline_completion/context/codebase_context.rs
// /// for generating inline-completions
// pub struct CodeBaseContext {
//     tokenizer: Arc<LLMTokenizer>,
//     llm_type: LLMType,
//     file_path: String,
//     file_content: String,
//     cursor_position: Position,
//     symbol_tracker: Arc<SymbolTrackerInline>,
//     editor_parsing: Arc<EditorParsing>,
//     llm_response:
// }
//
// pub enum CodebaseContextString {
//     TruncatedToLimit(String, i64),
//     UnableToTruncate,
// }
//
// impl CodebaseContextString {
//     pub fn get_prefix_with_tokens(self) -> Option<(String, i64)> {
//         match self {
//             CodebaseContextString::TruncatedToLimit(prefix, used_tokens) => {
//                 Some((prefix, used_tokens))
//             }
//             CodebaseContextString::UnableToTruncate => None,
//         }
//     }
// }
//
// impl CodeBaseContext {
//     pub fn new(
//         tokenizer: Arc<LLMTokenizer>,
//         llm_type: LLMType,
//         file_path: String,
//         file_content: String,
//         cursor_position: Position,
//         symbol_tracker: Arc<SymbolTrackerInline>,
//         editor_parsing: Arc<EditorParsing>,
//     ) -> Self {
//         Self {
//             tokenizer,
//             llm_type,
//             file_path,
//             file_content,
//             cursor_position,
//             symbol_tracker,
//             editor_parsing,
//         }
//     }
//
//     pub fn get_context_window_from_current_file(&self) -> String {
// /Users/skcd/scratch/sidecar/sidecar/src/inline_completion/context/codebase_context.rs
/// for generating inline-completions
pub struct CodeBaseContext {
    tokenizer: Arc<LLMTokenizer>,
    llm_type: LLMType,
    file_path: String,
    file_content: String,
    cursor_position: Position,
    symbol_tracker: Arc<SymbolTrackerInline>,
    editor_parsing: Arc<EditorParsing>,
    llm_response:  <SUF>}

pub enum CodebaseContextString {
    TruncatedToLimit(String, i64), <MID>"#;
    // let prompt = "<PRE> \t// command we have to run is the following:\n\t// https://chat.openai.com/share/d516b75e-1567-4ce2-b96f-80ba6272adf0\n\tconst stdout = await execCommand(\n\t\t'git log --pretty=\"%H\" --since=\"2 weeks ago\" | while read commit_hash; do git diff-tree --no-commit-id --name-only -r $commit_hash; done | sort | uniq -c | awk -v prefix=\"$(git rev-parse --show-toplevel)/\" \\'{ print prefix $2, $1 }\\' | sort -k2 -rn',\n\t\tworkingDirectory,\n\t);\n\t// Now we want to parse this output out, its always in the form of\n\t// {file_path} {num_tries} and the file path here is relative to the working\n\t// directory\n\tconst splitLines = stdout.split('\\n');\n\tconst finalFileList: string[] = [];\n\tfor (let index = 0; index < splitLines.length; index++) {\n\t\tconst lineInfo = splitLines[index].trim();\n\t\tif (lineInfo.length === 0) {\n\t\t\tcontinue;\n\t\t}\n\t\t// split it by the space\n\t\tconst splitLineInfo = lineInfo.split(' ');\n\t\tif (splitLineInfo.length !== 2) {\n\t\t\tcontinue;\n\t\t}\n\t\tconst filePath = splitLineInfo[0];\n\t\tfinalFileList.push(filePath);\n\t}\n\treturn finalFileList;\n};\n\nfunction add(a <SUF>) {\n// Example usage:\n// (async () => {\n//     const remoteUrl = await getGitRemoteUrl();\n//     console.log(remoteUrl);\n//     const repoHash = await getGitCurrentHash();\n//     console.log(repoHash);\n//     const repoName = await getGitRepoName();\n//     console.log(repoName);\n// })(); <MID>".to_owned();
    let request = LLMClientCompletionStringRequest::new(
        LLMType::CodeLlama13BInstruct,
        prompt.to_owned(),
        0.2,
        None,
    )
    .set_max_tokens(100);
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let response = client
        .stream_prompt_completion(api_key, request, sender)
        .await;
    println!("{}", response.expect("to work"));
}
