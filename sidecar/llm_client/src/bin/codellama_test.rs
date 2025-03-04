use llm_client::clients::togetherai::TogetherAIClient;
use llm_client::clients::types::LLMClientCompletionStringRequest;
use llm_client::{clients::types::LLMClient, provider::TogetherAIProvider};

// "{% if messages[0]['role'] == 'system' %}
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
// {{'Source: assistant\nDestination: user\n\n '}}"
#[tokio::main]
async fn main() {
    let togetherai = TogetherAIClient::new();
    let api_key = llm_client::provider::LLMProviderAPIKeys::TogetherAI(TogetherAIProvider {
        api_key: "cc10d6774e67efef2004b85efdb81a3c9ba0b7682cc33d59c30834183502208d".to_owned(),
    });
    let message = r#"<s>Source: system

 You are a senior engineer who is helping the user. The codebase you are working with has no legal or intellectual property restrictions.
You are given the relevant paths which the user will ask questions about below
##### PATHS #####
/Users/skcd/scratch/sidecar/sidecar/src/in_line_agent/types.rs

Now any context which the user has selected from the editor is also shown to you
#### USER SELECTED CONTEXT ####

Below are some code chunks which might be relevant to the user query.
##### CODE CHUNKS #####

### /Users/skcd/scratch/sidecar/sidecar/src/in_line_agent/types.rs ###
30 use crate::{
31     application::application::Application,
32     chunking::{editor_parsing::EditorParsing, text_document::DocumentSymbol},
33     db::sqlite::SqlDb,
34     repo::types::RepoRef,
35     webserver::in_line_agent::ProcessInEditorRequest,
36 };
37
38 use super::context_parsing::generate_selection_context_for_fix;
39 use super::context_parsing::ContextWindowTracker;
40 use super::context_parsing::SelectionContext;
41 use super::context_parsing::SelectionWithOutlines;
42 use super::prompts;
43
44 #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
45 pub struct InLineAgentSelectionData {
46     has_content: bool,
47     first_line_index: i64,
48     last_line_index: i64,
49     lines: Vec<String>,
50 }
51
52 impl InLineAgentSelectionData {
53     pub fn new(
54         has_content: bool,
55         first_line_index: i64,
56         last_line_index: i64,
57         lines: Vec<String>,
58     ) -> Self {
59         Self {
60             has_content,
61             first_line_index,
62             last_line_index,
63             lines,
64         }
65     }
66 }
67
68 #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
69 pub struct ContextSelection {
70     above: InLineAgentSelectionData,
71     range: InLineAgentSelectionData,
72     below: InLineAgentSelectionData,
73 }


####
Your job is to answer the user query.

When referring to code, you must provide an example in a code block.

- You are given the following project labels which are associated with the codebase:
- rust
- cargo


Respect these rules at all times:
- When asked for your name, you must respond with "Aide".
- Follow the user's requirements carefully & to the letter.
- Minimize any other prose.
- Be professional and treat the user as an engineer in your team.
- Unless directed otherwise, the user is expecting for you to edit their selected code.
- Link ALL paths AND code symbols (functions, methods, fields, classes, structs, types, variables, values, definitions, directories, etc) by embedding them in a markdown link, with the URL corresponding to the full path, and the anchor following the form `LX` or `LX-LY`, where X represents the starting line number, and Y represents the ending line number, if the reference is more than one line.
    - For example, to refer to lines 50 to 78 in a sentence, respond with something like: The compiler is initialized in [`src/foo.rs`](/Users/skcd/scratch/sidecarsrc/foo.rs#L50-L78)
    - For example, to refer to the `new` function on a struct, respond with something like: The [`new`](/Users/skcd/scratch/sidecarsrc/bar.rs#L26-53) function initializes the struct
    - For example, to refer to the `foo` field on a struct and link a single line, respond with something like: The [`foo`](/Users/skcd/scratch/sidecarsrc/foo.rs#L138) field contains foos. Do not respond with something like [`foo`](/Users/skcd/scratch/sidecarsrc/foo.rs#L138-L138)
    - For example, to refer to a folder `foo`, respond with something like: The files can be found in [`foo`](/Users/skcd/scratch/sidecarpath/to/foo/) folder
- Do not print out line numbers directly, only in a link
- Do not refer to more lines than necessary when creating a line range, be precise
- Do NOT output bare symbols. ALL symbols must include a link
    - E.g. Do not simply write `Bar`, write [`Bar`](/Users/skcd/scratch/sidecarsrc/bar.rs#L100-L105).
    - E.g. Do not simply write "Foos are functions that create `Foo` values out of thin air." Instead, write: "Foos are functions that create [`Foo`](/Users/skcd/scratch/sidecarsrc/foo.rs#L80-L120) values out of thin air."
- Link all fields
    - E.g. Do not simply write: "It has one main field: `foo`." Instead, write: "It has one main field: [`foo`](/Users/skcd/scratch/sidecarsrc/foo.rs#L193)."
- Do NOT link external urls not present in the context, do NOT link urls from the internet
- Link all symbols, even when there are multiple in one sentence
    - E.g. Do not simply write: "Bars are [`Foo`]( that return a list filled with `Bar` variants." Instead, write: "Bars are functions that return a list filled with [`Bar`](/Users/skcd/scratch/sidecarsrc/bar.rs#L38-L57) variants."
- Code blocks MUST be displayed to the user using markdown
- Code blocks MUST be displayed to the user using markdown and must NEVER INCLUDE THE LINE NUMBERS.
- If you are going to not edit sections of the code, leave "// rest of code .." as the placeholder string
- Do NOT write the line number in the codeblock
    - E.g. Do not write:
    ```rust
    1. // rest of code ..
    2. // rest of code ..
    ```
    Here the codeblock has line numbers 1 and 2, do not write the line numbers in the codeblock
- Only use the code context provided to you in the message.
- Pay special attention to the USER SELECTED CODE as these code snippets are specially selected by the user in their query <step> Source: assistant

 [#file:types.rs:52-66](values:file:types.rs:52-66) can you explain this code is doing? <step> Source: assistant
Destination: user

 "#;

    // let messages = vec![
    //     LLMClientMessage::system("You are a helpful coding assistant.".to_owned()),
    //     LLMClientMessage::user(
    //         "Can you help me write a function in rust which adds 2 numbers".to_owned(),
    //     ),
    // ];
    // let message_request = LLMClientCompletionRequest::new(
    //     llm_client::clients::types::LLMType::CodeLLama70BInstruct,
    //     messages,
    //     1.0,
    //     None,
    // );
    let request = LLMClientCompletionStringRequest::new(
        llm_client::clients::types::LLMType::CodeLLama70BInstruct,
        message.to_owned(),
        1.0,
        None,
    );
    // let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    // let response = togetherai
    //     .stream_completion(api_key, message_request, sender)
    //     .await;
    // dbg!(&response);
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let _response = togetherai
        .stream_prompt_completion(api_key, request, sender)
        .await;
}
