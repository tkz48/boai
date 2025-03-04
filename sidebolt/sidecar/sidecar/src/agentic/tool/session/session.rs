//! We can create a new session over here and its composed of exchanges
//! The exchanges can be made by the human or the agent

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use futures::StreamExt;
use llm_client::clients::types::LLMType;
use tokio::io::AsyncWriteExt;

use crate::{
    agentic::{
        symbol::{
            errors::SymbolError,
            events::{
                edit::SymbolToEdit,
                human::HumanAgenticRequest,
                message_event::{SymbolEventMessage, SymbolEventMessageProperties},
            },
            identifier::SymbolIdentifier,
            manager::SymbolManager,
            scratch_pad::ScratchPadAgent,
            tool_box::ToolBox,
            tool_properties::ToolProperties,
            types::SymbolEventRequest,
            ui_event::UIEventWithID,
        },
        tool::{
            devtools::screenshot::RequestScreenshotInput,
            errors::ToolError,
            file::semantic_search::SemanticSearchRequest,
            helpers::diff_recent_changes::DiffFileContent,
            input::{ToolInput, ToolInputPartial},
            lsp::{
                file_diagnostics::DiagnosticMap, find_files::FindFilesRequest,
                list_files::ListFilesInput, open_file::OpenFileRequest,
                search_file::SearchFileContentInput,
            },
            mcp::input::McpToolInput,
            plan::{
                generator::{Step, StepSenderEvent},
                service::{PlanService, PlanServiceError},
            },
            r#type::{Tool, ToolType},
            repo_map::generator::RepoMapGeneratorRequest,
            session::tool_use_agent::ToolUseAgentContextCrunchingInput,
            terminal::terminal::TerminalInput,
            test_runner::runner::TestRunnerRequest,
        },
    },
    chunking::text_document::{Position, Range},
    mcts::action_node::ActionNode,
    repo::types::RepoRef,
    user_context::types::{ImageInformation, UserContext},
};

use super::{
    chat::{
        SessionChatClientRequest, SessionChatMessage, SessionChatMessageImage,
        SessionChatToolReturn, SessionChatToolUse,
    },
    hot_streak::SessionHotStreakRequest,
    tool_use_agent::{
        ToolUseAgent, ToolUseAgentInput, ToolUseAgentInputOnlyTools, ToolUseAgentOutput,
        ToolUseAgentOutputType, ToolUseAgentReasoningInput, ToolUseAgentReasoningParams,
    },
};

#[derive(Debug)]
pub enum AgentToolUseOutput {
    // Tool input, Tool Use id, Updated session
    Success((ToolInputPartial, String, Session)),
    Failed(String),
    Cancelled,
    Errored(SymbolError),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum AideAgentMode {
    Edit = 1,
    Plan = 2,
    Chat = 3,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AideEditMode {
    Anchored = 1,
    Agentic = 2,
}

/// The exchange can be in one of the states
///
/// Its either that the edits made were accepted or rejected
/// it could also have been cancelled by the user
/// Default when its created is in running state
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExchangeState {
    Accepted,
    Rejected,
    Cancelled,
    Running,
    UserMessage,
}

impl Default for ExchangeState {
    fn default() -> Self {
        ExchangeState::Running
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExchangeType {
    HumanChat(ExchangeTypeHuman),
    AgentChat(ExchangeTypeAgent),
    // what do we store over here for the anchored edit, it can't just be the
    // user query? we probably have to store the snippet we were trying to edit
    // as well
    Edit(ExchangeTypeEdit),
    Plan(ExchangeTypePlan),
    ToolOutput(ExchangeTypeToolOutput),
}

// TODO(codestory): The user is probably going to add more context over here as they
// keep iterating with their requests over here, we have to do something about it
// or we can keep it simple and just make it so that we store the previous iterations over here
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeTypePlan {
    previous_queries: Vec<String>,
    query: String,
    user_context: UserContext,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeEditInformationAgentic {
    query: String,
    codebase_search: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeEditInformationAnchored {
    query: String,
    fs_file_path: String,
    range: Range,
    selection_context: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExchangeEditInformation {
    Agentic(ExchangeEditInformationAgentic),
    Anchored(ExchangeEditInformationAnchored),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeTypeEdit {
    information: ExchangeEditInformation,
    user_context: UserContext,
    exchange_type: AideEditMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeTypeHuman {
    query: String,
    user_context: UserContext,
    project_labels: Vec<String>,
    repo_ref: RepoRef,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeTypeToolOutput {
    tool_type: ToolType,
    output: String,
    exchange_id: String,
    user_context: UserContext,
    tool_use_id: String,
}

impl ExchangeTypeToolOutput {
    pub fn new(
        tool_type: ToolType,
        output: String,
        exchange_id: String,
        user_context: UserContext,
        tool_use_id: String,
    ) -> Self {
        Self {
            tool_type,
            output,
            exchange_id,
            user_context,
            tool_use_id,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeReplyAgentPlan {
    plan_steps: Vec<Step>,
    // plan discarded over here represents the fact that the plan we CANCELLED
    // it had other meanings but thats what we are going with now 🔫
    plan_discarded: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeReplyAgentChat {
    reply: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeReplyAgentEdit {
    edits_made_diff: String,
    accepted: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeReplyAgentTool {
    tool_type: ToolType,
    // we need some kind of partial tool input over here as well so we can parse
    // the data out properly
    // for now, I am leaving things here until I can come up with a proper API for that
    tool_input_partial: ToolInputPartial,
    thinking: String,
    // The tool use id which we need to send back along with the tool parameters
    // if they are present
    tool_use_id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExchangeReplyAgent {
    Plan(ExchangeReplyAgentPlan),
    Chat(ExchangeReplyAgentChat),
    Edit(ExchangeReplyAgentEdit),
    Tool(ExchangeReplyAgentTool),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeTypeAgent {
    reply: ExchangeReplyAgent,
    /// This points to the exchange id which we are replying to as the agent
    parent_exchange_id: String,
}

impl ExchangeTypeAgent {
    fn chat_reply(reply: String, parent_exchange_id: String) -> Self {
        Self {
            reply: ExchangeReplyAgent::Chat(ExchangeReplyAgentChat { reply }),
            parent_exchange_id,
        }
    }

    fn plan_reply(steps: Vec<Step>, parent_exchange_id: String) -> Self {
        Self {
            reply: ExchangeReplyAgent::Plan(ExchangeReplyAgentPlan {
                plan_steps: steps,
                plan_discarded: false,
            }),
            parent_exchange_id,
        }
    }

    fn edits_reply(edits_made: String, parent_exchange_id: String) -> Self {
        Self {
            reply: ExchangeReplyAgent::Edit(ExchangeReplyAgentEdit {
                edits_made_diff: edits_made,
                accepted: false,
            }),
            parent_exchange_id,
        }
    }

    fn tool_use(
        tool_input_partial: ToolInputPartial,
        tool_type: ToolType,
        thinking: String,
        parent_exchange_id: String,
        tool_use_id: String,
    ) -> Self {
        Self {
            reply: ExchangeReplyAgent::Tool(ExchangeReplyAgentTool {
                tool_type,
                tool_input_partial,
                thinking,
                tool_use_id,
            }),
            parent_exchange_id,
        }
    }
}

impl ExchangeTypeHuman {
    pub fn new(
        query: String,
        user_context: UserContext,
        project_labels: Vec<String>,
        repo_ref: RepoRef,
    ) -> Self {
        Self {
            query,
            user_context,
            project_labels,
            repo_ref,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Exchange {
    exchange_id: String,
    exchange_type: ExchangeType,
    #[serde(default)]
    exchange_state: ExchangeState,
    /// is_compressed implies that the exchange has been compressed by another
    /// in the future, this is esentially used to not keep polluting the context
    /// window for the agentic llm and make sure that the context window
    /// stays smallish and nimble
    #[serde(default)]
    is_compressed: bool,
    /// is_hidden implies that the user has been going through the undo/redo
    /// motions, and the exchanges that are undone are marked as hidden.
    /// Importantly, when the user sends a new message with some hidden messages,
    /// they get permanently truncated.
    #[serde(default)]
    is_hidden: bool,
}

impl Exchange {
    pub fn exchange_id(&self) -> &str {
        &self.exchange_id
    }

    fn human_chat(
        exchange_id: String,
        query: String,
        user_context: UserContext,
        project_labels: Vec<String>,
        repo_ref: RepoRef,
    ) -> Self {
        Self {
            exchange_id,
            exchange_type: ExchangeType::HumanChat(ExchangeTypeHuman::new(
                query,
                user_context,
                project_labels,
                repo_ref,
            )),
            exchange_state: ExchangeState::UserMessage,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn plan_request(exchange_id: String, query: String, user_context: UserContext) -> Self {
        Self {
            exchange_id,
            exchange_type: ExchangeType::Plan(ExchangeTypePlan {
                previous_queries: vec![],
                query,
                user_context,
            }),
            exchange_state: ExchangeState::UserMessage,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn agentic_edit(
        exchange_id: String,
        query: String,
        codebase_search: bool,
        user_context: UserContext,
    ) -> Self {
        Self {
            exchange_id,
            exchange_type: ExchangeType::Edit(ExchangeTypeEdit {
                information: ExchangeEditInformation::Agentic(ExchangeEditInformationAgentic {
                    query,
                    codebase_search,
                }),
                user_context,
                exchange_type: AideEditMode::Agentic,
            }),
            exchange_state: ExchangeState::UserMessage,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn anchored_edit(
        exchange_id: String,
        query: String,
        user_context: UserContext,
        range: Range,
        fs_file_path: String,
        selection_context: String,
    ) -> Self {
        Self {
            exchange_id,
            exchange_type: ExchangeType::Edit(ExchangeTypeEdit {
                information: ExchangeEditInformation::Anchored(ExchangeEditInformationAnchored {
                    query,
                    fs_file_path,
                    range,
                    selection_context,
                }),
                user_context,
                exchange_type: AideEditMode::Anchored,
            }),
            exchange_state: ExchangeState::UserMessage,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn agent_chat_reply(parent_exchange_id: String, exchange_id: String, message: String) -> Self {
        Self {
            exchange_id,
            exchange_type: ExchangeType::AgentChat(ExchangeTypeAgent::chat_reply(
                message,
                parent_exchange_id,
            )),
            exchange_state: ExchangeState::Running,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn agent_plan_reply(parent_exchange_id: String, exchange_id: String, steps: Vec<Step>) -> Self {
        Self {
            exchange_id,
            exchange_type: ExchangeType::AgentChat(ExchangeTypeAgent::plan_reply(
                steps,
                parent_exchange_id,
            )),
            exchange_state: ExchangeState::Running,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn agent_edits_reply(
        parent_exchange_id: String,
        exchange_id: String,
        edits_response: String,
    ) -> Self {
        Self {
            exchange_id,
            exchange_type: ExchangeType::AgentChat(ExchangeTypeAgent::edits_reply(
                edits_response,
                parent_exchange_id,
            )),
            exchange_state: ExchangeState::Running,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn agent_tool_use(
        parent_exchange_id: String,
        exchange_id: String,
        tool_input: ToolInputPartial,
        tool_type: ToolType,
        thinking: String,
        tool_use_id: String,
    ) -> Self {
        Self {
            exchange_id,
            exchange_type: ExchangeType::AgentChat(ExchangeTypeAgent::tool_use(
                tool_input,
                tool_type,
                thinking,
                parent_exchange_id,
                tool_use_id,
            )),
            exchange_state: ExchangeState::Running,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn tool_output(
        exchange_id: String,
        tool_type: ToolType,
        output: String,
        user_context: UserContext,
        tool_use_id: String,
    ) -> Self {
        Self {
            exchange_id: exchange_id.to_owned(),
            exchange_type: ExchangeType::ToolOutput(ExchangeTypeToolOutput::new(
                tool_type,
                output,
                exchange_id.clone(),
                user_context,
                tool_use_id,
            )),
            exchange_state: ExchangeState::Running,
            is_compressed: false,
            is_hidden: false,
        }
    }

    fn set_completion_status(mut self, accetped: bool) -> Self {
        if accetped {
            self.exchange_state = ExchangeState::Accepted;
        } else {
            self.exchange_state = ExchangeState::Rejected;
        }
        self
    }

    /// If the exchange has been left open and has not finished yet
    fn is_open(&self) -> bool {
        matches!(self.exchange_state, ExchangeState::Running)
            && matches!(self.exchange_type, ExchangeType::AgentChat(_))
    }

    /// Check if this is agent reply
    fn is_agent_work(&self) -> bool {
        matches!(self.exchange_type, ExchangeType::AgentChat(_))
    }

    fn is_still_running(&self) -> bool {
        matches!(self.exchange_state, ExchangeState::Running)
    }

    /// Assume that we will implement this later but we still have code edits
    /// everywhere
    fn has_code_edits(&self) -> bool {
        true
    }

    fn set_exchange_as_cancelled(&mut self) {
        self.exchange_state = ExchangeState::Cancelled;
    }

    /// Convert the exchange to a session chat message so we can send it over
    /// for inference
    ///
    /// We can have consecutive human messages now on every API so this is no
    /// longer a big worry
    async fn to_conversation_message(&self, is_json_mode: bool) -> Option<SessionChatMessage> {
        if self.is_compressed || self.is_hidden {
            return None;
        }
        let session_chat_message = match &self.exchange_type {
            ExchangeType::HumanChat(ref chat_message) => {
                // TODO(skcd): Figure out caching etc later on
                let prompt = chat_message.query.to_owned();
                let images = chat_message
                    .user_context
                    .images()
                    .into_iter()
                    .map(|user_context_image| {
                        SessionChatMessageImage::new(
                            user_context_image.r#type().to_owned(),
                            user_context_image.media_type().to_owned(),
                            user_context_image.data().to_owned(),
                        )
                    })
                    .collect();
                // TODO(skcd): We also need to pass messages here for the images which we are getting
                SessionChatMessage::user(prompt, images)
            }
            ExchangeType::AgentChat(ref chat_message) => {
                // This completely breaks we have to figure out how to covert
                // the various types of exchanges to a string here for passing
                // around as context
                let reply = chat_message.reply.clone();
                match reply {
                    ExchangeReplyAgent::Chat(chat_reply) => {
                        SessionChatMessage::assistant(chat_reply.reply.to_owned(), vec![])
                    }
                    ExchangeReplyAgent::Edit(edit_reply) => {
                        if edit_reply.accepted {
                            SessionChatMessage::assistant(
                                edit_reply.edits_made_diff.to_owned(),
                                vec![],
                            )
                        } else {
                            let edits_made = edit_reply.edits_made_diff.to_owned();
                            SessionChatMessage::assistant(
                                format!(
                                    r#"I made the following edits and the user REJECTED them
{edits_made}"#
                                ),
                                vec![],
                            )
                        }
                    }
                    ExchangeReplyAgent::Plan(plan_reply) => {
                        if plan_reply.plan_discarded {
                            SessionChatMessage::assistant(
                                "The Plan I came up with was REJECTED by the user".to_owned(),
                                vec![],
                            )
                        } else {
                            let plan_steps = plan_reply
                                .plan_steps
                                .into_iter()
                                .map(|step| {
                                    let step_title = step.title.to_owned();
                                    let step_description = step.description();
                                    let files_to_edit = step
                                        .file_to_edit()
                                        .unwrap_or("File to edit not present".to_owned());
                                    format!(
                                        r#"<step>
<files_to_edit>
<file>
{files_to_edit}
</file>
</files_to_edit>
<title>
{step_title}
</title>
<changes>
{step_description}
</changes>
</step>"#
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            SessionChatMessage::assistant(
                                format!(
                                    "I came up with the plan below and the user was happy
{plan_steps}"
                                ),
                                vec![],
                            )
                        }
                    }
                    ExchangeReplyAgent::Tool(tool_input) => {
                        if is_json_mode {
                            let tool_input_parameters =
                                &tool_input.tool_input_partial.to_json_value();
                            match tool_input_parameters {
                                Some(schema) => {
                                    // TODO(skcd): Figure out if the thinking here is in proper tags
                                    SessionChatMessage::assistant(
                                        tool_input.thinking.to_owned(),
                                        vec![],
                                    )
                                    .insert_tool_use(
                                        SessionChatToolUse::new(
                                            tool_input
                                                .tool_input_partial
                                                .to_tool_type()
                                                .to_string(),
                                            tool_input.tool_use_id.to_owned(),
                                            schema.clone(),
                                        ),
                                    )
                                }
                                None => SessionChatMessage::assistant(
                                    tool_input.tool_input_partial.to_string(),
                                    vec![],
                                ),
                            }
                        } else {
                            let tool_input_parameters = &tool_input.tool_input_partial;
                            let thinking = &tool_input.thinking;
                            SessionChatMessage::assistant(
                                format!(
                                    r#"<thinking>
{thinking}
</thinking>
{}"#,
                                    tool_input_parameters.to_string()
                                ),
                                vec![],
                            )
                        }
                    }
                }
            }
            ExchangeType::ToolOutput(ref tool_output) => {
                if is_json_mode {
                    SessionChatMessage::user("".to_owned(), vec![]).insert_tool_return_value(
                        SessionChatToolReturn::new(
                            tool_output.tool_use_id.to_owned(),
                            tool_output.tool_type.to_string(),
                            tool_output.output.to_owned(),
                        ),
                    )
                } else {
                    let images = tool_output
                        .user_context
                        .images()
                        .into_iter()
                        .map(|user_context_image| {
                            SessionChatMessageImage::new(
                                user_context_image.r#type().to_owned(),
                                user_context_image.media_type().to_owned(),
                                user_context_image.data().to_owned(),
                            )
                        })
                        .collect();
                    SessionChatMessage::user(
                        format!(
                            // tool output is generally an observation for the AI
                            // this looks like a small nit in the naming of the output but has
                            // big consequences
                            "Observation ({}): {}",
                            tool_output.tool_type.to_string(),
                            tool_output.output,
                        ),
                        images,
                    )
                }
            }
            ExchangeType::Plan(ref plan) => {
                let user_query = &plan.query;
                SessionChatMessage::user(
                    format!(
                        r#"I want a plan of edits to help solve this:
{user_query}"#
                    ),
                    vec![],
                )
            }
            ExchangeType::Edit(ref anchored_edit) => {
                let edit_information = &anchored_edit.information;
                let user_query = match edit_information {
                    ExchangeEditInformation::Agentic(agentic_edit) => {
                        let query = agentic_edit.query.to_owned();
                        format!(
                            r#"I want you to perform edits for my query:
<query>
{query}
</query>"#
                        );
                        query
                    }
                    ExchangeEditInformation::Anchored(anchored_edit) => {
                        let fs_file_path = anchored_edit.fs_file_path.to_owned();
                        let start_line = anchored_edit.range.start_line();
                        let end_line = anchored_edit.range.end_line();
                        let location = format!(r#"{fs_file_path}-{start_line}:{end_line}"#);
                        let query = anchored_edit.query.to_owned();
                        format!(
                            r#"I want to perform edits at {location}
<query>
{query}
</query>"#
                        )
                    }
                };
                SessionChatMessage::user(user_query, vec![])
            }
        };
        Some(session_chat_message)
    }

    /// Hot streak worthy message gets access to the diagnostics and allows the
    /// agent to auto-generaate a reply
    pub fn is_hot_streak_worthy_message(&self) -> bool {
        let exchange_type = &self.exchange_type;
        match exchange_type {
            ExchangeType::AgentChat(agent_chat) => match agent_chat.reply {
                ExchangeReplyAgent::Edit(_) | ExchangeReplyAgent::Plan(_) => true,
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Session {
    session_id: String,
    project_labels: Vec<String>,
    repo_ref: RepoRef,
    exchanges: Vec<Exchange>,
    storage_path: String,
    global_running_user_context: UserContext,
    tools: Vec<ToolType>,
    #[serde(default)]
    action_nodes: Vec<ActionNode>,
}

impl Session {
    pub fn new(
        session_id: String,
        project_labels: Vec<String>,
        repo_ref: RepoRef,
        storage_path: String,
        global_running_user_context: UserContext,
        tools: Vec<ToolType>,
    ) -> Self {
        Self {
            session_id,
            project_labels,
            repo_ref,
            exchanges: vec![],
            storage_path,
            global_running_user_context,
            tools,
            action_nodes: vec![],
        }
    }

    pub fn last_reasoning_node_list(&self) -> Vec<&ActionNode> {
        self.action_nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                // finds the last node which was a reasoning node
                node.action()
                    .map(|action| action.to_tool_type())
                    .flatten()
                    .map(|tool_type| tool_type == ToolType::Reasoning)
                    .unwrap_or_default()
            })
            .map(|(_idx, action_node)| action_node)
            .collect()
    }

    pub fn last_reasoning_node_if_any(&self) -> Option<usize> {
        self.action_nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                // finds the last node which was a reasoning node
                node.action()
                    .map(|action| action.to_tool_type())
                    .flatten()
                    .map(|tool_type| tool_type == ToolType::Reasoning)
                    .unwrap_or_default()
            })
            .map(|(idx, _)| idx)
            .last()
    }

    pub fn repo_ref(&self) -> &RepoRef {
        &self.repo_ref
    }

    pub fn add_action_node(&mut self, action_node: ActionNode) {
        self.action_nodes.push(action_node);
    }

    pub fn action_nodes(&self) -> &[ActionNode] {
        self.action_nodes.as_slice()
    }

    pub fn reset_exchanges(&mut self) {
        self.exchanges = vec![];
    }

    /// Updates the tools which are present in the session
    pub fn set_tools(mut self, tools: Vec<ToolType>) -> Self {
        self.tools = tools;
        self
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn storage_path(&self) -> &str {
        &self.storage_path
    }

    pub fn exchanges(&self) -> usize {
        self.exchanges.len()
    }

    fn find_exchange_by_id(&self, exchange_id: &str) -> Option<&Exchange> {
        self.exchanges
            .iter()
            .find(|exchange| &exchange.exchange_id == exchange_id)
    }

    fn find_exchange_by_id_mut(&mut self, exchange_id: &str) -> Option<&mut Exchange> {
        self.exchanges
            .iter_mut()
            .find(|exchange| &exchange.exchange_id == exchange_id)
    }

    /// Finds the exchange we are interested in and mutates the previous queries
    /// and the current query
    pub fn plan_iteration(
        mut self,
        exchange_id: String,
        query: String,
        user_context: UserContext,
    ) -> Session {
        self.global_running_user_context = self
            .global_running_user_context
            .merge_user_context(user_context.clone());
        let exchange_to_change = self
            .exchanges
            .iter_mut()
            .find(|exchange| exchange.exchange_id == exchange_id);
        if let Some(exchange_to_change) = exchange_to_change {
            match &mut exchange_to_change.exchange_type {
                ExchangeType::Plan(plan_exchange) => {
                    let mut previous_queries = plan_exchange.previous_queries.to_vec();
                    previous_queries.push(plan_exchange.query.to_owned());
                    plan_exchange.query = query;
                    plan_exchange.previous_queries = previous_queries;
                    plan_exchange.user_context = user_context;
                }
                _ => {}
            }
        }
        self
    }

    pub fn plan(
        mut self,
        exchange_id: String,
        query: String,
        user_context: UserContext,
    ) -> Session {
        self.global_running_user_context = self
            .global_running_user_context
            .merge_user_context(user_context.clone());
        let exchange = Exchange::plan_request(exchange_id, query, user_context);
        self.exchanges.push(exchange);
        self
    }

    pub fn get_parent_exchange_id(&self, exchange_id: &str) -> Option<Exchange> {
        self.exchanges
            .iter()
            .find(|exchange| &exchange.exchange_id == exchange_id)
            .map(|exchange| match &exchange.exchange_type {
                ExchangeType::AgentChat(ref agent_chat) => {
                    Some(agent_chat.parent_exchange_id.to_owned())
                }
                _ => None,
            })
            .flatten()
            .map(|parent_exchange_id| self.get_exchange_by_id(&parent_exchange_id))
            .flatten()
    }

    pub fn get_exchange_by_id(&self, exchange_id: &str) -> Option<Exchange> {
        self.exchanges
            .iter()
            .find(|exchange| &exchange.exchange_id == exchange_id)
            .cloned()
    }

    pub fn agentic_edit(
        mut self,
        exchange_id: String,
        query: String,
        user_context: UserContext,
        codebase_search: bool,
    ) -> Session {
        self.global_running_user_context = self
            .global_running_user_context
            .merge_user_context(user_context.clone());
        let exchange = Exchange::agentic_edit(exchange_id, query, codebase_search, user_context);
        self.exchanges.push(exchange);
        self
    }

    pub fn anchored_edit(
        mut self,
        exchange_id: String,
        query: String,
        user_context: UserContext,
        range: Range,
        fs_file_path: String,
        file_content_in_selection: String,
    ) -> Session {
        self.global_running_user_context = self
            .global_running_user_context
            .merge_user_context(user_context.clone());
        let exchange = Exchange::anchored_edit(
            exchange_id,
            query,
            user_context,
            range,
            fs_file_path,
            file_content_in_selection,
        );
        self.exchanges.push(exchange);
        self
    }

    pub fn tool_output(
        mut self,
        exchange_id: &str,
        tool_type: ToolType,
        output: String,
        user_context: UserContext,
        tool_use_id: String,
    ) -> Self {
        self.global_running_user_context = self
            .global_running_user_context
            .merge_user_context(user_context.clone());
        let exchange = Exchange::tool_output(
            exchange_id.to_owned(),
            tool_type,
            output,
            user_context,
            tool_use_id,
        );
        self.exchanges.push(exchange);
        self
    }

    // creates a message which tells the agent that this is a PR description
    pub fn pr_description(mut self, exchange_id: String, human_message: String) -> Session {
        let user_message = format!(
            r#"<pr_description>
{human_message}
</pr_description>"#
        );

        // add the action node
        let mut action_node = ActionNode::default_with_index(self.exchanges());
        action_node = action_node
            .set_message(human_message)
            .update_user_context(UserContext::default());
        self.action_nodes.push(action_node);

        // push the exchange
        let exchange = Exchange::human_chat(
            exchange_id,
            user_message,
            UserContext::default(),
            self.project_labels.to_vec(),
            self.repo_ref.clone(),
        );
        self.exchanges.push(exchange);
        self
    }

    pub async fn human_message_tool_use(
        mut self,
        exchange_id: String,
        human_message: String,
        all_files: Vec<String>,
        open_files: Vec<String>,
        _shell: String,
        user_context: UserContext,
    ) -> Session {
        let user_message = format!(
            r#"<editor_status>
<open_files>
{}
</open_files>
<visible_files>
{}
</visible_files>
</editor_status>
<user_provided_context>
{}
</user_provided_context>
<user_query>
{}
</user_query>"#,
            all_files.join("\n"),
            open_files.join("\n"),
            user_context
                .clone()
                .to_xml(HashSet::default())
                .await
                .unwrap_or_default(),
            human_message.to_owned()
        );

        // add the action node
        let mut action_node = ActionNode::default_with_index(self.exchanges());
        action_node = action_node
            .set_message(human_message)
            .update_user_context(user_context.clone());
        self.action_nodes.push(action_node);

        // push the exchange
        let exchange = Exchange::human_chat(
            exchange_id,
            user_message,
            user_context,
            self.project_labels.to_vec(),
            self.repo_ref.clone(),
        );
        self.exchanges.push(exchange);
        self
    }

    pub fn human_message(
        mut self,
        exchange_id: String,
        human_message: String,
        user_context: UserContext,
        project_labels: Vec<String>,
        repo_ref: RepoRef,
    ) -> Session {
        self.global_running_user_context = self
            .global_running_user_context
            .merge_user_context(user_context.clone());
        let exchange = Exchange::human_chat(
            exchange_id,
            human_message,
            user_context,
            project_labels,
            repo_ref,
        );
        self.exchanges.push(exchange);
        self
    }

    fn last_exchange(&self) -> Option<&Exchange> {
        self.exchanges.last()
    }

    pub async fn undo_including_exchange_id(
        mut self,
        exchange_id: &str,
    ) -> Result<Self, SymbolError> {
        // keep grabbing the exchanges until we hit the exchange_id we are interested in
        // over  here, that become our new exchange
        let new_exchanges = self
            .exchanges
            .into_iter()
            .take_while(|exchange| &exchange.exchange_id != exchange_id)
            .collect::<Vec<_>>();
        self.exchanges = new_exchanges;
        Ok(self)
    }

    pub async fn move_to_checkpoint(mut self, exchange_id: &str) -> Result<Self, SymbolError> {
        // Find the index of the target exchange
        let target_index = self
            .exchanges
            .iter()
            .position(|exchange| &exchange.exchange_id == exchange_id);

        if let Some(target_index) = target_index {
            // Mark exchanges based on their position relative to the checkpoint
            for (i, exchange) in self.exchanges.iter_mut().enumerate() {
                // If target_index is 0, we want to compress it and everything after
                // Otherwise, we only want to compress everything after the target_index
                let will_compress = if target_index == 0 {
                    i >= target_index
                } else {
                    i > target_index
                };
                exchange.is_hidden = will_compress;
            }
        }
        Ok(self)
    }

    pub async fn react_to_feedback(
        mut self,
        exchange_id: &str,
        step_index: Option<usize>,
        accepted: bool,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Self, SymbolError> {
        // We have to do a couple of things here, since for plans we might have partial
        // acceptance
        // - find the step list of the plan until which we have accepted the changes
        // - if its an anchored edit then mark it completely accepted or rejected
        // Here first we make sure that an exchange of the form exists
        // if it does we mark that exchange as closed and also update its state
        self.exchanges = self
            .exchanges
            .into_iter()
            .map(|exchange| {
                if &exchange.exchange_id == exchange_id {
                    // we have an exchange over here matching our id
                    // now we need to carefully understand if its a plan or if its an edit
                    // if its a plan we should accept it until that step and discard all the steps
                    // post that index
                    match exchange.exchange_type {
                        ExchangeType::AgentChat(agent_exchange) => {
                            let parent_exchange_id = agent_exchange.parent_exchange_id.to_owned();
                            let exchange_reply = match agent_exchange.reply {
                                ExchangeReplyAgent::Plan(mut plan_step) => {
                                    if let Some(step_index) = step_index {
                                        // now here only keep the steps until the index we are interested in
                                        if step_index == 0 {
                                            plan_step.plan_discarded = true;
                                        } else {
                                            plan_step.plan_steps.truncate(step_index + 1);
                                        }
                                    }
                                    ExchangeReplyAgent::Plan(plan_step)
                                }
                                ExchangeReplyAgent::Edit(mut edit_step) => {
                                    edit_step.accepted = accepted;
                                    ExchangeReplyAgent::Edit(edit_step)
                                }
                                ExchangeReplyAgent::Chat(chat_step) => {
                                    ExchangeReplyAgent::Chat(chat_step)
                                }
                                ExchangeReplyAgent::Tool(tools) => ExchangeReplyAgent::Tool(tools),
                            };
                            Exchange {
                                exchange_id: exchange_id.to_owned(),
                                exchange_type: ExchangeType::AgentChat(ExchangeTypeAgent {
                                    reply: exchange_reply,
                                    parent_exchange_id,
                                }),
                                exchange_state: exchange.exchange_state,
                                is_compressed: exchange.is_compressed,
                                is_hidden: exchange.is_hidden,
                            }
                        }
                        _ => exchange,
                    }
                } else {
                    exchange
                }
            })
            .collect();

        let exchange_to_react = self
            .exchanges
            .iter()
            .find(|exchange| &exchange.exchange_id == exchange_id)
            .map(|exchange| match &exchange.exchange_type {
                ExchangeType::AgentChat(agentic_chat) => match agentic_chat.reply {
                    ExchangeReplyAgent::Chat(_) => None,
                    ExchangeReplyAgent::Edit(_) => Some(AideAgentMode::Edit),
                    ExchangeReplyAgent::Plan(_) => Some(AideAgentMode::Plan),
                    ExchangeReplyAgent::Tool(_) => None,
                },
                _ => None,
            })
            .flatten();

        // give feedback to the editor that our state has changed
        if accepted {
            if matches!(exchange_to_react, Some(AideAgentMode::Plan)) {
                let _ = message_properties
                    .ui_sender()
                    .send(UIEventWithID::plan_as_accepted(
                        self.session_id.to_owned(),
                        exchange_id.to_owned(),
                    ));
            }
            if matches!(exchange_to_react, Some(AideAgentMode::Edit)) {
                let _ = message_properties
                    .ui_sender()
                    .send(UIEventWithID::edits_accepted(
                        self.session_id.to_owned(),
                        exchange_id.to_owned(),
                    ));
            }
        } else {
            if matches!(exchange_to_react, Some(AideAgentMode::Plan)) {
                let _ = message_properties
                    .ui_sender()
                    .send(UIEventWithID::plan_as_cancelled(
                        self.session_id.to_owned(),
                        exchange_id.to_owned(),
                    ));
            }
            if matches!(exchange_to_react, Some(AideAgentMode::Edit)) {
                let _ = message_properties.ui_sender().send(
                    UIEventWithID::edits_cancelled_in_exchange(
                        self.session_id.to_owned(),
                        exchange_id.to_owned(),
                    ),
                );
            }
        }

        // now close the exchange
        println!("session::react_to_feedback::finished_exchange");
        let _ = message_properties
            .ui_sender()
            .send(UIEventWithID::finished_exchange(
                self.session_id.to_owned(),
                message_properties.request_id_str().to_owned(),
            ));
        Ok(self)
    }

    pub async fn context_crunching(
        &self,
        tool_use_agent: ToolUseAgent,
        original_user_instruction: String,
        // where is the current agent iteration starting from
        last_reasoning_action_node_from: usize,
        // reasoning action nodes
        reasoning_action_nodes: Vec<&ActionNode>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<ToolUseAgentOutput, SymbolError> {
        let mut agent_action_nodes = self.action_nodes[last_reasoning_action_node_from..]
            .into_iter()
            .map(|action_node| action_node.clone())
            .collect::<Vec<_>>();

        // remove the first node from agent_action_nodes if it is a reasoning node
        // since we are starting from the last_reasoning_action_node_from
        if agent_action_nodes
            .first()
            .map(|action_node| action_node.tool_type() == Some(ToolType::ContextCrunching))
            .unwrap_or_default()
        {
            if !agent_action_nodes.is_empty() {
                // this removes the first element if it is present
                agent_action_nodes.remove(0);
            }
        }

        let context_crunching_input = ToolUseAgentContextCrunchingInput::new(
            original_user_instruction,
            agent_action_nodes,
            reasoning_action_nodes
                .into_iter()
                .map(|action_node| action_node.clone())
                .collect(),
            message_properties,
        );

        tool_use_agent
            .context_crunching(context_crunching_input)
            .await
    }

    /// We perform the reasoning here on the agent and grab the output and pass that
    /// as the new instruction for the agent
    pub async fn get_reasoning_instruction(
        self,
        tool_use_agent: ToolUseAgent,
        user_instruction: String,
        action_nodes_from: usize,
        tool_use_reasoning_input: Option<ToolUseAgentReasoningParams>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<ToolUseAgentReasoningParams, SymbolError> {
        // for reasoning the flow looks like the following:
        // there is a user message which is always present
        // then there are messages which the agent has taken
        // and finally there is the
        let action_nodes = self.action_nodes[action_nodes_from..]
            .into_iter()
            .map(|action_node| action_node.clone())
            .collect::<Vec<_>>();

        let tool_reasoning_input = ToolUseAgentReasoningInput::new(
            user_instruction,
            action_nodes,
            tool_use_reasoning_input,
            message_properties,
        );

        tool_use_agent.reasoning_output(tool_reasoning_input).await
    }

    pub async fn get_tool_to_use(
        mut self,
        tool_box: Arc<ToolBox>,
        exchange_id: String,
        parent_exchange_id: String,
        tool_use_agent: ToolUseAgent,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<AgentToolUseOutput, SymbolError> {
        let mut converted_messages = vec![];
        let is_json_mode = tool_use_agent.is_json_mode_and_eval();
        for previous_message in self.exchanges.iter() {
            let converted_message = previous_message.to_conversation_message(is_json_mode).await;
            if let Some(converted_message) = converted_message {
                converted_messages.push(converted_message);
            }
        }

        // grab the terminal output if anything is present and pass it as part of the
        // agent input
        let pending_spawned_process_output = tool_box
            .grab_pending_subprocess_output(message_properties.clone())
            .await?;

        // now we can invoke the tool use agent over here and get the parsed input and store it
        // figure out how to pass in the is_json_mode over here, or we can implicitly decide
        // inside but the flows are a bit different altho there are code paths which should make
        // this work
        let output = if is_json_mode {
            let input = ToolUseAgentInputOnlyTools::new(
                converted_messages,
                // get the json schema for the tools over here
                // we are testing out the new thinking tool over here
                self.tools
                    .to_vec()
                    .into_iter()
                    .filter_map(|tool_type| tool_box.tools().get_tool_json(&tool_type))
                    .collect(),
                message_properties.clone(),
            );
            tool_use_agent.invoke_json_tool_swe_bench(input).await
        } else {
            // Now we can create the input for the tool use agent
            let tool_use_agent_input = ToolUseAgentInput::new(
                converted_messages,
                self.tools
                    .to_vec()
                    .into_iter()
                    .filter_map(|tool_type| tool_box.tools().get_tool_description(&tool_type))
                    .collect(),
                self.tools
                    .to_vec()
                    .into_iter()
                    .filter_map(|tool_type| tool_box.tools().get_tool_reminder(&tool_type))
                    .collect(),
                pending_spawned_process_output,
                message_properties.clone(),
            );
            tool_use_agent.invoke(tool_use_agent_input).await
        };
        let usage_stats = output.as_ref().map(|output| output.usage_statistics()).ok();

        // we match on the output type
        match output.map(|output| output.output_type()) {
            Ok(ToolUseAgentOutputType::Success(tool_use_success)) => {
                let tool_input_partial = tool_use_success.tool_parameters();
                let tool_thinking = tool_use_success.thinking();
                let tool_use_id = tool_use_success.tool_use_id();
                // send over a UI event over here to inform the editor layer that we found a tool to use
                let _ = message_properties
                    .ui_sender()
                    .send(UIEventWithID::tool_use_detected(
                        message_properties.root_request_id().to_owned(),
                        message_properties.request_id_str().to_owned(),
                        tool_input_partial.clone(),
                        tool_thinking.to_owned(),
                    ));
                let tool_type = tool_input_partial.to_tool_type();

                // add the action node for it
                let mut action_node = ActionNode::default_with_index(self.exchanges());
                action_node = action_node.set_action_tools(tool_input_partial.clone());
                action_node.set_llm_usage_statistics_maybe(usage_stats);
                self.action_nodes.push(action_node);

                // add to our exchange over here
                self.exchanges.push(Exchange::agent_tool_use(
                    parent_exchange_id,
                    exchange_id.to_owned(),
                    tool_input_partial.clone(),
                    tool_type,
                    tool_thinking.to_owned(),
                    tool_use_id.to_owned(),
                ));
                Ok(AgentToolUseOutput::Success((
                    tool_input_partial.clone(),
                    tool_use_id.to_owned(),
                    self,
                )))
            }
            Ok(ToolUseAgentOutputType::Failure(input_string)) => {
                // add action node to it
                let mut action_node = ActionNode::default_with_index(self.exchanges());
                action_node = action_node
                    .set_action_error(input_string.to_owned())
                    .error_observation(input_string.to_owned());
                action_node.set_llm_usage_statistics_maybe(usage_stats);
                self.action_nodes.push(action_node);

                // return the value
                Ok(AgentToolUseOutput::Failed(input_string))
            }
            Err(e) => {
                // add action node over here
                let mut action_node = ActionNode::default_with_index(self.exchanges());
                action_node = action_node
                    .set_action_error(e.to_string())
                    .error_observation(e.to_string());
                action_node.set_llm_usage_statistics_maybe(usage_stats);
                self.action_nodes.push(action_node);

                Ok(AgentToolUseOutput::Errored(e))
            }
        }
    }

    /// This reacts to the last message and generates the reply for the user to handle
    ///
    /// we should have a way to sync this up with a queue based system so we react to events
    /// one after another
    pub async fn reply_to_last_exchange(
        self,
        exchange_reply: AideAgentMode,
        tool_box: Arc<ToolBox>,
        parent_exchange_id: String,
        aide_rules: Option<String>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Self, SymbolError> {
        let last_exchange = self.last_exchange();
        if last_exchange.is_none() {
            return Ok(self);
        }

        // plan and edit todos are intentional. Should never be hit, but double check @skcd
        match exchange_reply {
            AideAgentMode::Chat => {
                self.chat_reply(tool_box, parent_exchange_id, aide_rules, message_properties)
                    .await
            }
            AideAgentMode::Plan => {
                todo!("plan branch")
            }
            AideAgentMode::Edit => {
                todo!("edit branch not supported")
            }
        }
    }

    /// This sends back a reply to the user message, using the context we have from before
    async fn chat_reply(
        self,
        tool_box: Arc<ToolBox>,
        parent_exchange_id: String,
        aide_rules: Option<String>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Self, SymbolError> {
        println!("session::chat_reply");
        // over here we want to convert all the previous exchanges to a context prompt
        // and then generate the appropriate things required
        let last_exchange = self.last_exchange();
        if last_exchange.is_none() {
            return Ok(self);
        }
        let last_exchange = last_exchange.expect("is_none to hold").clone();

        // Now that we have a new response exchange id we want to start streaming the reply back
        // to the user
        let last_exchange_type = last_exchange.exchange_type;
        match last_exchange_type {
            ExchangeType::HumanChat(_) => {
                self.human_chat_message_reply(
                    tool_box,
                    parent_exchange_id,
                    aide_rules,
                    message_properties,
                )
                .await
            }
            ExchangeType::AgentChat(_agent_message) => {
                todo!("figure out what to do over here")
            }
            ExchangeType::Edit(_edit) => {
                todo!("figure out what to do over here")
            }
            ExchangeType::Plan(_plan) => {
                todo!("figure out what to do over here")
            }
            ExchangeType::ToolOutput(_tool_output) => {
                todo!("figure out what to do over here")
            }
        }
    }

    /// Create the stream which will reply to the user
    async fn human_chat_message_reply(
        mut self,
        tool_box: Arc<ToolBox>,
        parent_exchange_id: String,
        aide_rules: Option<String>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Session, SymbolError> {
        println!("session::human_chat_message_reply");
        // take everything until the exchange id of the message we are supposed to
        // reply to
        let mut converted_messages = vec![];
        for previous_message in self.exchanges.iter() {
            let converted_message = previous_message.to_conversation_message(false).await;
            if let Some(converted_message) = converted_message {
                converted_messages.push(converted_message);
            }
        }
        let exchange_id = message_properties.request_id_str().to_owned();
        let llm_properties = message_properties.llm_properties().clone();

        let tool_input = SessionChatClientRequest::new(
            tool_box
                .recently_edited_files(Default::default(), message_properties.clone())
                .await?,
            self.global_running_user_context.clone(),
            converted_messages,
            aide_rules,
            self.repo_ref.clone(),
            self.project_labels.to_vec(),
            self.session_id.to_owned(),
            exchange_id.to_owned(),
            message_properties.ui_sender(),
            message_properties.cancellation_token(),
            llm_properties,
        );
        let chat_output = tool_box
            .tools()
            .invoke(ToolInput::ContextDrivenChatReply(tool_input))
            .await
            .map_err(|e| SymbolError::ToolError(e))?
            .context_drive_chat_reply()
            .ok_or(SymbolError::WrongToolOutput)?
            .reply();
        self.exchanges.push(Exchange::agent_chat_reply(
            parent_exchange_id,
            exchange_id.to_owned(),
            chat_output,
        ));
        let ui_sender = message_properties.ui_sender();
        // finsihed the exchange here since we have replied already
        let _ = ui_sender.send(UIEventWithID::finished_exchange(
            self.session_id.to_owned(),
            exchange_id,
        ));
        Ok(self)
    }

    /// We want to make sure that any open exchanges are accepted as we make
    /// progress towards are current exchange
    pub fn accept_open_exchanges_if_any(
        mut self,
        message_properties: SymbolEventMessageProperties,
    ) -> Self {
        let exchanges_to_close = self
            .exchanges
            .iter()
            .filter_map(|exchange| {
                if exchange.is_open() {
                    Some(exchange.exchange_id.to_owned())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        exchanges_to_close.into_iter().for_each(|exchange_id| {
            // mark the exchange as accepted
            let _ = message_properties
                .ui_sender()
                .send(UIEventWithID::edits_accepted(
                    self.session_id.to_owned(),
                    exchange_id.to_owned(),
                ));
            // mark the exchange as closed
            let _ = message_properties
                .ui_sender()
                .send(UIEventWithID::finished_exchange(
                    self.session_id.to_owned(),
                    exchange_id,
                ));
        });

        // now update all our exchanges to accepted
        self.exchanges = self
            .exchanges
            .into_iter()
            .map(|exchange| {
                if exchange.is_open() {
                    exchange.set_completion_status(true)
                } else {
                    exchange
                }
            })
            .collect();

        self
    }

    /// We have to map the plan revert exchange-id over here to be similar to
    /// the previous plan exchange-id, doing this will allow us to make sure
    /// that we are able to keep track of the edits properly
    pub async fn perform_plan_revert(
        self,
        plan_service: PlanService,
        previous_plan_exchange_id: String,
        step_index: usize,
        tool_box: Arc<ToolBox>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Self, SymbolError> {
        let original_plan = plan_service
            .load_plan_from_id(
                &plan_service.generate_unique_plan_id(&self.session_id, &previous_plan_exchange_id),
            )
            .await
            .map_err(|e| SymbolError::IOError(e))?;

        let exchange_id = message_properties.request_id_str().to_owned();

        if step_index == 0 {
            let ui_sender = message_properties.ui_sender();
            // revert the changes back by talking to the editor
            let _ = tool_box
                .undo_changes_made_during_session(
                    self.session_id.to_owned(),
                    previous_plan_exchange_id.to_owned(),
                    Some(step_index),
                    message_properties.clone(),
                )
                .await;

            let reply =
                "I have reverted the full plan, let me know how I can be of help?".to_owned();
            let _ = ui_sender.send(UIEventWithID::chat_event(
                self.session_id.to_owned(),
                exchange_id.to_owned(),
                reply.to_owned(),
                Some(reply.to_owned()),
            ));

            // update our exchanges and add what we did
            // TODO(skcd): Not sure what to do over here
            // self.exchanges.push(Exchange::agent_reply(
            //     message_properties.request_id_str().to_owned(),
            //     reply,
            //     AideAgentMode::Plan,
            // ));

            // now close the exchange
            let _ = ui_sender.send(UIEventWithID::finished_exchange(
                self.session_id.to_owned(),
                exchange_id,
            ));
        } else {
            let updated_plan = original_plan.drop_plan_steps(step_index);
            let ui_sender = message_properties.ui_sender();
            // send all the updated plan steps to the exchange
            updated_plan
                .steps()
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| *idx < step_index)
                .for_each(|(idx, plan_step)| {
                    let _ = ui_sender.send(UIEventWithID::plan_complete_added(
                        self.session_id.to_owned(),
                        previous_plan_exchange_id.to_owned(),
                        idx,
                        plan_step.files_to_edit().to_vec(),
                        plan_step.title().to_owned(),
                        plan_step.description().to_owned(),
                    ));
                });

            // revert the changes back by talking to the editor
            let _ = tool_box
                .undo_changes_made_during_session(
                    self.session_id.to_owned(),
                    previous_plan_exchange_id.to_owned(),
                    Some(step_index),
                    message_properties.clone(),
                )
                .await;

            // now send a message to the exchange telling that we have reverted
            // the changes
            let _ = ui_sender.send(UIEventWithID::chat_event(
                self.session_id.to_owned(),
                exchange_id.to_owned(),
                "I have reverted the changes made by the plan".to_owned(),
                Some("I have reverted the changes made by the plan".to_owned()),
            ));

            // update our exchanges and add what we did
            // TODO(skcd): Not sure what to do over here
            // self.exchanges.push(Exchange::agent_reply(
            //     message_properties.request_id_str().to_owned(),
            //     "I have reverted the changes made by the plan".to_owned(),
            //     AideAgentMode::Plan,
            // ));

            // now close the exchange
            let _ = ui_sender.send(UIEventWithID::finished_exchange(
                self.session_id.to_owned(),
                exchange_id,
            ));
        }

        Ok(self)
    }

    pub async fn perform_plan_generation(
        mut self,
        plan_service: PlanService,
        plan_id: String,
        parent_exchange_id: String,
        exchange_in_focus: Option<Exchange>,
        aide_rules: Option<String>,
        plan_storage_path: String,
        tool_box: Arc<ToolBox>,
        symbol_manager: Arc<SymbolManager>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Self, SymbolError> {
        // one of the bugs here is that the last exchange is now of the agent
        // replying to the user, so the exchange type is different completely
        // so can we pass it top down instead of getting the exchange here implicitly
        let Some(Exchange {
            exchange_id: _,
            exchange_type:
                ExchangeType::Plan(ExchangeTypePlan {
                    // when doing plan generation we are looking at the previous
                    // queries
                    previous_queries,
                    query,
                    user_context: _,
                }),
            exchange_state: _,
            is_compressed: _,
            is_hidden: _,
        }) = exchange_in_focus
        else {
            return Ok(self);
        };
        // take everything until the exchange id of the message we are supposed to
        // reply to
        let mut converted_messages = vec![];
        for previous_message in self.exchanges.iter() {
            let converted_message = previous_message.to_conversation_message(false).await;
            if let Some(converted_message) = converted_message {
                converted_messages.push(converted_message);
            }
        }
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut stream_receiver = tokio_stream::wrappers::UnboundedReceiverStream::new(receiver);

        // we should set our exchange over here
        self.exchanges.push(Exchange::agent_plan_reply(
            parent_exchange_id.to_owned(),
            message_properties.request_id_str().to_owned(),
            vec![],
        ));
        self.save_to_storage().await?;
        let mut agent_reply_exchange = self
            .exchanges
            .iter_mut()
            .find(|exchange| &exchange.exchange_id == message_properties.request_id_str());

        let exchange_id = message_properties.request_id_str().to_owned();
        let session_id = self.session_id.to_owned();

        // send a message over here that inference has started on the plan
        let _ = message_properties
            .ui_sender()
            .send(UIEventWithID::inference_started(
                message_properties.root_request_id().to_owned(),
                message_properties.request_id_str().to_owned(),
            ));

        // send a message over here that the plan is regenerating
        let _ = message_properties
            .ui_sender()
            .send(UIEventWithID::plan_regeneration(
                message_properties.root_request_id().to_owned(),
                message_properties.request_id_str().to_owned(),
            ));

        let ui_sender = message_properties.ui_sender();
        let _ = ui_sender.send(UIEventWithID::start_plan_generation(
            session_id.to_owned(),
            exchange_id.to_owned(),
        ));

        let cloned_message_properties = message_properties.clone();
        let cloned_plan_service = plan_service.clone();
        let global_running_context = self.global_running_user_context.clone();
        let cloned_aide_rules = aide_rules.clone();
        let plan = tokio::spawn(async move {
            cloned_plan_service
                .create_plan(
                    plan_id,
                    query.to_owned(),
                    previous_queries.to_vec(),
                    // always send the global running context over here
                    global_running_context,
                    cloned_aide_rules,
                    converted_messages,
                    false,
                    plan_storage_path,
                    Some(sender),
                    cloned_message_properties.clone(),
                )
                .await
        });

        // Create a channel for edits
        let (edits_sender, mut edits_receiver) = tokio::sync::mpsc::channel::<Option<Step>>(1);

        // Clone necessary variables for the edit task
        let symbol_manager_clone = symbol_manager.clone();
        let tool_box_clone = tool_box.clone();
        let message_properties_clone = message_properties.clone();

        // uncomment to test terminal command
        // let res = tool_box_clone
        //     .use_terminal_command("ls", message_properties_clone.clone())
        //     .await;
        // println!(
        //     "session::perform_plan_generation::terminal_command::res({:?})",
        //     res
        // );

        // Spawn the edit task
        let cloned_aide_rules = aide_rules.clone();
        let edit_task = tokio::spawn(async move {
            let mut steps_up_until_now = 0;
            let aide_rules = cloned_aide_rules;
            while let Some(step) = edits_receiver.recv().await {
                let previous_steps_up_until_now = steps_up_until_now;
                steps_up_until_now += 1;
                if step.is_none() {
                    break;
                }
                let step = step.expect("is_none to hold");
                println!("session::perform_plan_generation::new_step_found");
                let step_title = step.title.to_owned();
                let step_description = step.description();
                let instruction = format!(
                    r#"{step_title}
{step_description}"#
                );
                if let Some(file_to_edit) = step.file_to_edit() {
                    let file_open_response = tool_box_clone
                        .file_open(file_to_edit.to_owned(), message_properties_clone.clone())
                        .await?;
                    let hub_sender = symbol_manager_clone.hub_sender();
                    let (edit_done_sender, edit_done_receiver) = tokio::sync::oneshot::channel();
                    let _ = hub_sender.send(SymbolEventMessage::new(
                        SymbolEventRequest::simple_edit_request(
                            SymbolIdentifier::with_file_path(&file_to_edit, &file_to_edit),
                            SymbolToEdit::new(
                                file_to_edit.to_owned(),
                                file_open_response.full_range(),
                                file_to_edit.to_owned(),
                                vec![instruction.to_owned()],
                                false,
                                false,
                                true,
                                instruction.to_owned(),
                                None,
                                false,
                                None,
                                true,
                                None,
                                vec![],
                                Some(previous_steps_up_until_now.to_string()),
                            )
                            .set_aide_rules(aide_rules.clone()),
                            ToolProperties::new(),
                        ),
                        message_properties_clone.request_id().clone(),
                        message_properties_clone.ui_sender().clone(),
                        edit_done_sender,
                        message_properties_clone.cancellation_token(),
                        message_properties_clone.editor_url(),
                        message_properties_clone.llm_properties().clone(),
                    ));
                    println!("session::perform_plan_generation::edit_event::hub_sender::send");
                    let _ = edit_done_receiver.await;
                    println!("session::perform_plan_generation::edits_done::hub_sender::happy");
                }
            }
            Ok::<(), SymbolError>(())
        });

        let mut generated_steps = vec![];

        while let Some(step_message) = stream_receiver.next().await {
            match step_message {
                StepSenderEvent::NewStep(step) => {
                    {
                        if let Some(ref mut agent_reply_exchange) = agent_reply_exchange {
                            match &mut agent_reply_exchange.exchange_type {
                                ExchangeType::AgentChat(ref mut agent_chat) => {
                                    match &mut agent_chat.reply {
                                        ExchangeReplyAgent::Plan(ref mut plan_reply) => {
                                            plan_reply.plan_steps.push(step.clone());
                                        }
                                        _ => {}
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    generated_steps.push(step.clone());
                    let _ = edits_sender.send(Some(step)).await;
                }
                StepSenderEvent::NewStepTitle(title_found) => {
                    let _ = message_properties
                        .ui_sender()
                        .send(UIEventWithID::plan_title_added(
                            self.session_id.to_owned(),
                            exchange_id.clone(),
                            title_found.step_index(),
                            title_found.files_to_edit().to_vec(),
                            title_found.title().to_owned(),
                        ));
                }
                StepSenderEvent::NewStepDescription(description_update) => {
                    let _ = message_properties.ui_sender().send(
                        UIEventWithID::plan_description_updated(
                            self.session_id.to_owned(),
                            exchange_id.clone(),
                            description_update.index(),
                            description_update.delta(),
                            description_update.description_up_until_now().to_owned(),
                            description_update.files_to_edit().to_vec(),
                        ),
                    );
                }
                StepSenderEvent::DeveloperMessage(developer_message_delta) => {
                    let _ = message_properties
                        .ui_sender()
                        .send(UIEventWithID::chat_event(
                            self.session_id.to_owned(),
                            exchange_id.to_owned(),
                            "".to_owned(),
                            Some(developer_message_delta),
                        ));
                }
                StepSenderEvent::Done => {
                    let _ = edits_sender.send(None).await;
                    break;
                }
            }
        }

        // Close the edits sender and await the edit task
        // println!("session::perform_plan_generation::edits_sender::closed");
        // edits_sender.closed().await;

        println!("session::perform_plan_generation::edit_task::closed");
        let _ = edit_task.await;

        println!("session::perform_plan_generation::stream_receiver::closed");
        stream_receiver.close();

        // there is a race condition with cancel_running_exchange's invocation of
        // set_exchange_as_cancelled, which also saves to storage.
        self.save_to_storage().await?;

        match plan.await {
            // only return actively when we have an llm client error
            // this is because we are throwing a 401 on the llm client when we have an unauthroized request
            // which we should catch and bubble up
            Ok(Err(e)) => match e {
                PlanServiceError::SymbolError(SymbolError::ToolError(tool_error))
                | PlanServiceError::ToolError(tool_error) => match tool_error {
                    ToolError::LLMClientError(llm_client_error) => {
                        return Err(SymbolError::LLMClientError(llm_client_error));
                    }
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        }

        // send a message over here that the request is in review now
        // since we generated something for the plan
        if !message_properties.cancellation_token().is_cancelled() {
            println!("session::perform_plan_generation::cancellation_token::not_cancelled");
            let _ = message_properties
                .ui_sender()
                .send(UIEventWithID::request_review(
                    message_properties.root_request_id().to_owned(),
                    message_properties.request_id_str().to_owned(),
                ));
            let _ = message_properties
                .ui_sender()
                .send(UIEventWithID::plan_as_finished(
                    message_properties.root_request_id().to_owned(),
                    message_properties.request_id_str().to_owned(),
                ));
        }
        Ok(self)
    }

    /// we are going to perform the agentic editing
    pub async fn perform_agentic_editing(
        self,
        scratch_pad_agent: ScratchPadAgent,
        root_directory: String,
        aide_rules: Option<String>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Self, SymbolError> {
        let last_exchange = self.last_exchange();
        if let Some(Exchange {
            exchange_id: _,
            exchange_type:
                ExchangeType::Edit(ExchangeTypeEdit {
                    information:
                        ExchangeEditInformation::Agentic(ExchangeEditInformationAgentic {
                            query,
                            codebase_search,
                        }),
                    user_context: _,
                    ..
                }),
            exchange_state: _,
            is_compressed: _,
            is_hidden: _,
        }) = last_exchange
        {
            let edits_performed = scratch_pad_agent
                .human_message_agentic(
                    HumanAgenticRequest::new(
                        query.to_owned(),
                        root_directory,
                        *codebase_search,
                        self.global_running_user_context.clone(),
                        aide_rules,
                        false,
                    ),
                    message_properties.clone(),
                )
                .await;
            match edits_performed {
                Ok(_) => println!("session::perform_agentic_editing::finished_editing"),
                Err(_) => println!("Failed to edit"),
            };

            println!("session::finished_agentic_editing_exchange");
        }
        Ok(self)
    }

    /// We perform the anchored edit over here
    pub async fn perform_anchored_edit(
        mut self,
        parent_exchange_id: String,
        scratch_pad_agent: ScratchPadAgent,
        aide_rules: Option<String>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Self, SymbolError> {
        let last_exchange = self.last_exchange().cloned();
        if let Some(Exchange {
            exchange_id: _,
            exchange_type:
                ExchangeType::Edit(ExchangeTypeEdit {
                    information:
                        ExchangeEditInformation::Anchored(ExchangeEditInformationAnchored {
                            query,
                            fs_file_path,
                            range,
                            selection_context: _,
                        }),
                    ..
                }),
            exchange_state: _,
            is_compressed: _,
            is_hidden: _,
        }) = last_exchange
        {
            let mut converted_messages = vec![];
            for previous_message in self.exchanges.iter() {
                let converted_message = previous_message.to_conversation_message(false).await;
                if let Some(converted_message) = converted_message {
                    converted_messages.push(converted_message);
                }
            } // send a message over that the inference will start in a bit
            let _ = message_properties
                .ui_sender()
                .send(UIEventWithID::inference_started(
                    message_properties.root_request_id().to_owned(),
                    message_properties.request_id_str().to_owned(),
                ));

            // we want to set our state over here that we have started working on it
            // We want to get the changes which have been performed here for the anchored
            // edit especially on the location we are interested in and not anywhere else
            // self.exchanges.push(Exchange::agent_reply(
            //     message_properties.request_id_str().to_owned(),
            //     "thinking".to_owned(),
            //     AideAgentMode::Edit,
            // ));
            // send a message that we are starting with the edits over here
            // we want to make a note of the exchange that we are working on it
            // once we have the eidts, then we also have to make sure that we track
            // that we are working on some exchange
            let _ = message_properties
                .ui_sender()
                .send(UIEventWithID::edits_started_in_exchange(
                    message_properties.root_request_id().to_owned(),
                    message_properties.request_id_str().to_owned(),
                    vec![fs_file_path.to_owned()],
                ));
            let edits_performed = scratch_pad_agent
                .anchor_editing_on_range(
                    range.clone(),
                    fs_file_path.to_owned(),
                    query.to_owned(),
                    converted_messages,
                    self.global_running_user_context
                        .clone()
                        .to_xml(Default::default())
                        .await
                        .map_err(|e| SymbolError::UserContextError(e))?,
                    aide_rules,
                    message_properties.clone(),
                )
                .await;

            match edits_performed {
                Ok(edits_performed) => {
                    self.exchanges.push(Exchange::agent_edits_reply(
                        parent_exchange_id,
                        message_properties.request_id_str().to_owned(),
                        edits_performed,
                    ));
                }
                Err(_) => self.exchanges.push(Exchange::agent_edits_reply(
                    parent_exchange_id,
                    message_properties.request_id_str().to_owned(),
                    "Failed to edit selection properly".to_owned(),
                )),
            }

            let _ = message_properties
                .ui_sender()
                .send(UIEventWithID::edits_marked_complete(
                    message_properties.root_request_id().to_owned(),
                    message_properties.request_id_str().to_owned(),
                ));

            // send a message over here static that we can ask the user for review
            let _ = message_properties
                .ui_sender()
                .send(UIEventWithID::request_review(
                    message_properties.root_request_id().to_owned(),
                    message_properties.request_id_str().to_owned(),
                ));
        }
        Ok(self)
    }

    /// Grab the references over here and
    pub async fn hot_streak_message(
        mut self,
        exchange_id: &str,
        tool_box: Arc<ToolBox>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<(), SymbolError> {
        let exchange_by_id = self.get_exchange_by_id(exchange_id);
        // if its a plan we want to look at the files which were part of the executed
        // steps
        let files_to_edit_if_plan = match exchange_by_id {
            Some(Exchange {
                exchange_id: _,
                exchange_type:
                    ExchangeType::AgentChat(ExchangeTypeAgent {
                        reply:
                            ExchangeReplyAgent::Plan(ExchangeReplyAgentPlan {
                                plan_steps,
                                plan_discarded: _,
                            }),
                        parent_exchange_id: _,
                    }),
                exchange_state: _,
                is_compressed: _,
                is_hidden: _,
            }) => {
                // do something over here
                let files_to_edit = plan_steps
                    .into_iter()
                    .filter_map(|plan_step| plan_step.file_to_edit())
                    .collect::<Vec<_>>();
                files_to_edit
            }
            _ => vec![],
        };
        // if its an anchored edit then we want to look at the parent of the
        // exchange_id to which we are creating a hot streak for (since our own
        // exchange has no data about the file edited)
        let parent_exchange = self.get_parent_exchange_id(exchange_id);
        if let None = parent_exchange {
            return Ok(());
        }
        let parent_exchange = parent_exchange.expect("if let None to hold");
        let parent_exchange_id = parent_exchange.exchange_id.to_owned();
        let files_to_check_if_edit = match parent_exchange {
            Exchange {
                exchange_id: _,
                exchange_type:
                    ExchangeType::Edit(ExchangeTypeEdit {
                        information:
                            ExchangeEditInformation::Anchored(ExchangeEditInformationAnchored {
                                query: _,
                                fs_file_path,
                                range: _,
                                selection_context: _,
                            }),
                        ..
                    }),
                exchange_state: _,
                is_compressed: _,
                is_hidden: _,
            } => {
                vec![fs_file_path.to_owned()]
            }
            _ => vec![],
        };
        let final_files = files_to_edit_if_plan
            .into_iter()
            .chain(files_to_check_if_edit)
            .collect::<Vec<_>>();

        let mut converted_messages = vec![];
        for previous_message in self.exchanges.iter() {
            let converted_message = previous_message.to_conversation_message(false).await;
            if let Some(converted_message) = converted_message {
                converted_messages.push(converted_message);
            }
        }
        let (diagnostics, mut extra_variables) = tool_box
            .grab_workspace_diagnostics(message_properties.clone())
            .await?;
        // get the diagnostics over here properly
        let diagnostics_grouped_by_file: DiagnosticMap =
            diagnostics
                .into_iter()
                .fold(HashMap::new(), |mut acc, error| {
                    acc.entry(error.fs_file_path().to_owned())
                        .or_insert_with(Vec::new)
                        .push(error);
                    acc
                });

        let mut user_context = UserContext::new(vec![], vec![], None, vec![]);
        user_context = user_context.add_variables(extra_variables.to_vec());

        for (fs_file_path, lsp_diagnostics) in diagnostics_grouped_by_file.iter() {
            let extra_variables_type_definitions = tool_box
                .grab_type_definition_worthy_positions_using_diagnostics(
                    fs_file_path,
                    lsp_diagnostics.to_vec(),
                    message_properties.clone(),
                )
                .await;
            if let Ok(extra_variables_type_definitions) = extra_variables_type_definitions {
                extra_variables.extend(extra_variables_type_definitions.to_vec());
                user_context = user_context.add_variables(extra_variables_type_definitions);
            }
        }

        // add the diagnostics context over here
        self.global_running_user_context = self
            .global_running_user_context
            .merge_user_context(user_context);

        // now get the diagnostics which are part of the references over here
        let user_query = if final_files.is_empty() {
            // return early if we have no files.. bizzare but okay
            "Reflect on the recent changes you made and if there is anything you can improve"
                .to_owned()
        } else if extra_variables.is_empty() {
            // Over here we should ask the agent to reflect on its work and suggest
            // better changes over here or gotchas for the user to keep in mind
            "Reflect on the recent changes you made and if there is anything you can improve"
                .to_owned()
        } else {
            PlanService::format_diagnostics(&diagnostics_grouped_by_file)
        };

        // now send a message first listing out the files we are going to look at
        let message = "Looking at Language Server errors ...\n".to_owned();
        let _ = message_properties
            .ui_sender()
            .send(UIEventWithID::chat_event(
                self.session_id().to_owned(),
                message_properties.request_id_str().to_owned(),
                "".to_owned(),
                Some(message),
            ));
        // send all the gathered references over here
        let _ = message_properties
            .ui_sender()
            .send(UIEventWithID::send_variables(
                self.session_id().to_owned(),
                message_properties.request_id_str().to_owned(),
                extra_variables,
            ));

        // we can use the tool_box to send a message over here
        let response = tool_box
            .tools()
            .invoke(ToolInput::ContextDriveHotStreakReply(
                SessionHotStreakRequest::new(
                    tool_box
                        .recently_edited_files(Default::default(), message_properties.clone())
                        .await?,
                    self.global_running_user_context.clone(),
                    converted_messages,
                    user_query,
                    self.repo_ref.clone(),
                    self.project_labels.clone(),
                    self.session_id.to_owned(),
                    message_properties.request_id_str().to_owned(),
                    message_properties.ui_sender().clone(),
                    message_properties.cancellation_token().clone(),
                    message_properties.llm_properties().clone(),
                ),
            ))
            .await
            .map_err(|e| SymbolError::ToolError(e))?
            .get_context_driven_hot_streak_reply()
            .ok_or(SymbolError::WrongToolOutput)?
            .reply()
            .to_owned();
        self.exchanges.push(Exchange::agent_chat_reply(
            parent_exchange_id,
            message_properties.request_id_str().to_owned(),
            response,
        ));
        self.save_to_storage().await?;
        // finsihed the exchange here since we have replied already
        let _ = message_properties
            .ui_sender()
            .send(UIEventWithID::finished_exchange(
                self.session_id.to_owned(),
                message_properties.request_id_str().to_owned(),
            ));
        Ok(())
    }

    pub fn truncate_hidden_exchanges(&mut self) {
        println!(
            "session::truncate_hidden_exchanges::before({})",
            self.exchanges.len()
        );
        self.exchanges.retain(|exchange| !exchange.is_hidden);
        println!(
            "session::truncate_hidden_exchanges::after({})",
            self.exchanges.len()
        );
    }

    pub fn has_running_code_edits(&self, exchange_id: &str) -> bool {
        let found_exchange = self.find_exchange_by_id(exchange_id);
        match found_exchange {
            Some(exchange) => {
                exchange.is_agent_work() && exchange.is_still_running() && exchange.has_code_edits()
            }
            None => false,
        }
    }

    pub fn set_exchange_as_cancelled(
        mut self,
        exchange_id: &str,
        message_properties: SymbolEventMessageProperties,
    ) -> Self {
        if self.has_running_code_edits(exchange_id) {
            let found_exchange = self.find_exchange_by_id_mut(exchange_id);
            if let Some(exchange) = found_exchange {
                exchange.set_exchange_as_cancelled();
                match &mut exchange.exchange_type {
                    ExchangeType::AgentChat(ref mut agent_chat) => match &mut agent_chat.reply {
                        ExchangeReplyAgent::Edit(_) => {
                            let _ = message_properties.ui_sender().send(
                                UIEventWithID::edits_cancelled_in_exchange(
                                    message_properties.root_request_id().to_owned(),
                                    message_properties.request_id_str().to_owned(),
                                ),
                            );
                        }
                        ExchangeReplyAgent::Plan(ref mut plan) => {
                            plan.plan_discarded = true;
                            let _ = message_properties.ui_sender().send(
                                UIEventWithID::plan_as_cancelled(
                                    message_properties.root_request_id().to_owned(),
                                    message_properties.request_id_str().to_owned(),
                                ),
                            );
                        }
                        _ => {}
                    },
                    _ => {}
                };
            }
        }
        self
    }

    pub async fn invoke_tool(
        mut self,
        tool_type: ToolType,
        tool_input_partial: ToolInputPartial,
        tool_use_id: String,
        tool_box: Arc<ToolBox>,
        root_directory: String,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Self, SymbolError> {
        let tool_use_time_taken = std::time::Instant::now();
        let exchange_id = message_properties.request_id_str().to_owned();
        match tool_input_partial {
            ToolInputPartial::TestRunner(test_runner) => {
                let editor_url = message_properties.editor_url().to_owned();
                let fs_file_paths = test_runner.fs_file_paths();
                let input =
                    ToolInput::RunTests(TestRunnerRequest::new(fs_file_paths.to_vec(), editor_url));
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?;

                let test_runner_output = response.get_test_runner().unwrap();

                // Truncate and format the test output
                let formatted_output = {
                    let raw_output = test_runner_output.test_output();
                    let test_files =
                        format!("Running tests in files:\n{}\n", fs_file_paths.join("\n"));

                    if raw_output.lines().count() > 50 {
                        // Take the first 0 lines and last 100 lines
                        let first_lines: Vec<_> = raw_output.lines().take(0).collect();
                        let last_lines: Vec<_> = raw_output.lines().rev().take(100).collect();

                        format!(
                            "{}\n{}\n\n[...test execution...]\n\n{}",
                            test_files,
                            first_lines.join("\n"),
                            last_lines.into_iter().rev().collect::<Vec<_>>().join("\n")
                        )
                    } else {
                        format!("{}\n{}", test_files, raw_output)
                    }
                };

                println!(
                    "session::invoke_tool::formatted_output: {:?}",
                    formatted_output
                );

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(formatted_output.to_owned());
                }

                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    formatted_output, // truncated
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::AskFollowupQuestions(followup_question) => {
                // this waits for the user-feedback so we do not need to react or
                // do anything after this

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(followup_question.question().to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                    action_node.mark_as_terminal();
                }
            }
            ToolInputPartial::AttemptCompletion(attempt_completion) => {
                println!("LLM reached a stop condition");
                println!("{:?}", &attempt_completion);
                // no need to send anything when we are attempting completion since we are done
                // over here with the tool use itself
                // figure out what to do over here

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(attempt_completion.to_string());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                    action_node.mark_as_terminal();
                }
            }
            ToolInputPartial::CodeEditing(code_editing) => {
                let mut fs_file_path = code_editing.fs_file_path().to_owned();
                if !std::path::Path::new(&fs_file_path).is_absolute() {
                    // we need to join the file path here with the root directory
                    println!("fixing relative file path");
                    fs_file_path = std::path::Path::new(&root_directory)
                        .join(fs_file_path)
                        .to_string_lossy()
                        .to_string();
                }
                println!("Code editing: {}", fs_file_path);
                let file_contents = tool_box
                    .file_open(fs_file_path.to_owned(), message_properties.clone())
                    .await?
                    .contents();

                let instruction = code_editing.instruction().to_owned();

                // keep track of the file content which we are about to modify over here
                let old_file_content = tool_box
                    .file_open(fs_file_path.to_owned(), message_properties.clone())
                    .await;

                // if the file is very very large then we chunk it up and use search and replace
                // on individual chunks instead
                let updated_code = {
                    let default_range =
                    // very large end position
                    Range::new(Position::new(0, 0, 0), Position::new(10_000, 0, 0));

                    let symbol_to_edit = SymbolToEdit::new(
                        fs_file_path.to_owned(),
                        default_range,
                        fs_file_path.to_owned(),
                        vec![instruction.clone()],
                        false,
                        false, // is_new
                        false,
                        "".to_owned(),
                        None,
                        false,
                        None,
                        false,
                        None,
                        vec![], // previous_user_queries
                        None,
                    );

                    let symbol_identifier = SymbolIdentifier::new_symbol(&fs_file_path);

                    tool_box
                        .code_editing_with_search_and_replace(
                            &symbol_to_edit,
                            &fs_file_path,
                            &file_contents,
                            &default_range,
                            "".to_owned(),
                            instruction.clone(),
                            &symbol_identifier,
                            None,
                            None,
                            message_properties.clone(),
                        )
                        .await // big expectations but can also fail, we should handle it properly
                }?;

                // now that we have modified the file we can ask the editor for the git-diff of this file over here
                // and we also have the previous state over here
                let diff_changes = tool_box
                    .recently_edited_files_with_content(
                        vec![fs_file_path.to_owned()].into_iter().collect(),
                        match old_file_content {
                            Ok(old_file_content) => {
                                vec![DiffFileContent::new(
                                    fs_file_path.to_owned(),
                                    old_file_content.contents(),
                                    Some(updated_code),
                                )]
                            }
                            Err(_) => vec![],
                        },
                        message_properties.clone(),
                    )
                    .await?;

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(format!(
                        r#"I performed the edits which you asked for, here is the git diff for it:
{}"#,
                        diff_changes.l1_changes()
                    ));
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                // we need to take the L1 level changes here since those are the ones we are interested in and then add
                // that as a human message over here
                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    format!(
                        r#"I performed the edits which you asked for, here is the git diff for it:
{}"#,
                        diff_changes.l1_changes()
                    ),
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::LSPDiagnostics(diagnostics) => {
                println!("LSP diagnostics: {:?}", diagnostics);
                // figure out what do to with this, we should probably just gather all the diagnostics
                // and pass it along as a user message
                let diagnostics_output = dbg!(
                    tool_box
                        .grab_workspace_diagnostics(message_properties.clone())
                        .await
                )
                .expect("big expectation for diagnostics to never fail");
                let diagnostics_grouped_by_file: DiagnosticMap = diagnostics_output
                    .0
                    .into_iter()
                    .fold(HashMap::new(), |mut acc, error| {
                        acc.entry(error.fs_file_path().to_owned())
                            .or_insert_with(Vec::new)
                            .push(error);
                        acc
                    });

                let formatted_diagnostics =
                    PlanService::format_diagnostics(&diagnostics_grouped_by_file);

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(formatted_diagnostics.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }
                // send an update over here
                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            formatted_diagnostics.to_owned(),
                        ));
                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    formatted_diagnostics,
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::FindFile(find_files) => {
                println!("find files: {}", find_files.pattern());
                let find_files_input = FindFilesRequest::new(
                    find_files.pattern().to_owned(),
                    root_directory.to_owned(),
                );
                let input = ToolInput::FindFiles(find_files_input);
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?;
                let find_files_output = response
                    .get_find_files_response()
                    .ok_or(SymbolError::WrongToolOutput)?;
                let mut response = find_files_output
                    .files()
                    .into_iter()
                    .map(|file_path| file_path.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                if response.trim().is_empty() {
                    response = "0 results found".to_owned();
                }
                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            response.to_owned(),
                        ));
                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(response.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    response,
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::ListFiles(list_files) => {
                println!("list files: {}", list_files.directory_path());
                let list_files_input = ListFilesInput::new(
                    list_files.directory_path().to_owned(),
                    list_files.recursive(),
                    message_properties.editor_url(),
                );
                let input = ToolInput::ListFiles(list_files_input);
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?;
                let list_files_output = response
                    .get_list_files_directory()
                    .ok_or(SymbolError::WrongToolOutput)?;
                let mut response = list_files_output
                    .files()
                    .into_iter()
                    .map(|file_path| file_path.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                // add a response that we did not find any results
                if response.trim().is_empty() {
                    response = "0 results found".to_owned();
                }
                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            response.to_owned(),
                        ));

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(response.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    response,
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::OpenFile(open_file) => {
                let open_file_path = open_file.fs_file_path().to_owned();
                let request = OpenFileRequest::new(
                    open_file_path,
                    message_properties.editor_url(),
                    open_file.start_line(),
                    open_file.end_line(),
                );
                let input = ToolInput::OpenFile(request);
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?
                    .get_file_open_response()
                    .ok_or(SymbolError::WrongToolOutput)?
                    .to_string();
                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            response.to_owned(),
                        ));

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(response.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    response,
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::SemanticSearch(semantic_search) => {
                // now we start executing the semantic search based on the query we are getting
                // over here
                let input = ToolInput::SemanticSearch(SemanticSearchRequest::new(
                    root_directory,
                    message_properties
                        .llm_properties()
                        .clone()
                        .set_llm(LLMType::DeepSeekCoderV3),
                    semantic_search.search_query().to_owned(),
                    message_properties.clone(),
                ));

                let mut semantic_search_response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?
                    .get_semantic_search_response()
                    .ok_or(SymbolError::WrongToolOutput)?
                    .file_decisions();

                // now we first filter out the responses here which are negative and then sort them
                // by score and then finally
                semantic_search_response = semantic_search_response
                    .into_iter()
                    .filter(|(_, filter_result)| filter_result.is_relevant())
                    .collect::<Vec<_>>();

                // sort by their scores in ascending order
                semantic_search_response.sort_by(|a, b| {
                    a.1.score()
                        .partial_cmp(&b.1.score())
                        .unwrap_or(std::cmp::Ordering::Greater)
                });
                // reverse it so its in decreasing order
                semantic_search_response.reverse();
                // keep only the top 20 files
                semantic_search_response.truncate(20);
                // now create the response for to show to the user
                let semantic_search_response = semantic_search_response
                    .into_iter()
                    .map(|(fs_file_path, semantic_search_result)| {
                        format!(
                            r#"## file_path: {}
reason: {}"#,
                            fs_file_path,
                            semantic_search_result.reason()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(semantic_search_response.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            semantic_search_response.to_owned(),
                        ));
                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    semantic_search_response.to_owned(),
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::SearchFileContentWithRegex(search_file) => {
                println!("search file: {}", search_file.directory_path());
                let request = SearchFileContentInput::new(
                    search_file.directory_path().to_owned(),
                    search_file.regex_pattern().to_owned(),
                    search_file.file_pattern().map(|s| s.to_owned()),
                    message_properties.editor_url(),
                );
                let input = ToolInput::SearchFileContentWithRegex(request);
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))
                    .map(|output| output.get_search_file_content_with_regex());

                // do not early fail when search file content with regex fails
                let response = match response {
                    Ok(Some(search_result_output)) => search_result_output.response().to_owned(),
                    Err(e) => e.to_string(),
                    Ok(None) => {
                        return Err(SymbolError::WrongToolOutput);
                    }
                };

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(response.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                // if the cancellation token is set, then we should not update
                // our state over here with the broken terminal output
                if message_properties.cancellation_token().is_cancelled() {
                    return Ok(self);
                }
                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            response.to_owned(),
                        ));
                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    response.to_owned(),
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::TerminalCommand(terminal_command) => {
                println!("terminal command: {}", terminal_command.command());
                let command = terminal_command.command().to_owned();
                let wait_for_exit = terminal_command.wait_for_exit().to_owned();
                let request =
                    TerminalInput::new(command, message_properties.editor_url(), wait_for_exit);
                let input = ToolInput::TerminalCommand(request);
                let tool_output = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?
                    .terminal_command()
                    .ok_or(SymbolError::WrongToolOutput)?;

                let output = tool_output.output().to_owned();
                let mut output_lines: Vec<String> =
                    output.lines().map(|line| line.to_string()).collect();

                // only keep 1k lines, hand waving this into the world 🪄
                let max_lines = 1_000;
                if output_lines.len() > max_lines {
                    let start_index = output_lines.len() - max_lines;
                    output_lines = output_lines.split_off(start_index);
                    output_lines.insert(0, "[... previous output truncated ...]".to_owned());
                }

                // Process each line to add truncation indicators
                let max_chars = 500;
                let truncation_suffix = "... truncated";
                let suffix_len = truncation_suffix.chars().count();

                for line in &mut output_lines {
                    let char_count = line.chars().count();
                    if char_count > max_chars {
                        let take_chars = max_chars.saturating_sub(suffix_len);
                        let mut truncated = String::with_capacity(max_chars);

                        // Add truncated content
                        truncated.extend(line.chars().take(take_chars));

                        // Add suffix
                        truncated.push_str(&truncation_suffix);

                        // Replace original line
                        *line = truncated;
                    }
                }

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(output.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                // if the cancellation token is set, then we should not update
                // our state over here with the broken terminal output
                if message_properties.cancellation_token().is_cancelled() {
                    return Ok(self);
                }
                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            output.to_owned(),
                        ));
                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    output,
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::RepoMapGeneration(repo_map_request) => {
                println!(
                    "repo map generation request: {}",
                    repo_map_request.to_string()
                );
                let request = ToolInput::RepoMapGeneration(RepoMapGeneratorRequest::new(
                    repo_map_request.directory_path().to_owned(),
                    3000,
                ));
                let tool_output = tool_box
                    .tools()
                    .invoke(request)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?
                    .repo_map_generator_response()
                    .ok_or(SymbolError::WrongToolOutput)?;
                let repo_map_str = tool_output.repo_map().to_owned();

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(repo_map_str.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            repo_map_str.to_owned(),
                        ));
                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    repo_map_str.to_owned(),
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::ContextCrunching(_context_crunching) => {
                // we do not call this as a tool explicitly
            }
            ToolInputPartial::CodeEditorParameters(_code_editor_parameters) => {
                // we do not use this tool via the session.invoke_tool flow at all
            }
            ToolInputPartial::Reasoning(_) => {
                // we do not call this as a tool explicitly
            }
            ToolInputPartial::Thinking(_) => {
                // we don't do any tool calling here but take the input of the thought
                // and store it as part of our observation, not even sending the UI event
                // here since we are running stateless from now on

                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut("Your thought has been logged.".to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }
                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    "Your thought has been logged".to_owned(),
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
            ToolInputPartial::RequestScreenshot(_) => {
                println!("request_screenshot");
                let request_screenshot_input =
                    RequestScreenshotInput::new(message_properties.editor_url());
                let input = ToolInput::RequestScreenshot(request_screenshot_input);
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?;
                let request_screenshot_output = response
                    .get_request_screenshot_response()
                    .ok_or(SymbolError::WrongToolOutput)?;
                let response = request_screenshot_output.to_llm_image();

                // Convert SessionChatMessageImage to ImageInformation
                let image_info = ImageInformation::new(
                    response.r#type().to_owned(),
                    response.media().to_owned(),
                    response.data().to_owned(),
                );

                // we have the tool output over here
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut("Screenshot captured successfully".to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    "Screenshot captured successfully".to_owned(),
                    UserContext::default().add_image(image_info),
                    exchange_id.to_owned(),
                );
            }
            ToolInputPartial::McpTool(partial) => {
                println!("mcp tool {}", partial.full_name);
                let input = ToolInput::McpTool(McpToolInput { partial });
                let tool_output = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| SymbolError::ToolError(e))?;
                let response = tool_output
                    .get_mcp_response()
                    .ok_or(SymbolError::WrongToolOutput)?;
                let data = &response.data;
                let _ =
                    message_properties
                        .ui_sender()
                        .send(UIEventWithID::tool_output_delta_response(
                            message_properties.root_request_id().to_owned(),
                            message_properties.request_id_str().to_owned(),
                            "".to_owned(),
                            data.to_owned(),
                        ));
                if let Some(action_node) = self.action_nodes.last_mut() {
                    action_node.add_observation_mut(data.to_owned());
                    action_node.set_time_taken_seconds(tool_use_time_taken.elapsed().as_secs_f32());
                }

                self = self.tool_output(
                    &exchange_id,
                    tool_type.clone(),
                    data.to_owned(),
                    UserContext::default(),
                    tool_use_id.to_owned(),
                );
            }
        }
        Ok(self)
    }

    /// Helper for atomic file operations
    pub async fn atomic_file_operation(
        storage_path: &str,
        content: String,
    ) -> Result<(), SymbolError> {
        // Create a temporary file path by appending .tmp to the original path
        let temp_path = format!("{}.tmp", storage_path);

        // Ensure the content is properly trimmed to avoid any trailing whitespace issues
        let trimmed_content = content.trim_end();

        // Write to temporary file first
        let mut temp_file = tokio::fs::File::create(&temp_path)
            .await
            .map_err(|e| SymbolError::IOError(e))?;

        temp_file
            .write_all(trimmed_content.as_bytes())
            .await
            .map_err(|e| SymbolError::IOError(e))?;

        // Ensure all data is written to disk
        temp_file
            .sync_all()
            .await
            .map_err(|e| SymbolError::IOError(e))?;

        // Close the file explicitly before renaming
        drop(temp_file);

        // Atomically rename temp file to target file
        tokio::fs::rename(&temp_path, storage_path)
            .await
            .map_err(|e| SymbolError::IOError(e))?;

        Ok(())
    }

    async fn save_to_storage(&self) -> Result<(), SymbolError> {
        let serialized = serde_json::to_string(self).unwrap();
        Self::atomic_file_operation(self.storage_path(), serialized).await
    }
}
