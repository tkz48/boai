use super::types::{FillInMiddleFormatter, FillInMiddleRequest};
use either::Either;
use llm_client::clients::types::{
    LLMClientCompletionRequest, LLMClientCompletionStringRequest, LLMClientMessage,
};

pub struct ClaudeFillInMiddleFormatter;

impl ClaudeFillInMiddleFormatter {
    pub fn new() -> Self {
        Self
    }

    pub fn few_shot_messages(&self) -> Vec<LLMClientMessage> {
        vec![
            LLMClientMessage::user(
                r#"<prefix>

fn main() {
    let mut numbers = vec![1, 2, 3, 4, 5];
    for number in &mut numbers {
        *number += 10;
    }
    println!("Updated numbers: {:?}", numbers);
    let sum = numbers.iter().sum::<i32>();
    println!("Sum of numbers: {}", sum);
</prefix>
<insertion_point>
    let average = <code_inserted></code_inserted>
</insertion_point>
<suffix>
    println!("Average of numbers: {}", average);
}
</suffix>"#
                    .to_owned(),
            ),
            LLMClientMessage::assistant(
                r#"<code_inserted>
sum as f64 / numbers.len() as f64;
</code_inserted>"#
                    .to_owned(),
            ),
            //             LLMClientMessage::user(
            //                 r#"<prompt>
            // class Car:
            //     def __init__(self, make, model, year):
            //         self.make = make
            //         self.model = model
            //         self.year = year

            //     def get_car_details(self):
            //         <<CURSOR>>

            //     def get_year_details(self):
            //         return self.year

            // my_car = Car("Toyota", "Camry", 2022)

            // print(my_car.get_car_details())
            // </prompt>"#
            //                     .to_owned(),
            //             ),
            //             LLMClientMessage::assistant(
            //                 r#"<reply>
            // return f"{self.year} {self.make} {self.model}"
            // </reply>"#
            //                     .to_owned(),
            //             ),
        ]
    }
}

impl FillInMiddleFormatter for ClaudeFillInMiddleFormatter {
    fn fill_in_middle(
        &self,
        request: FillInMiddleRequest,
    ) -> Either<LLMClientCompletionRequest, LLMClientCompletionStringRequest> {
        let system_prompt = r#"You are an intelligent code autocomplete model trained to generate code completions from the cursor position. Given a code snippet with a cursor position marked by <<CURSOR>>, your task is to generate the code that should appear at the <<CURSOR>> to complete the code logically.

To generate the code completion, follow these guidelines:
1. Analyze the code before and after the cursor position to understand the context and intent of the code.
2. If provided, utilize the relevant code snippets from other locations in the codebase to inform your completion. 
3. Generate code that logically continues from the cursor position, maintaining the existing code structure and style.
4. Avoid introducing extra whitespace unless necessary for the code completion.
5. Output only the completed code, without any additional explanations or comments.
6. The code you generate will be inserted at the <<CURSOR>> location, so be mindful to write code that logically follows from the <<CURSOR>> location.
7. You have to always start your reply with <code_inserted> as show in the interactions with the user.
8. You should stop generating code and end with </code_inserted> when you have logically completed the code block you are supposed to autocomplete.
9. Use the same indentation for the generated code as the position of the <<CURSOR>> location. Use spaces if spaces are used; use tabs if tabs are used.
        
Remember, your goal is to provide the most appropriate and efficient code completion based on the given context and the location of the cursor. Use your programming knowledge and the provided examples to generate high-quality code completions that meet the requirements of the task."#;
        let prefix = request.prefix();
        let suffix = request.suffix();
        let insertion_prefix = request.current_line_content();
        let prefix_lines = prefix
            .lines()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        let suffix_lines = suffix.lines().take(4).collect::<Vec<_>>().join("\n");
        let fim_request = format!(
            r#"<prompt>
<prefix>
{prefix}
</prefix>
<insertion_point>
{insertion_prefix}<<CURSOR>>
</insertion_point>
<suffix>
{suffix}
</suffix>
</prompt>

As a reminder the section in <prompt> where you have to make changes is over here
<reminder>
{prefix_lines}
<insertion_point>
{insertion_prefix}<<CURSOR>>
</insertion_point>
{suffix_lines}
</reminder>"#
        );
        let assistant_partial_answer = LLMClientMessage::assistant(
            format!(
                r#"<code_inserted>
{prefix_lines}
{insertion_prefix}"#
            )
            .to_owned(),
        );
        let mut final_messages = vec![
            LLMClientMessage::system(system_prompt.to_owned()),
            LLMClientMessage::user(fim_request),
        ];
        // if the insertion prefix is not all whitespace then we can add words to
        // the llm mouth
        if !insertion_prefix.trim().is_empty() {
            final_messages.push(assistant_partial_answer);
        }
        let mut llm_request =
            LLMClientCompletionRequest::new(request.llm().clone(), final_messages, 0.1, None);
        if let Some(max_tokens) = request.completion_tokens() {
            llm_request = llm_request.set_max_tokens(max_tokens);
        }
        either::Left(llm_request)
    }
}
