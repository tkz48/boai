# Install vllm here
pip install vllm

# Then install huggingface-cli
pip install -U "huggingface_hub[cli]"

# Login to huggingface
huggingface-cli login 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


# Next we need to download the deepseek-coder-6.7b model
huggingface-cli download deepseek-ai/deepseek-coder-6.7b-instruct


# Now we can run vllm endpoint using the following command
python -m vllm.entrypoints.openai.api_server --model deepseek-ai/deepseek-coder-6.7b-instruct  --port 42424 --served-model-name deepseek-coder-6b --max-model-len 3000