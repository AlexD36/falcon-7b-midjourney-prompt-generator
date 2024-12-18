# Fine-Tuned Falcon 7B for Generating MidJourney Prompts

![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Used%20for%20Fine--Tuning-orange?style=flat-square)

This repository contains a fine-tuned version of the open-source Falcon 7B large language model (LLM), designed to generate detailed and complex MidJourney prompts from simple user input sentences. The fine-tuning process was performed using a custom dataset prepared with Relevance AI and insights from ChatGPT.

## Overview

The fine-tuned model simplifies the process of creating high-quality prompts for MidJourney, an AI-based image generation tool. By inputting a brief description or idea, users can obtain detailed, structured prompts optimized for use in MidJourney, making creative exploration more accessible and efficient.

## Features

- **Efficient Prompt Generation**: Transforms simple user input into complex and detailed prompts tailored for MidJourney.
- **Fine-Tuned on Custom Dataset**: The model is fine-tuned on a dataset specifically created for relevance to image generation and prompt quality.
- **Open-Source Foundations**: Built on the Falcon 7B LLM from HuggingFace.

## Dataset Preparation

The dataset was created using the following steps:

1. **Dataset Collection**: Leveraged Relevance AI to prepare a dataset by reverse-engineering prompts from MidJourney and designing ChatGPT-based prompts.
2. **Bulk Processing**: Processed a large number of inputs to generate a robust database of prompt pairs.
3. **Dataset**: The dataset used for fine-tuning can be found [here](https://docs.google.com/spreadsheets/d/1u2bbcSRV99t0Bg9AHFtakpnI3NrC_cVXlR6tZ7yOKlM/edit?gid=456317866#gid=456317866).

## Model Details

- **Base Model**: Falcon 7B from HuggingFace.
- **Fine-Tuning Framework**: HuggingFace Transformers and Accelerate.
- **Purpose**: Generate high-quality, detailed MidJourney prompts from simple descriptions.
- **Development Environment**: Fine-tuning was performed using Google Colab to leverage its GPU capabilities for efficient training.

## Usage

### Installation

Clone the repository and install the required dependencies:

```bash
$ git clone [<repository-url>](https://github.com/AlexD36/falcon-7b-midjourney-prompt-generator)
$ cd <repository-directory>
$ pip install -r requirements.txt
```

### Inference

Load the fine-tuned model and generate prompts:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "<your-model-name>"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input a simple sentence
input_sentence = "A serene sunset over a mountain lake"

# Generate prompt
inputs = tokenizer(input_sentence, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
generated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Prompt:", generated_prompt)
```

### Example

**Input**: "A futuristic city at night"

**Output**: "A futuristic cityscape illuminated by neon lights at night, featuring towering skyscrapers, flying cars, and a bustling crowd."

## Model Performance

The fine-tuned model demonstrates the ability to:

- Create detailed prompts with creative elements.
- Maintain relevance and coherence with user input.
- Produce MidJourney-ready prompts efficiently.

## Future Work

- Expand the dataset to include more diverse scenarios and styles.
- Improve fine-tuning to enhance creativity and adaptability.
- Provide pre-trained weights for community use.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements or features.

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for providing the Falcon 7B LLM.
- [Relevance AI](https://relevance.ai/) for dataset preparation tools.
- [MidJourney](https://www.midjourney.com/) for inspiring creative prompt generation.
- [Google Colab](https://colab.research.google.com/) for providing a powerful environment for fine-tuning the model.

