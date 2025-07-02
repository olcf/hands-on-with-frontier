# Prompt Engineering

Large Language Models (LLMs) like ChatGPT are powerful tools trained on massive amounts of text to predict and generate human-like language. They don’t "think" or "reason" the way humans do; instead, they statistically predict the next word in a sentence based on the input they’ve seen. <br />

Because LLMs respond based on patterns, *how you ask a question directly affects the quality of the answer*. This is why **prompt engineering** — the practice of crafting inputs to get the most effective outputs — is so important. It helps bridge the gap between your intention and what the model actually delivers. <br />

The main thing that you need to know: asking just one question is **rarely** enough to help. Unless you have a generic and simple problem — such as asking an LLM to remind you what the area of a sphere is — it likely won’t be a result applicable to your research. But don't fret! There are ways to get the results you are looking for.

### Basic Steps

You need to start with an *informed, reasonable question* that you have at least some basic research on. Then, you can ask the LLM a series of questions that (ideally) build on one another. Here is the way to go about that:
1. Use the LLM to suggest **reputable** reading material on your question or problem.
2. Based on what you learn from your reading, ask your first question
3. *Refine* your question based on the answer it gave you. 
4. Test the proposed *solution / insight* to the *issue / body of knowledge*, and once again refine your question. 
5. Continue this loop of refining and testing your query until the issue is resolved.

It's important to note that these steps can be applied to **any** problem, not just coding. We are just coding here for the sake of practice!


## Exercise Requirements
For this activity, you will need a wifi-enabled web browser and access to an LLM. You can use any LLM you choose, but we have listed a few below:

| LLM Model                                 | Notes on model                                     | URL |
|-------------------------------------------|----------------------------------------------------|-----|
| OpenAI ChatGPT (GPT-3.5)                  | No account required <br />Cannot reliably browse the web |[Chat GPT](https://chatgpt.com/)|
| Claude.ai (Claude 3 Haiku)                | Good in-depth reasoning <br />Requires an Account        |[Claude](https://claude.ai/)|
| Hugging Face Chat (Mistral, Zephyr, etc.) | Flexible options <br />Requires an Account               |[HuggingFace](https://huggingface.co/)|

## Prompt Engineering Tips

These tips are inspired by OpenAI’s [official prompt engineering](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api) best practices that we have tailored for generating high-complexity Python code to handle simulations, mathematical modeling, and other related fields.<br /> 

To be clear, these tips are **not** just for Python code. They work great in whatever field you are looking towards and are applicable in most - if not all - fields. Our exercises that we will be doing later are simply about graphical Python.<br />

Without further ado, here are the tips!

---

## 1. Do Your Research

**Tip:** To write a clear and specific prompt, you need to have a good understanding of the thing you're modeling. If you're not already an expert, do some research using *non-LLM-generated sources.* LLM models are a **supplement** to knowledge, not a **replacement**. <br />

**Why It Works:** LLMs (like ChatGPT) can sometimes "hallucinate"—they’ll give confident-sounding answers that may not be factually correct. To avoid being misled, always cross-check important information using **reputable, verifiable sources**. You can use the model to help *find* sources if it can see the internet or has it cached—but then you should verify that those sources are real and trustworthy. <br />

#### Reputable Source Types

1. University Course Websites
   * Look for .edu domains or syllabi from instutions like MIT OpenCourseWare, Harvard CS, UC Berkeley, etc.
   * Example: MIT 18.06 Linear Algebra
2. Scientific Journals & Prepint Servers<br />
Peer-reviewed journals via:
   * [Google Scholar](https://scholar.google.com/)
   * [PubMed](https://pubmed.ncbi.nlm.nih.gov/) (biomedical)
   * [arXiv](https://arxiv.org/) (physics, CS, math -not peer-reviewed, but respected highly)<br />
   **Tip:** Ask the LLM to find the DOI or publisher link, then visit that site.
3. Textbooks and Reference Books<br />
   You can get these suggested by a model, but you can verify them on:
   * [OpenStax](https://openstax.org/)
   * [Project Gutenberg](https://www.gutenberg.org/)
4. Educational Coding Platforms
   Python/math tutorials from:
   * [Real Python](https://realpython.com/)
   * [GeeksforGeeks](https://www.geeksforgeeks.org/)
   * [W3schools](https://www.w3schools.com/python/) (full learn-python tutorial)
5. Government and Academic Research Instritutions<br />
   Reputable sites with .gov or .org domains, such as:
   * [NASA](https://www.nasa.gov/)
   * [NOAA](https://www.noaa.gov/)
   * [Oak Ridge National Laboratory](https://ornl.gov/)
   * [National Institues of Health](https://www.nih.gov/)
6. Professional Societies
   * IEEE, ACM, AMS, SIAM, APS, etc. offer trusted publications and learning resources.
   * They often have "student resources" sections.

#### Example Use

Starting Prompt: <br />
> Tell me about Mandelbrot sets.

Refined Prompt after some Reading: <br />
> Give me 3 educational sources that explain Mandelbrot sets and show how to draw them in Python. Include information on which parameters affect rendering

## 2. Be Specific and Clear

**Tip:** Spell out what you want, down to the function’s purpose and expected inputs/outputs. Be as specific to the project as you can. The more detail, the better! <br />

**Why it Works:** LLMs work better with precise instructions because they’re pattern-matchers, **not** mind-readers. <br />

#### Example Use

Bad Prompt: <br />
> Write python for Leslie Matrix

Better Prompt: <br />
> Write a Python function to simulate population growth using a Leslie matrix. Use a total of three age groups.

## 3. Use Follow-Up Questions to Reiterate

**Tip:** After a first response, refine your request by building on what you got. <br />

Why it Works: LLMs don’t improve over time—but *you* can improve the result by prompting iteratively, using the model’s previous output as context. <br />

#### Example Use

Let's say you are working on creating that Leslie Matrix code. You prompt the model: <br />
> Write a Python function to simulate population growth using a Leslie matrix for three age groups.
The model might generate something like this:
```python
initialized_data = [...]
...

for year in range(years):
    new_population = [0, 0, 0]
    new_population[0] = (
        f_rate[0] * population[0] +
        f_rate[1] * population[1] +
        f_rate[2] * population[2]
    )
    new_population[1] = ...
```
Which is all fine and good. But what if you wanted it to use a cleaner, more robust method for matrix multiplication? You can then build upon your previous prompt with:
> Now rewrite the function using vectorized NumPy operations instead of for-loops.

## 4. Use Step-By-Step Instructions

**Tip:** Break down the task into order parts. <br />

**Why It Works:** LLMs respond well to sequential logic—just like the way their training data is structured. <br />

#### Example Usage

> First define a 3x3 Leslie matrix. Then initialize a population vector. Then use matrix multiplication to simulate population change over 10 years.

## 5. Provide Background Information in the Prompt

**Tip:** Briefly include necesarry context in prompts to avoid assumptions <br />

**Why It Works:** LLMs don't retain past knowledge unless you reintroduce it. Supplying context narrows down ambiguity. <br />

#### Example Usage

> Assume the user is modeling weather systems on a grid. Write a Python class that updates a 2D grid based on neighbor values.

## 6. Specify the Desired Format and Output Structure

**Tip:** Tell the model to output code only, include comments, or return JSON-format if needed. You can also ask it to <br />

**Why It Works:** LLMs try to mimic the *style* of exmaples. Giving format expectations set a clear *stylistic precedent*. As for output structure, asking them to organize their output helps them output more logically and mirror well-formed coding practice. <br />

#### Example Usage

> Return only the Python code with the inline comments explaining each step.
> Divide the output into three sections: (1) input parameters, (2) computation, and (3) Matplotlib visualization.

## 7. Ask for Comparisons or Alternatives 

**Tip:** Have the LLM create and explain multiple approaches towards a problem as well as their trade-offs. This is *especially* good for coding optimizations! <br />

**Why It Works:** This triggers pattern generation from comparative texts, such as "Method A vs. Method B" documents present in the model's training data. <br />

#### Example Usage

> Write two Python functions to estimate pi—one using Monte Carlo simulation, and one using a Taylor series expansion. Compare their performance.

## Request Explanations Alongside Code

**Tip:** Ask for brief explanations of what each block of code is doing. <br />

**Why It Works:** This makes the output reader-friendly (scaffolds understanding) and ensures that the model is *synthesizing sources to make a solution*, not just *copy-pasting one solution and attempting to apply it*. <br /> 

#### Example Usage

> Write a function to numerically solve a homogeneous 2nd-order differential equation given variable coefficents as an input. Explain what the code does at each step.

# Assignment - Fractal Coding Challenge
Use what you’ve learned about prompt engineering to **design a prompt for ChatGPT (or another free LLM)** that gets it to write Python code that draws a fractal and uses user input of your choice to determine the parameters that shape or color the fractal. <br />
  
**Goal:** <br />
Generate code that creates a fractal image (e.g., Mandelbrot set, Koch snowflake, Sierpinski triangle). Modify it to take some parameters as input that allows the user to customize it to make it their personal fractal- so for example, you have them enter their first name or their favorite number  and then have the program use that to generate unique input parameters for generating the fractal. <br />

Creativity is key! You can make the customization whatever input you would like. <br />

### Your Task:
1. Write a **clear, well-structured prompt** using at least 4 of the techniques we talked about.
2. Paste your prompt into your large-language model (LLM) of choice.
3. Cut and paste the generated code into a file on your machine and attempt to run it. You might have to modify the code to fit, as LLMs often times does not perfectly write the code in your context.
4. Modify the prompt using the tips and techniques. Retest the new modified code.
5. Continue to generate more prompts to improve your code until satisfied. Repeat until the code is functional.
<br />

Once you have finished, put the following into a text tile:
* Your original prompt
* The code and image it generated
* A short, 2-3 sentence reflection on this workshop. What technique helped the most? What technique didn't help? What would you change?  

