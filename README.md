<div align="center">

  <h1> TinyStories Regional <img src="https://png.pngtree.com/png-vector/20220812/ourmid/pngtree-indian-flag-design-png-png-image_6108311.png" width="30"> </h1>
  <i>A framework for development of Small Language Models for Indian regional languages, serving both as a practical alternative to LLMs and as a foundation for comparative analysis of tokenization strategies, machine translation performance, and linguistic complexity</i>
  <a href="https://vizuara.ai/"> <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3a1a36ff-6d9a-4ee7-9494-3ae38adfe134_1920x600.png" alt="Vizuara Logo" style="width:90%;"> </a>

  [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat)](https://arxiv.org/abs/1234.56789)
  [![Huggingfaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/TinyStories-Regional)

  [![nirvan](https://img.shields.io/badge/nirvan-black?logo=github&logoColor=white&labelColor=black&color=black&style=flat)](https://github.com/nirvan840)
  [![malhar](https://img.shields.io/badge/malhar-black?logo=github&logoColor=white&labelColor=black&color=black&style=flat)](https://github.com/malharinamdar)
  [![agnivo](https://img.shields.io/badge/agnivo-black?logo=github&logoColor=white&labelColor=black&color=black&style=flat)](https://github.com/agme2019)
  [![raj](https://img.shields.io/badge/🌐-rajdandekar-black?logo=globe&logoColor=white&labelColor=black&color=black&style=flat)](https://www.linkedin.com/in/raj-abhijit-dandekar-67a33118a/?originalSubdomain=in)
  
</div>
<br>

> [!IMPORTANT]
> * <i> `✨ Going through a paper can be tough!✨`</i>
> * <i> `✨ Below is an easy-to-understand comprehensive guides and results for our research ✨`</i>
> * <i> This repository provides resources and code for our TinyStories-Regional framework, which extends <br> the TinyStories approach ([Eldan & Li, 2023](https://arxiv.org/abs/2305.07759)) to three major Indian languages: `Hindi, Marathi, and Bangla`. </i>
> * <i> Our framework enables `training and inference` with Small Language Models containing `5-50M` parameters. </i>


> [!NOTE]
> #### _A special thanks to_
> * <i> <a href="https://tensordock.com/">TensorDock</a> for providing compute! Check them out for easy-to-deploy and affordable GPU/CPU VMs 💚 </i>
> * <i> Microsoft for inspiring us with their original <a href="https://arxiv.org/abs/2305.07759">TinyStories</a> paper 💙 </i>
> * <i> <a href="https://huggingface.co/sarvamai">Sarvam</a>, <a href="https://huggingface.co/TWO">SUTRA</a>, and <a href="https://karpathy.ai/">Andrej Karpathy</a> for their open-source efforts ❤️ </i>

> [!WARNING]
> * <i> The first version of our TinyStories Regional paper is now on arXiv and is currently being refined :) </i>
> * <i> Some references to the paper below currently might not be accessible </i> 

```sh
git clone https://github.com/nirvan840/Vizuara-TinyStories-Regional.git
```
```sh
pip install g4f[all] aiolimter transformers datasets huggingface_hub sentencepiece tiktoken wandb tqdm torch numpy 
```

---

<br> 

# 📚 Table of Contents

- ### 🗂️ Dataset Generation
  - #### [✍️ Preparing Prompts](#preparing-prompts)
  - #### [💬 Prompting an LLM](#prompting-a-model)
- ### ⚙️ Training SLMs
  - #### [🔤 Tokenizing Data](#tokenizing-data)
  - #### [🏋️ Training the Model](#training-the-model)
- ### 🔍 Inference and Evaluation
  - #### [🤖 SLM Inference](#inference-models-local-or-hf)
  - #### [📊 Evaluate Inference/Stories](#evaluate-inference-stories)
- ### 📈 Results
  - #### [⚙️ Hyperparameter Comparisons](#hyperparameter-comparisons)
  - #### [💡 Inference Examples](#inference-examples)
  - #### [🆚 Synthetic VS Translated](#synthetic-vs-translated)
  - #### [🆚 Tokenizer Comparison](#tokenizer-comparison)
  - #### [🔎 Language Complexity](#language-complexity)
  - #### [✅ A Fitting Use Case](#a-fitting-use-case)
- ### 💰 Costs
  - #### [⏱️ Training Time and Costs](#training-time-and-costs)
  - #### [🔄 Replicating the Project](#replicating-the-project)

---

<br> 

# 🗂️ Dataset Generation 

> [!WARNING] 
> <i> This repository provides code to generate data by making API calls to SOTA models (4o, 4o-mini, Gemini-flash-1.5, etc.) using the [GPT-4-free (G4F)](https://github.com/xtekky/gpt4free) repository. This repository is provided for research purposes only. We do not intend to promote using this repository for large-scale dataset generation; respect all terms of service for any API or model you use. Ensure appropriate attribution and licensing for any generated content </i>

> [!NOTE]
> - Our datasets for Hindi, Marathi and Bangla, generated using GPT-4o-mini, are open-sourced on our HF
> - Translated versions (Hindi and Bangla) of [Microsoft's TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset can also be found on our HF
> - Translated versions (our Hindi ➡️ Bangla ; Hindi ➡️ Marathi ; Bangla ➡️ Hindi) will be soon on our HF

<h2 id="preparing-prompts">✍️ Preparing Prompts</h2>
<ul>
  <p><li>Each prompt is generated by sampling a unique set of a noun, a verb, an adjective and a feature</li></p>
  <p><li>To modify the list of nouns, verbs, etc., please modify <code>.txt</code> files at <code>prompting/prompt_gen/<i>&lt;lanauge&gt;</i></code></li></p>
  <p></p><li>Prompt templates/complexities can be referred to/modified through <code>prompting/prompt_gen/create_prompts.py</code>.</li> 
      <ul> <li>We compare various complexities in our paper and find <code>2+</code> to be optimal</li> </ul></p>
  <p><li>Unique (sampling is unique and not random) prompts can be generated by running <code>generate_prompts.py</code></li></p>
  <p><li>Generated prompts are written to a <code>.json</code> file. Sharding the file is recommended (below)</li></p>
</ul>

<i> To generate prompts please run: </i>
```sh
python prompting/prompt_gen/generate_prompts.py
```

<br>

<h2 id="prompting-a-model">💬 Prompting an LLM</h2>
<ul>
  <p><li>Prompts are read from the <code>.json</code> file/shards and "sent" to the specified LLM (using <a href="https://github.com/xtekky/gpt4free">G4F</a>)</li></p> 
  <p><li>Multithreaded API calls result in a max speed of <code>100 stories/min</code> for GPT-4o-mini and GPT-4o (occasionally).</li>
      <ul>
        <li>It is recommended number of threads = number of <code>4 x vCPUs</code></li>
        <li>Each thread writes to a common file. Once every 10% of total progress</li>
        <li><i>Optimal config</i>: <code>16 vCPUs</code>, running <code>4 sessions</code> concurrently (one for each shard), each with <code>16 threads</code></li>
      </ul></p>
  <p><li>Please look into <code>prompting/make_requests.py</code> for customizing the prompting schema</li>
      <ul>
        <li>Generated stories are written to <code>.json</code> files</li>
        <li>It is recommended to upload these files to HF for seamless integration while training models (below)</li>
      </ul>
  </p>
  <p><li>Please look into <code>prompting/request_helper.py</code> for a detailed look into the process</li>
      <ul>
        <li>API/LLM is prompted until a valid story is generated for each prompt</li>
        <li>Various regex (data cleanup) features</li>
      </ul></p>
</ul>

_Optimal prompt complexity/template:_
```text
f```Write a short story in {language} (in Devanagari script) suitable for 5-to-7-year-old children.
Use simple, easy-to-understand words and limit the story to 3-4 short paragraphs (around 200-300 words).
The story should feature a clear beginning, middle, and end. Incorporate the verb "{verb}", the noun "{noun}", and the adjective "{adjective}" naturally into the story.
The story should also integrate the conclusion/tone "{feature1}" through actions and outcomes without directly stating the tone (e.g., do not use "खुश" or similar words explicitly).
Remember to only use simple words and keep the story short!

Return the output as a JSON dictionary in the following format:
{
    "story": "your_generated_story"
}```
```
_To start the data generation process, please run the script:_
```sh
python prompting/make_requests.py
```
_TIP💡: To run data generation in the background (detached VM session):_
```sh
tmux new -s session_name
```

---

<br>

# ⚙️ Training Small Language Models (SLMs) 

> [!IMPORTANT]
> * It is essential that data is tokenized correctly <bos token> story <eos token> (read below)
> * Lower end GPUs <code>T4</code> (Collab), <code>P100</code> (Kaggle) can be used to train models in <code><24hrs</code> on our datasets!

> [!NOTE]
> * We utilize Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) repository (with modifications) to train models
> * Our training script supports multi-GPU training (DDP), progress (TQDM) and logging support (WANDB), along with easy customizability


<h2 id="tokenizing-data">🔤 Tokenizing Data</h2>
<ul>
  <p><li>The <code>.json</code> files uploaded to HF in the previous stage server as our dataset</li></p>
  <p><li>The entire dataset is tokenized before training. Token IDs are stored in <code>.bin</code> files
      <ul>
        <li>The dataset to be tokenized can be chosen as per <code>training-inference/data/prepare.py line 38-46</code></li>
            <ul>
              <li>To tokenize a custom HF dataset, please look into lines <code>59-61</code> & <code>112-133</code></li>
            </ul>
        <li>The <code>.bin</code> files must be appropriately placed in a folder in <code>training-inference/data/</code></li>
            <ul>
              <li>This folder must be specified in <code>config.py</code> under the <code>dataset</code> variable</li>
            </ul>
        <li>Use <code>training-inference/data/decode_data.py</code> to decode and print first 500 tokens from a <code>.bin</code> file</li>
            <ul>
              <li> Ensure that the decoded tokens follow the format: <code>&lt;bos token&gt; story1 &lt;eos token&gt; &lt;bos token&gt; story2 &lt;eos token&gt;...</code></li>
            </ul>
      </ul>
  </li></p>
  <p><li>Tokenization is carried out by the script <code>training-inference/data/prepare.py</code></li></p>
  <p><li>We provide direct support for the following tokenizers:
      <ul>
        <li><a href="https://huggingface.co/sarvamai/sarvam-1">Sarvam (sarvam-1)</a> (Indic)</li>
        <li><a href="https://huggingface.co/TWO/sutra-mlt256-v2">Sutra (mlt256-v2)</a> (Indic)</li>
        <li><a href="https://github.com/openai/tiktoken">Tiktoken (GPT-2)</a> (English)</li>
      </ul>
  </li></p>
  <p><li>We provide easy support for any additional tokenizers available on HF.
      <ul>
        <li>Specify the new tokenizer along similar lines as<code>training-inference/data/prepare.py line 19-35</code></li>
        <li>For this to work, HF tokenizers must have valid <code>eos</code> and <code>bos</code> tokens</li>
      </ul>
  </li></p>
</ul>

_To tokenize an HF dataset, run the script:_
```sh
python training-inference/data/prepare.py
```

<br>

<h2 id="training-the-model">🏋️ Training the Model</h2>
<ul>
  <p><li>Training can be resumed for locally saved models as well as those from Vizuara-HF!</li></p>
  <p><li>Changes to the training configuration can be easily made through <code>training-inference/config.py</code></li></p>
  <p><li>Model weights are checkpointed every <code>eval_iters</code> and saved at <code>training-inference/out/</code> as <code>.pt</code> files</li></p>
</ul>

_To start training, run:_
```sh
# Single GPU
python training-inference/train.py training-inference/config.py
```
```sh
# Multi GPU (DDP)
torchrun --standalone --nproc_per_node=num_gpus training-inference/train.py training-inference/config.py
```
        
_Automated multi-config training:_
```sh
chmod +x training-inference/utils/automate-training.sh
./training-inference/utils/automate-training.sh
```

---

<br>

# 🔍 Inference and Evaluation

> [!IMPORTANT]
> * All evaluations are performed by SOTA LLMs, which may be "biased" towards a language
> * Thus, an <code>8/10</code> Hindi story might not be equivalent in "quality" to an <code>8/10</code> Bangla story
> * It is crucial to understand that as these models evolve, "ratings"/results will drift as compared to today

> [!NOTE]
> * Given the small size of the models, CPU inference is supported!
> * It is crucial to ensure tokenization occurs correctly (refer below)!
> * Data generation scripts are repurposed for evaluation of stories produced by SLMs

<h2 id="inference-models-local-or-hf">🤖 SLM Inference (Local or HF)</h2>
<ul>
  <p><li>Refer to the sample settings at the bottom of <code>training-inference/config.py</code></li>
      <ul>
        <li>Choose between locally saved models or those from our HF by toggling <code>load_from_hf</code></li>
        <li>Stories generated per prompt, temperature and top_k can be modified</li>
      </ul>
  <p>
  <p><li>Various tokenizers can be used for inference</li>
      <ul>
        <li>We provide direct support for Sarvam, SUTRA and Tikoken</li>
        <li>For adding your own tokenizers a complete understanding of <code>sample.py</code> is recommended :)</li>
      </ul>
  <p>
  <p><li>Multiple prompts (each on a new line) can be mentioned in a text file.
      <ul>
        <li>Refer to <code>training-inference/&lt;language&gt;-prompts.txt</code></li>
        <li>Model can be asked to generate multiple unique stories for each prompt.</li>
        <li><code>.txt</code> and <code>.json</code> outputs are supported.</li>
      </ul>
  </li></p>
</ul>

_To run inference, run the script:_
```sh
python training-inference/sample.py 
```

<br>

<h2 id="evaluate-inference-stories">📊 Evaluate Inference/Stories</h2>
<ul>
  <p><li>We repurpose the data generation code to send stories (using G4F) to SOTA LLMs for evaluation</code></li>
      <ul>
        <li>Set <code>evaluate</code> to True in <code>prompting/make_requests.py</code> and run the script</li>
      </ul>
  <p><li>Each of our SLMs is evaluated in the following manner:</code></li>
      <ul>
        <li>We prepared <code>1000</code> equivalent prompts for each language <code>training-inference/prompt-&lt;langugage&gt;.txt</code></li>
            <ul>
              <li>These <code>1000</code> prompts cover themes such as <code>Adventure, Creativity, Courage, etc..</code></li>
              <li>All the themes and other details can be found in <code>prompting/o1_inference-prompt-generation</code></li>
            </ul>
        <li>Each model produces <code>3 stories</code> per prompt which are sent to SOTA LLMs for eavluation</li>
      </ul>
  </p>
</ul>

_For each model, <code>3000</code> stories are evaluated according to the prompt:_ 
```text
f'''{story}

The given {language} short story is for 5-7-year-old children.
Keeping in mind the target demographic, rate the story on a scale of 1-10 for context awareness, completeness, grammar, fluency, and creativity.
Evaluate context awareness by strictly assessing how well the story's middle and end align with the prompt "{prompt}".
Also, provide an overall rating on a scale of 1-10.

Only return a JSON dictionary in the following format: 
{
"context awareness": "your_context-awareness_score",
"completeness": "your_completeness_score",
"grammar": "your_grammar_score",
"fluency": "your_fluency_score",
"creativity": "your_creativity_score",
"overall": "your_overall_rating"
}'''
```

---

<br>

# 📈 Results
> [!NOTE]
> * In line with the evaluation section above, all stories are evaluated using GPT-4o

<h2 id="hyperameter-comparisons">⚙️ Hyperparameter Comparisons</h2>

> * _This subsection covers results for models trained on synthetic (generated by GPT-4o) data_
> * _We recommend not to compare scores across languages_ 

### Table 1: *Hindi 🪔* – Hyper-parameter Comparison  
*Higher the better — Scores are out of 10 — 3000 stories per configuration evaluated*

| Hidden Size | Layer | Model Size (M) | Eval Loss | Context | Completeness | Creativity | Fluency | Grammar | Overall |
|-------------|-------|-------------|-----------|---------|---------------|-------------|---------|----------|---------|
| 64          | 2     | 4.46        | 1.408     | 5.865   | 6.826         | 7.217       | 7.472   | 7.969    | 7.130   |
| 64          | 6     | 4.65        | 1.182     | 6.412   | 7.122         | 7.314       | 7.901   | 8.466    | 7.439   |
| 64          | 12    | 5.00        | 1.057     | 6.374   | 7.227         | 7.390       | 7.959   | 8.450    | 7.480   |
| 512         | 2     | 41.00       | 0.654     | 7.054   | 7.661         | 7.705       | 8.427   | 8.746    | 7.919   |
| 512         | 6     | 53.00       | 0.518     | 7.734   | 7.783         | 7.806       | 8.554   | 8.912    | 8.185   |
| 512         | 12    | 73.00       | 0.519     | 7.572   | 7.659         | 7.718       | 8.458   | 8.862    | 8.054   |
| 1024        | 2     | 94.00       | 0.581     | 7.344   | 7.798         | 7.829       | 8.516   | 8.825    | 8.062   |
| 1024        | 7     | 153.00      | 0.513     | 7.695   | 7.806         | 7.830       | 8.580   | 8.910    | 8.164   |

### Table 2: *Marathi 🥁* – Hyper-parameter Comparison  
*Higher the better — Scores are out of 10 — 3000 stories per configuration evaluated*

| Hidden Size | Layer | Model Size (M) | Eval Loss | Context | Completeness | Creativity | Fluency | Grammar | Overall |
|-------------|-------|-------------|-----------|---------|---------------|-------------|---------|----------|---------|
| 64          | 2     | 4.46        | 3.128     | 5.618   | 6.615         | 7.525       | 6.823   | 7.411    | 6.799   |
| 64          | 6     | 4.65        | 2.843     | 6.171   | 6.974         | 7.435       | 7.390   | 8.103    | 7.215   |
| 64          | 12    | 5.00        | 2.524     | 6.219   | 7.009         | 7.288       | 7.471   | 8.184    | 7.210   |
| 512         | 2     | 41.00       | 2.330     | 6.994   | 7.396         | 7.521       | 8.002   | 8.603    | 7.691   |
| 512         | 6     | 53.00       | 2.076     | 7.245   | 7.407         | 7.553       | 8.106   | 8.723    | 7.807   |
| 512         | 12    | 73.00       | 1.811     | 7.281   | 7.565         | 7.664       | 8.156   | 8.739    | 7.881   |
| 1024        | 2     | 94.00       | 0.680     | 6.728   | 7.184         | 7.484       | 7.687   | 8.295    | 7.476   |
| 1024        | 7     | 153.00      | 0.619     | 7.275   | 7.152         | 7.540       | 7.986   | 8.625    | 7.698   |

### Table 3: *Bangla 🐅* – Hyper-parameter Comparison  
*Higher the better — Scores are out of 10 — 3000 stories per configuration evaluated*

| Hidden Size | Layer | Model Size (M) | Eval Loss | Context | Completeness | Creativity | Fluency | Grammar | Overall |
|-------------|-------|-------------|-----------|---------|---------------|-------------|---------|----------|---------|
| 64          | 2     | 4.46        | 1.514     | 6.663   | 7.097         | 7.469       | 7.797   | 8.424    | 7.490   |
| 64          | 6     | 4.65        | 1.245     | 6.533   | 7.225         | 7.482       | 7.975   | 8.454    | 7.534   |
| 64          | 12    | 5.00        | 1.136     | 6.760   | 7.289         | 7.563       | 7.968   | 8.507    | 7.617   |
| 512         | 2     | 41.00       | 0.693     | 7.373   | 7.491         | 7.644       | 8.314   | 8.782    | 7.922   |
| 512         | 6     | 54.00       | 0.559     | 7.507   | 7.645         | 7.693       | 8.420   | 8.816    | 8.016   |
| 512         | 12    | 73.00       | 0.544     | 7.525   | 7.718         | 7.743       | 8.450   | 8.836    | 8.041   |
| 1024        | 2     | 94.00       | 0.609     | 7.407   | 7.470         | 7.626       | 8.293   | 8.786    | 7.916   |
| 1024        | 7     | 153.00      | 0.557     | 7.567   | 7.639         | 7.740       | 8.409   | 8.832    | 8.037   |

### Key Findings 
- SLMs with only `53M parameters` can `consistently` generate coherent, grammatical stories in Indian languages
- Increasing model size from 4.5M to 153M parameters consistently improves performance across all languages
- Grammar scores show the highest performance, while context awareness remains the most challenging
- Smaller (5M) models can generate coherent stories, albeit with poor contextual awareness
- The `optimal configuration` appears to be `53M`: `Hidden Dimension: 512` with `Layers: 6`, balancing performance and efficiency

<br>

<h2 id="inference-examples">💡 Inference Results</h2>

> * _All inference examples are from models trained on synthetic data_
> * _`Starting prompts` for inference for each story are mentioned in `** ... **`_
> * _Currently, models only support story completion based on starting prompts :)_

### Hindi 5M
```text
**घने जंगल में, एक चमकदार पेड़ रहस्यमय शक्ति से गूंज रहा था।** एक दिन, उन्होंने देखा कि पेड़ पर बहुत सारे रंग-बिरंगे फूल खिले हुए हैं। मोहन ने सोचा, "ये फूल कितने सुंदर हैं!" उसने अपने दोस्तों से कहा, "चलो, हम यहाँ कुछ खास करते हैं।" सभी दोस्त एक साथ मिलकर पेड़ के नीचे बैठ गए।

बच्चे बहुत खुश थे और उन्होंने तय किया कि वे यहाँ कुछ मजेदार करेंगे। मोहन ने एक छोटी सी बंदूक मांगी और कहा, "हम इसे पानी देंगे ताकि यह और भी सुंदर बने।" सभी ने मिलकर पत्थर को पानी दिया और बहुत ध्यान से काम किया। लेकिन एक दोस्त ने कहा, "तुम्हें इसे बहुत पसंद आएगा।" मोहन ने समझाया कि हमें एक-दूसरे के साथ मिलकर काम करना चाहिए।

फिर, मोहन ने कहा, "हम सबको मिलकर काम करना चाहिए है। हमें हमेशा मिलकर काम करना चाहिए।" सभी ने सहमति जताई और एक पेड़ के नीचे बैठ गए। जैसे ही उन्होंने घंटी को बजाया, पेड़ ने एक मीठी आवाज़ में कहा, "धन्यवाद, बच्चों! तुमने मेरे लिए यह कर दिया।" सभी बच्चे चौंक गए और पेड़ के पास बैठ गए। उस दिन मोहन और उसके दोस्तों ने सीखा कि एकता में ताकत होती है। उन्होंने समझा कि जब हम मिलकर काम करते हैं, तो हम हमेशा एक-दूसरे की मदद कर सकते हैं। और इस तरह, मोहन और उसके दोस्तों ने एक नई दोस्ती की शुरुआत की।

>> Evaluation: Context: 6 | Completeness: 8 | Grammar: 9 | Fluency: 8 | Creativity: 7 | Overall: 7.5
```

<details>
<summary> <h4> <i> English Translation for Hindi 5M </h4> </i> </summary>
  
```text
**In the dense forest, a glowing tree was echoing with mysterious power.** One day, they saw that many colorful flowers had bloomed on the tree. Mohan thought, "These flowers are so beautiful!" He said to his friends, "Come on, let's do something special here." All the friends sat together under the tree.

The children were very happy and they decided that they would do something fun here. Mohan asked for a small gun and said, "We will give it water so that it becomes even more beautiful." Everyone gave water to the stone together and worked very carefully. But one friend said, "You will really like this." Mohan explained that we should all work together.

Then Mohan said, "We all should work together. We should always work together." Everyone agreed and sat under a tree. As soon as they rang the bell, the tree said in a sweet voice, "Thank you, children! You did this for me." All the children were surprised and sat near the tree.

That day, Mohan and his friends learned that there is strength in unity. They understood that when we work together, we can always help each other. And in this way, Mohan and his friends began a new friendship.
```
</details>

### Hindi 53M 
```text
**घने जंगल में, एक चमकदार पेड़ रहस्यमय शक्ति से गूंज रहा था।** पेड़ की पत्तियाँ सूरज की रोशनी में चमक रही थीं। बच्चे उस पेड़ के पास गए और देखा कि पेड़ के नीचे एक प्यारा सा खरगोश बैठा है। खरगोश ने कहा, "मैंने सुना है कि इस जंगल में एक जादुई फूल है, जो हर साल एक बार खिलता है, तुम्हारी इच्छा पूरी करता है।"

बच्चों ने सोचा कि अगर वे उस जादुई फूल को ढूंढ लेंगे, तो उनकी सभी इच्छाएँ पूरी हो जाएँगी। उन्होंने खरगोश से पूछा, "हम उस फूल को कैसे ढूंढ सकते हैं?" खरगोश ने मुस्कुराते हुए कहा, "तुम्हें अपने दिल की सच्चाई से काम करना होगा।"

बच्चों ने मिलकर जंगल में खोज शुरू की। उन्होंने कई पेड़ और झाड़ियाँ पार कीं। अंत में, उन्होंने एक चमकदार पेड़ देखा, जो पहले से भी ज्यादा सुंदर था। बच्चों ने उस पेड़ के चारों ओर बैठकर अपनी इच्छाएँ मांगी। अचानक, पेड़ पर एक सुगंधित फूल खिलता हुआ दिखाई दिया। बच्चों ने उस फूल को देखकर एक-दूसरे को देखा और उनके दिलों में एक रहस्यमय रूप से गरमाहट भरा अनुभव हुआ।

>> Evaluation: Context: 9 | Completeness: 9 | Grammar: 9 | Fluency: 9 | Creativity: 8 | Overall: 8.8
```

<details>
<summary> <h4> <i> English Translation for Hindi 53M </h4> </i> </summary>
  
```text
**In the dense forest, a glowing tree was echoing with mysterious power.** The leaves of the tree were shining in the sunlight. The children went near that tree and saw that a cute little rabbit was sitting under the tree. The rabbit said, "I have heard that there is a magical flower in this forest, which blooms only once a year and fulfills your wish."

The children thought that if they found that magical flower, all their wishes would come true. They asked the rabbit, "How can we find that flower?" The rabbit smiled and said, "You must work with the truth of your heart."

The children together began searching in the forest. They crossed many trees and bushes. Finally, they saw a glowing tree, even more beautiful than before. The children sat around that tree and made their wishes. Suddenly, a fragrant flower bloomed on the tree. Seeing that flower, the children looked at each other and felt a mysteriously warm sensation in their hearts.
```
</details>

### Marathi 5M 
```text
**सूर्योदयाच्या वेळी, दोन घट्ट मित्र नदीकाठच्या अनोळखी पाऊलखुणांचा पाठलाग करत होते.** त्यांनी एकत्र येऊन एक मोठा झाड पाहिला, जो त्यांना पाहून त्यांच्यातील आवाज ऐश्वर्यपूर्ण आवाजात एक अद्भुत दृश्य दिसलं. ते झाड खूप सुंदर होते. आर्यन त्या झाडाच्या पानांवर चढून खेळत होता.

त्यांच्या आवाज आणि आर्यनने ठरवलं की, ते झाडावर चढून, झाडावर चढतील. झाडावर चढल्यानंतर आर्यनने एका मोठ्या झाडाच्या खाली एक मोठा पक्षी पाहिला. तो पक्षी आकाशात उडत होता, पण आर्यन खूप खूष झाला. आर्यनने त्या पक्षाला स्पर्श केला आणि त्याच्या पानावर एक मोठा आवाज आला.

त्या दिवशी आर्यनने त्या पक्ष्याने एक जादुई पंख घेतला आणि त्या पक्ष्याला एकत्र बसले. ते पक्षी आकाशात उडायला लागले, पण चिरंतन एकदम सुंदर होता. तो पक्षी खूप सुंदर आणि सुंदर होता, तो त्याला म्हणाला, "तू एकटा असणारा गाणं गाणं ऐकतो, परंतु मी तुझ्या सोबत आहेस!" आर्यनने हळू हळू हळूच उचलून त्याच्या गाण्यात रमले एक अद्भुत गोष्ट शिकली. आता तो एकदम शांत होता, जो सर्वांच्या मनात राहिला.

>> Evaluation: Context: 6 | Completeness: 7 | Grammar: 6 | Fluency: 6 | Creativity: 8 | Overall: 6.5
```

<details>
<summary> <h4> <i> English Translation for Marathi 5M </h4> </i> </summary>
  
```text
**At sunrise, two close friends were following some unfamiliar footprints by the riverside.** Together, they came upon a large tree, and as they looked at it, a majestic voice echoed within them, revealing a wondrous sight. The tree was incredibly beautiful. Aryan began climbing and playing among its leaves.

Moved by the voice and their curiosity, Aryan decided to climb the tree. Once at the top, he saw a large bird resting beneath the tree. The bird soon soared into the sky, and Aryan felt immense joy. As he reached out and touched the bird, a loud sound came from the leaves.

That day, Aryan received a magical feather from the bird, and he sat with the bird. The bird began flying through the sky, and the moment was breathtaking. The bird was stunning and graceful. It said to Aryan, “You may sing songs alone, but I am with you!” Slowly, Aryan lifted his spirits and became immersed in its song, learning a wonderful lesson. From that moment on, he was filled with peace—a peace that stayed in everyone’s hearts.
```
</details>

### Marathi 53M 
```text
**सूर्योदयाच्या वेळी, दोन घट्ट मित्र नदीकाठच्या अनोळखी पाऊलखुणांचा पाठलाग करत होते.**

त्यांनी एक गडद जागा पाहिली जिथे एक मोठा, चमकणारा दगड होता. "हे काय आहे?" समीर विचारला. आर्यनने उत्तर दिलं, "कदाचित हे नक्षत्र आहे. चला, आपण त्या दगडाच्या मागे जाऊन पाहू!" ते धाडसाने दगड ओलांडू लागले. दगडाच्या मागे एक लपलेलं गुहा दिसलं.

गुहेच्या आत गेल्यावर, त्यांना अनेक रंगीत रत्न आणि चमकदार वस्त्रं सापडली. समीर आणि आर्यन खूप खूश झाले. त्यांनी ठरवलं की ते या गुप्त ठिकाणाला परत येतील. त्या दिवशी, त्यांनी खूप मजा केली आणि गुप्त ठिकाण शोधून काढले. त्यांनी एकत्र येऊन त्या गुप्त ठिकाणाबद्दल आपल्या मित्रांना सांगितलं. त्या दिवशी त्यांनी एक अद्भुत अनुभव घेतला, जो सदैव लक्षात राहील.

>> Evaluation: Context: 7 | Completeness: 9 | Grammar: 8 | Fluency: 8 | Creativity: 8 | Overall: 8
```

<details>
<summary> <h4> <i> English Translation for Marathi 53M </h4> </i> </summary>
  
```text
**At sunrise, two close friends were following unfamiliar footprints along the riverside.**

They saw a dark place where there was a large, glowing stone. "What is this?" Sameer asked. Aryan replied, "Maybe it's a star fragment. Come on, let’s go see behind that stone!" They bravely started crossing the stone. Behind the stone, they saw a hidden cave.

When they entered the cave, they found many colorful gems and shining robes. Sameer and Aryan were very happy. They decided they would return to this secret place. That day, they had a lot of fun and discovered a hidden spot. Together, they told their friends about the secret place. That day, they had a wonderful experience that they would always remember.
```
</details>


### Bangla 5M 
```text
**সূর্যোদয়ের সময়, দুই সেরা বন্ধু নদীর ধারে অদ্ভুত পায়ের ছাপ অনুসরণ করছিল।** সে মনে মনে ভাবছিল, আজকের দিনটি কত সুন্দর হয়েছে। সে দৌড়াতে দৌড়াতে নদীর কাছে পৌঁছাল। নদীর জল ছিল পরিষ্কার এবং নীলু ফুলটি ঝরঝরে।

সাগরের নিচে পৌঁছানোর পর, রবি দেখতে পেল একটি ছোট্ট মাছ। মাছটি খুব সুন্দর ছিল এবং তার রঙ ছিল সাদা। রবি মাছটিকে খুব ভালোবাসত। সে মাছটিকে কাছে নিয়ে গেল এবং কিছুক্ষণ মাছটিকে দেখল। মাছটি বলল, "তুমি খুব ভালো কাজ করেছ! আমি তোমার জন্য জল ছিটিয়ে দিতে চাই।" রবি খুব খুশি হলো এবং মাছটিকে আরও কাছে নিয়ে গেল।

রবি মাছটিকে ধন্যবাদ জানাল। সে জানাল, মাছটি তার পাশে আছে, যেন তার সাথে খেলা করছে। রবি মাছটির সাথে খেলা শুরু করল এবং মাছটি তার সাথে দৌড়াতে লাগল। সেদিন রবি বুঝতে পারল, প্রকৃতির সাথে বন্ধুত্ব করা সবসময় ভালো।

>> Evaluation: Context: 6 | Completeness: 7 | Grammar: 8 | Fluency: 7 | Creativity: 8 | Overall: 7.5
```

<details>
<summary> <h4> <i> English Translation for Bangla 5M </h4> </i> </summary>
  
```text
**At sunrise, two best friends were following strange footprints by the river.** One of them was thinking to himself how beautiful the day had become. He ran and reached the riverside. The river’s water was clear, and the blue flower was fresh.

After reaching under the sea, Robi saw a little fish. The fish was very beautiful and its color was white. Robi loved the fish very much. He brought the fish close and watched it for a while. The fish said, "You’ve done a very good job! I want to sprinkle water for you." Robi was very happy and brought the fish even closer.

Robi thanked the fish. He felt like the fish was beside him, as if playing with him. Robi started playing with the fish, and the fish began to run along with him. That day, Robi realized that making friends with nature is always a good thing.
```
</details>

### Bangla 53M
```text
**সূর্যোদয়ের সময়, দুই সেরা বন্ধু নদীর ধারে অদ্ভুত পায়ের ছাপ অনুসরণ করছিল।** তাদের কাছে এক অদ্ভুত দারুন ছিল, তারা যেন তাদের জন্য নতুন একটি গল্প বলবে। এই গল্পটা ছিল খুব মজার এবং তারা সেই গল্প বলার সময় হাসাহাসি করতে লাগল।\

সন্ধ্যা হলে, তারা বাড়ির পথে হাঁটতে থাকল। সূর্য তখন আকাশে লাল হয়ে উঠেছিল এবং নদীর পাড়ে আধো আলোয় সবকিছুকে উজ্জ্বল করে দিয়েছিল। শিশুরা একসাথে হাঁটতে হাঁটতে মনে মনে ভাবছিল, আজকের দিনটা কত সুন্দর ছিল! তারা জানত, এই ধরনের সময়গুলো তাদের বন্ধুত্বকে আরও শক্তিশালী হবে।

এখন তারা প্রতিদিন খেলতে আসবে, নতুন নতুন গল্প বলবে। নদীর শান্ত জল এবং সূর্যের আলোতে তাদের খেলা সব সময় মনে রাখার মতো থাকবে।

>> Evaluation: Context: 9 | Completeness: 9 | Grammar: 8 | Fluency: 9 | Creativity: 8 | Overall: 9
```

<details>
<summary> <h4> <i> English Translation for Bangla 53M </h4> </i> </summary>
  
```text
**At sunrise, two best friends were following strange footprints by the riverbank.** They had a strange charm with them, as if it would tell them a new story. That story was very funny and while telling it, they started laughing.

In the evening, they began walking home. The sun had turned red in the sky and lit everything brightly along the riverbank in the dim light. The children, walking together, were thinking to themselves how beautiful the day was! They knew that moments like these would make their friendship even stronger.

Now they will come to play every day, tell new stories. Their games in the calm water of the river and in the sunlight will always be something to remember.
```
</details>


<br>

<h2 id="synthetic-vs-translated">🆚 Synthetic vs Translated</h2>

> * _We compare evaluation scores of the 53M model trained on synthetic data (generated by 4o) vs translated data (deeptranslate)_
> * _We have two variants of translated data: Translation of Microsoft's English TinyStories, Translating our synthetic data across languages_

### Key Findings 
- *Very Soon!* 

<br>

<h2 id="tokenizer-comparison">🆚 Tokenizer Comparison</h2>

> _We compare models trained on our synthetic data tokenized via various Indic and non-Indic tokenizers_

### Table 4: Comparison of Tokenizers across Hindi, Marathi, and Bangla  
*Higher the better — Scores are out of 10 — 3000 stories per configuration evaluated*

#### 🪔 Hindi (53M Model)

| Tokenizer Name | Eval Loss | Context | Completeness | Creativity | Fluency | Grammar | Overall |
|----------------|-----------|---------|---------------|-------------|---------|----------|---------|
| Sarvam         | 0.518     | 7.734   | 7.783         | 7.806       | 8.554   | 8.912    | 8.158   |
| SUTRA          | 0.522     | 7.548   | 7.449         | 7.584       | 8.292   | 8.875    | 7.950   |
| Tiktoken       | 0.149     | 6.974   | 7.106         | 7.360       | 7.889   | 8.681    | 7.602   |

#### 🥁 Marathi (53M Model)

| Tokenizer Name | Eval Loss | Context | Completeness | Creativity | Fluency | Grammar | Overall |
|----------------|-----------|---------|---------------|-------------|---------|----------|---------|
| Sarvam         | 0.645     | 7.245   | 7.407         | 7.553       | 8.106   | 8.723    | 7.807   |
| SUTRA          | 0.608     | 7.523   | 7.162         | 7.483       | 8.012   | 8.724    | 7.781   |
| Tiktoken       | 0.167     | 7.014   | 6.742         | 7.137       | 7.524   | 8.451    | 7.374   |

#### 🐅 Bangla (53M Model) 

| Tokenizer Name | Eval Loss | Context | Completeness | Creativity | Fluency | Grammar | Overall |
|----------------|-----------|---------|---------------|-------------|---------|----------|---------|
| Sarvam         | 0.569     | 7.507   | 7.645         | 7.693       | 8.420   | 8.816    | 8.016   |
| SUTRA          | 0.608     | 7.614   | 7.374         | 7.595       | 8.212   | 8.657    | 7.928   |
| Tiktoken       | 0.135     | 7.118   | 6.989         | 7.358       | 7.778   | 8.614    | 7.572   |


## Key Findings
- `Language-specific` tokenizers (Sarvam, SUTRA) consistently `outperform general-purpose` alternatives (Tiktoken)
- `Sarvam` achieves the `highest overall` scores for Hindi and Bangla, with competitive performance for Marathi
- Tiktoken shows the lowest perplexity but struggles with capturing context and completeness
- `Renyi entropy` analysis shows `Sarvam` produces more concentrated token distributions (lower entropy values) compared to SUTRA
- `MorphScore analysis` reveals `SUTRA` better preserves morphological boundaries despite higher entropy

<br>

<h2 id="language-complexity">🔎 Language Complexity </h2>

> [!NOTE]
> * We employ a `dual-perspective` approach using `MorphScore and Rényi entropy` to quantitatively analyze the linguistic complexity of three major Indian languages: Hindi, Bengali, and Marathi
> * We compare language complexity given a certain tokenizer. Results may vary for different tokenizers.
> * Language complexity refers to how hard it is for the SLM to "learn" the training data.

### MorphScore
- We analysed the `morphological alignment of tokenizers` for each language using the `MorphScore`, which quantifies alignment between tokenizer outputs and linguistic morphemes.
- A `morpheme` is the smallest unit of language with meaning, serving as a basic building block for words.

#### Table 5: MorphScore Analysis 
*Higher MorphScore indicates better alignment with true morpheme boundaries*

| Language    |   SUTRA  | SARVAM   |
|-------------|----------|----------|
| Hindi       | 0.7268   | 0.7276   |
| Bnegali     | 0.3002   | 0.3194   | 
| Marathi     | 0.6671   | 0.6620   |

### Rényi Entropy
- To assess the tokenization quality of Sarvam and SUTRA for Indic languages, we employed Rényi entropy as an information-theoretic metric to quantify the uncertainty and diversity in the token distribution resulting from each tokenizer. 
- Specifically, we computed Rényi entropy with `𝛼 = 2.5` across the training corpora for Hindi, Bengali, and Marathi using both tokenizers.
- The choice of 𝛼 `emphasizes high-probability tokens`, highlighting token concentration and how evenly probability mass is distributed across the vocabulary.y.

#### Table 6: Rényi Entropy Analysis
- *Higher Rényi entropy is better when diverse, balanced token usage is desired*
- *Lower Rényi entropy is better when concentrated; efficient tokenization on frequent tokens is preferred.*

| Language    | SUTRA   | Sarvam  |
|-------------|---------|---------|
| Hindi       | 7.1530  | 6.2852  |
| Bengali     | 7.4135  | 6.3579  |
| Marathi     | 7.7620  | 6.5449  |

<br>

<h2 id="a-fitting-use-case">✅ A Fitting Use Case</h2>

### For Researchers
- Comparative analysis of tokenizers, translation methods, and language complexity is resource-intensive with LLMs due to high training and inference costs.
- We introduce a lightweight framework using Small Language Models (SLMs) to enable efficient, scalable comparisons—offering results aligned with LLMs at a fraction of the cost.

### General Purpose
- Regional languages are underrepresented in NLP research.
- Our SLM framework helps bridge this gap by enabling fast, low-cost generation of regional short stories.
- This supports educators in creating child-friendly content with minimal effort. 

### We would love to help you!
- Please contact us if you wish to learn more about training SLMs for your unique regional languages! 

---

<br>

# 💰 Costs

> [!IMPORTANT]
> * `Pipeline Overview`:
>    - Prompt Generation (free 🎁)
>    - Data Generation (free using G4F 🎁)
>    - Training a SLM (<20 USD using TensorDock 💚)
>    - Inference (CPU inference supported ~free 🎁)
> * `Total Cost` to generate your custom Regional-SLM `~15 USD` :)
> * `First Time Setup Effort`:
>    - Assuming intermediate competancy with DL and LLMs
>    - `2-6 hours`; Time is money, after all :)

#### _Detailed breakdown soon!_

---

<br>

# 📝 Citation
If you use Vizuara's TinyStories Regional in your research, please cite us using the following BibText template: 

```text
@misc{TinyStories-Regional,
  author = {Nirvan Patil, Malhar Inamdar, Agnivo Gosai, Raj Dandekar},
  title = {TinyStories Regional: A Framework for Developing Regional SLMs as Proxies for Comparative Analysis},
  year = {2025},
  howpublished = {\url{https://github.com/nirvan840/Vizuara-TinyStories-Regional}},
  note = {Research conducted as part of Vizuara AI Labs}
}

```
