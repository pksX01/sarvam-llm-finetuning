import os
from creds import get_secret
import wandb
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers

def merge_columns(row):
    data = list()
    data.append({"content": row["instruction"], "role": "user"})
    data.append({"content": row["output"], "role": "assistant"})
    row["messages"] = data
    #row['instr_op'] = row['instruction'] + " -> " + str(row['output'])
    return row

def preprocess_function(example):
    if not example.get("messages"):
        return None

    # Ensure all messages have 'content' and 'role', and content is a string
    valid_messages = []
    for msg in example["messages"]:
        if isinstance(msg.get("content"), str) and msg.get("role"):
            valid_messages.append(msg)
        else:
            print(f"Skipping invalid message: {msg}")

    if not valid_messages:
        return None

    try:
        model_ips = tokenizer.apply_chat_template(valid_messages, tokenize=False)
        tokenized_ips = tokenizer(model_ips)
        tokenized_ips["labels"] = tokenized_ips["input_ids"].copy()
        return tokenized_ips
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

def generate_response(question):
    message = [{"role": "user", "content": question}]
    model_ip = tokenizer.apply_chat_template(message, tokenize=False)
    tokenized_ip = tokenizer(model_ip, return_tensors="pt").to("cuda")
    model.eval()
    op_tokens = model.generate(
        **tokenized_ip,
        max_new_tokens=250,
        temperature=0.01,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    op = tokenizer.decode(op_tokens[0], skip_special_tokens=True)
    print("\n\nGenerated response:", op, sep="\n")

if __name__ == '__main__':
    os.environ["HF_TOKEN"] = get_secret("HF_TOKEN")
    os.environ["WANDB_API_KEY"] = get_secret("WANDB_API_KEY")
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    run = wandb.init(project="sarvam_1_llm_finetuning_bhojpuri_qa")

    ds = load_dataset("SatyamDev/alpaca_data_cleaned_bhojpuri")

    ds['train'] = ds['train'].map(merge_columns)

    model_id = "sarvamai/sarvam-1"

    #model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}
                            {% set loop_messages = messages[1:] %}
                            {% set system_message = messages[0]['content'] %}
                            {% else %}
                            {% set loop_messages = messages %}
                            {% set system_message = false %}
                            {% endif %}
                            {% for message in loop_messages %}
                            {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
                            {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
                            {% endif %}
                            {% if loop.index0 == 0 and system_message != false %}
                            {% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}
                            {% else %}
                            {% set content = message['content'] %}
                            {% endif %}
                            {% if message['role'] == 'user' %}
                            {{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
                            {% elif message['role'] == 'assistant' %}
                            {{ ' '  + content.strip() + ' ' + eos_token }}
                            {% endif %}
                            {% endfor %}"""

    tokenizer.add_tokens("[PAD]", special_tokens=True)
    tokenizer.pad_token = "[PAD]"
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.push_to_hub("pksx01/sarvam-1-it-bhojpuri",
                    use_auth_token=True,
                    commit_message="added tokenizer",
                    private=False)

    ds = ds.map(preprocess_function, remove_columns=ds["train"].column_names)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=["lm_head", "k_proj", "q_proj", "v_proj" "o_proj", "gate_proj", "down_proj", "up_proj"]
        #bias="none",
        #task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=ds['train'],
        args=transformers.TrainingArguments(
            num_train_epochs=1,
            save_total_limit=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_steps=10,
            weight_decay=0.0001,
            #max_steps=500,
            learning_rate=1e-5,
            #fp16=True,
            bf16=True,
            save_steps=50,
            logging_steps=50,
            output_dir="sarvam-1-it-bhojpuri",
            report_to="wandb"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer)
    )

    #model.config.use_cache = False #enable it for inference
    trainer.train()

    """model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained("outputs")"""

    """lora_config = LoraConfig.from_pretrained('outputs')
    model = get_peft_model(model, lora_config)"""


    generate_response("भारत के पहिला प्रधानमंत्री के रहे?")
    generate_response("स्वस्थ रहे खातिर तीन गो टिप्स दीं।")
    generate_response("हमनी के वायु प्रदूषण के कइसे कम कर सकेनी जा?")
    generate_response("october k mahine me bharat me kaha kaha ghumal thik rahi?")
    generate_response("हमार सवाल के जवाब भोजपुरी भाषा में दीं। इज़राइल देश के स्थापना कब भइल रहे और ओमे में केकर केकर महत्वपूर्ण भूमिका रहे?")

    run.finish()

    # Pushing fine-tuned model to HuggingFace Hub
    model.push_to_hub("pksx01/sarvam-1-it-bhojpuri",
                    use_auth_token=True,
                    commit_message="One complete epoch training",
                    private=False)