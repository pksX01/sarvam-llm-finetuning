from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class RequestBody(BaseModel):
    query: str


app = FastAPI()

@app.on_event("startup")
async def on_start():
    global model, tokenizer, peft_model

    model_id = "pksx01/sarvam-1-it-bhojpuri"
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "sarvamai/sarvam-1",
        torch_dtype=torch.bfloat16,  # BF16 is supported on M1 CPUs for faster calculations
        #load_in_8bit=True,
        device_map="cpu"
    )

    model.resize_token_embeddings(len(tokenizer))

    # Load the PEFT model
    peft_model = PeftModel.from_pretrained(
        model,
        model_id,
        is_trainable=False
    )


@app.post('/generate')
async def get_response(request: RequestBody):

    message = [{"role": "user", "content": request.query}]
    model_ip = tokenizer.apply_chat_template(message, tokenize=False)
    tokenized_ip = tokenizer(model_ip, return_tensors="pt")

    peft_model.eval()
    with torch.no_grad():
        op_tokens = peft_model.generate(
            **tokenized_ip,
            max_new_tokens=250,
            temperature=0.01,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(op_tokens[0], skip_special_tokens=True).strip()
    