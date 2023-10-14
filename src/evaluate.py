from transformers import GPT2LMHeadModel, GPT2Tokenizer

def evaluate_model():
    # Load pretrained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("./output")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Encode text input and get model output
    input_text = "Hello, how are you?"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output = model(input_ids)
    
    # Your evaluation code here

if __name__ == "__main__":
    evaluate_model()
