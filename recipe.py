from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or "gpt2-large" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Recipe generation function
def generate_recipe(ingredients, max_length=150):
    # Convert ingredients list to a string prompt
    ingredients_text = ', '.join(ingredients)
    prompt = f"Ingredients: {ingredients_text}\nRecipe:\n"

    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate recipe using the model
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.7,  # Creativity control
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    # Decode the generated text
    recipe = tokenizer.decode(output[0], skip_special_tokens=True)
    return recipe

# Example usage
ingredients = ["chicken", "garlic", "olive oil", "salt", "pepper", "rosemary"]
recipe = generate_recipe(ingredients)
print(recipe)
