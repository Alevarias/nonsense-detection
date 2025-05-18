import transformers
import pandas as pd
import spacy
from pathlib import Path
import os
import torch
import glob
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import faiss
import pickle
import textacy.extract
print("Running with transformers v", transformers.__version__)

DATA_FOLDER = "demo_data"
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000
retriever = SentenceTransformer("all-mpnet-base-v2")
RST_BTWN_FILES = True

CHECKPOINT_ROOT = "trained_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 32

def main():

    premise_np = 'premise_sentences.npy'
    premise_idx = 'premise_index.faiss'
    top_k_facts = 3
    top_k_premises = 3

    # Wipe previous stored files
    for path in [premise_np, premise_idx]:
        if os.path.exists(path):
            os.remove(path)

    # Create the knowledge graph and premise list
    dim = retriever.get_sentence_embedding_dimension()
    premise_list = []
    premise_index = faiss.IndexFlatIP(dim)


    print("Loading model")
    # Find the latest checkpoint of T5 Model
    checkpoint_dirs = glob.glob(os.path.join(CHECKPOINT_ROOT, 'checkpoint-*'))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_ROOT}, please check the path")
    latest_ckpt = sorted(
        checkpoint_dirs,
        key=lambda x: int(x.rsplit('-', 1)[-1])
    )[-1]
    print(f"Loading model from latest checkpoint: {latest_ckpt} on {DEVICE}...")

    # Load the T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(latest_ckpt)
    model = T5ForConditionalGeneration.from_pretrained(latest_ckpt)
    model.to(DEVICE)
    model.eval()

    # Iterate through all files in the testing_data folder
    for filename in os.listdir(DATA_FOLDER):
        # If the file is a .txt file, we read it sentence by sentence using spacy
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_FOLDER, filename), "r", encoding="utf-8") as file:
                text = file.read()
            text = text.replace("\n", " ")
            
            # Split the text into sentences
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            # print(sentences)

        # For now, if the file isn't a .txt file, we skip it
        else:
            print(f"Skipping non-txt file: {filename}")
            continue
        
        # Iterating through the sentences in the text
        for idx, hypothesis in enumerate(sentences):

            if premise_list:
                hyp_emb = retriever.encode([hypothesis], convert_to_numpy=True, show_progress_bar=False)
                faiss.normalize_L2(hyp_emb)

                top_k_premises = min(top_k_premises, premise_index.ntotal)
                Dp, Ip = premise_index.search(hyp_emb.astype('float32'), top_k_premises)

                premises = [premise_list[i] for i in Ip[0]]

            else: 
                premises = []

            if len(premises) != 0:

                for premise in premises:

                    input_text = f"mnli premise: {premise} ; hypothesis: {hypothesis}"
                    inputs = tokenizer(
                        input_text,
                        max_length=256,
                        truncation=True,
                        padding=False,
                        return_tensors="pt"
                    ).to(DEVICE)

                    with torch.no_grad():
                        pred_ids = model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_length=MAX_TARGET_LEN, 
                            num_beams=1
                        )
                    
                    pred_label = tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

                    # Print out the results
                    if pred_label == "contradiction":
                        print("####################################################################################")
                        print("Contradiction found between:")
                        print("Premise:    ", premise)
                        print("Hypothesis: ", hypothesis)
                    else:
                        print("####################################################################################")
                        print("Label: ", pred_label)
                        print("Premise:    ", premise)
                        print("Hypothesis: ", hypothesis)

            # Logs premises
            emb = retriever.encode([hypothesis], convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(emb)
            premise_index.add(emb.astype('float32'))
            premise_list.append(hypothesis)


        np.save(premise_np, np.array(premise_list, dtype=object))
        faiss.write_index(premise_index, premise_idx)

        print("Premise list saved.")

if __name__ == "__main__":
    main()
