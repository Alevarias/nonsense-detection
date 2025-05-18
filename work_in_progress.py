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

DATA_FOLDER = "testing_data"
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000
retriever = SentenceTransformer("all-mpnet-base-v2")
RST_BTWN_FILES = True

CHECKPOINT_ROOT = "D:/models2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 32

def get_text(el):
        if hasattr(el, 'text'):
            return el.text
        if isinstance(el, (list, tuple)):
            return ' '.join([t.text for t in el])
        return str(el)

def main():

    triples_path = 'triples.pkl'
    kg_index_path = 'triple_embeddings.faiss'
    premise_np = 'premise_sentences.npy'
    premise_idx = 'premise_index.faiss'
    top_k_facts = 3
    top_k_premises = 3

    # Wipe previous stored files
    for path in [triples_path, kg_index_path, premise_np, premise_idx]:
        if os.path.exists(path):
            os.remove(path)

    # Create the knowledge graph and premise list
    dim = retriever.get_sentence_embedding_dimension()
    facts_list = []
    kg_index = faiss.IndexFlatIP(dim)
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
            # Retrieve facts
            if facts_list:
                hyp_emb = retriever.encode([hypothesis], convert_to_numpy=True, show_progress_bar=False)
                faiss.normalize_L2(hyp_emb)
                Df, If = kg_index.search(hyp_emb.astype('float32'), top_k_facts)
                facts = [facts_list[i] for i in If[0]]
            # Retrieve the facts from the knowledge graph

            else: 
                facts = []
                # print("No facts found, skipping to next sentence")

            if premise_list:
                hyp_emb = retriever.encode([hypothesis], convert_to_numpy=True, show_progress_bar=False)
                faiss.normalize_L2(hyp_emb)
                Dp, Ip = premise_index.search(hyp_emb.astype('float32'), top_k_premises)
                premises = [premise_list[i] for i in Ip[0]]

            else: 
                premises = []
                # print("No premises found, skipping to next sentence")

            fact_str = " ; ".join(facts)
            premise_str = " ; ".join(premises)
            input_text = f"mnli premise: {premise_str} ; {fact_str} hypothesis: {hypothesis}"
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
                print("Contradiction found between:")
                print("Premise:    ", premise_str)
                print("Rules:     ", fact_str)
                print("Hypothesis: ", hypothesis)
            else:
                print("Label: ", pred_label)
                print("Premise:    ", premise_str)
                print("Rules:     ", fact_str)
                print("Hypothesis: ", hypothesis)

            # Logs facts if there is no contradiction
            if pred_label != 'contradiction':
                ie_doc = nlp(hypothesis)
                triples = list(textacy.extract.subject_verb_object_triples(ie_doc))
                for subj, verb, obj in triples:
                    subj_text = get_text(subj).strip()
                    obj_text  = get_text(obj).strip()
                    # handle verb lemma extraction
                    if hasattr(verb, 'lemma_'):
                        rel = verb.lemma_.lower()
                    elif isinstance(verb, (list, tuple)):
                        rel = ' '.join([t.lemma_ for t in verb]).lower()
                    else:
                        rel = get_text(verb).lower()

                    if any(kw in rel for kw in ["require", "has", "is", "are", "cannot", "needs"]):
                        triple_text = f"{subj_text} {rel} {obj_text}"
                        emb = retriever.encode([triple_text], convert_to_numpy=True)
                        faiss.normalize_L2(emb)
                        kg_index.add(emb.astype('float32'))
                        facts_list.append(triple_text)

            # Logs premises
            emb = retriever.encode([hypothesis], convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(emb)
            premise_index.add(emb.astype('float32'))
            premise_list.append(hypothesis)

        # Saves premise list and knowledge graph in files
        with open(triples_path, 'wb') as f:
            pickle.dump(facts_list, f)
        faiss.write_index(kg_index, kg_index_path)
        np.save(premise_np, np.array(premise_list, dtype=object))
        faiss.write_index(premise_index, premise_idx)

        print("Knowledge graph and premise list saved.")

        # embeddings = embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
        # faiss.normalize_L2(embeddings)

        # dim = embeddings.shape[1]
        # index = faiss.IndexFlatIP(dim)
        # index.add(embeddings)

        # faiss.write_index(index, 'premise_index.faiss')
        # np.save('premise_sentences.npy', np.array(sentences, dtype=object))
        # index = faiss.read_index('premise_index.faiss')
        # sentences = np.load('premise_sentences.npy', allow_pickle=True).tolist()

        # embedder = SentenceTransformer("all-mpnet-base-v2")
        # hypo_embedding = embedder.encode([hypothesis], convert_to_numpy=True, show_progress_bar=True)
        # faiss.normalize_L2(hypo_embedding)

        # # Gets the top 3 most similar previous sentences
        # D, I = index.search(hypo_embedding, top_k=3)
        # top_premises = [sentences[i] for i in I[0]]

        # # Gets the top 3 most similar facts
        # Df, If = kg_index.search(hypo_embedding, top_k=3)
        # top_facts = [facts_list[i] for i in If[0]]

        # facts = retrieve_facts(hypothesis, ...)

        # premise_content = " ".join(top_premises)


        # # Looks at each previous related sentence in the text and compares it to the current sentence
        # # to see if they contradict each other
        # for premise in premise_list:
        #     # Actual imput
        #     text_in = f"mnli premise: {premise} hypothesis: {hypothesis}"
        #     inputs = tokenizer(
        #         text_in,
        #         max_length=256,
        #         truncation=True,
        #         padding=False,
        #         return_tensors="pt"
        #     ).to(DEVICE)

        #     with torch.no_grad():
        #         pred_ids = model.generate(
        #             inputs.input_ids,
        #             attention_mask=inputs.attention_mask,
        #             max_length=MAX_TARGET_LEN, 
        #             num_beams=1
        #         )

        #     pred_label = tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

        #     if pred_label == "contradiction":
        #         print("Contradiction found between:")
        #         print("Premise:    ", premise)
        #         print("Hypothesis: ", hypothesis)

    








if __name__ == "__main__":
    main()





















