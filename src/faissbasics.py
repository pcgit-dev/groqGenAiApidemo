from langchain_cohere import CohereEmbeddings
import numpy as np
from dotenv import load_dotenv
import os
import faiss

load_dotenv()
# Create the embeddings model - Dimension = 384
# Cohere model in use
# model_name = "embed-english-light-v3.0"

# Create the embeddings model - Dimension = 1024
os.environ['COHERE_API_KEY']=os.getenv("COHERE_API_KEY")
model_name = "embed-english-v3.0"
embeddings_model = CohereEmbeddings(model=model_name)

corpus = [
  "A man is eating food.", "A man is eating a piece of bread.",
  "The chef is preparing a delicious meal in the kitchen.", "A chef is tossing vegetables in a sizzling pan.",
  "A man is riding a horse.", "A man is riding a white horse on an enclosed ground.",
  "A woman is playing violin.", "A musician is tuning his guitar before the concert.",
  "The girl is carrying a baby.", "The baby is giggling while playing with her toys.",
  "The family is having a picnic under the shady oak tree.", "A group of friends is hiking up the mountain trail.",
  "The mechanic is repairing a broken-down car in the garage.", "The old man is feeding breadcrumbs to the ducks at the pond.",
  "The artist is sketching a beautiful landscape at sunset.", "A man is painting a colorful mural on the city wall.",
  "A team of scientists is conducting experiments in the laboratory.", "A group of students is studying together in the library.",
  "The birds are chirping happily in the morning sun.", "The dog is chasing its tail around the backyard.",
  "A group of children are playing soccer in the park.", "A monkey is playing drums.",
  "A boy is flying a kite in the open field.", "Two men pushed carts through the woods.",
  "A woman is walking her dog along the beach.", "A young girl is reading a book under a shady tree.",
  "The dancer is gracefully performing on stage.", "The farmer is harvesting ripe tomatoes from the vine."
]

# A list of embeddings
corpus_embeddings = embeddings_model.embed_documents(corpus);

embedding_dimension = len(corpus_embeddings[0])

#Convert List of embedding to numpy , as numpy can store n dim arrays
corpus_embeddings_numpy = np.array(corpus_embeddings).astype(np.float32)

#IndexFlatL2 measures the L2 (or Euclidean) distance between all given points between our query vector, and the vectors loaded into the index. 
#It's simple, very accurate, but not too fast. L2 distance calculation between a query vector xq and our indexed vectors (shown as y)

# Create index 
index_flatl2 = faiss.IndexFlatL2(embedding_dimension)

# Is trained
print("Is trained : ",index_flatl2.is_trained)

# Add embeddings
index_flatl2.add(corpus_embeddings_numpy)

print("Size of index : ",index_flatl2.ntotal)

#Query index.search Takes as parameter an ndarray with embedding Returns the Distance:ndarray, Indexes:ndarray
test_docs = [
    'I am a foodie',
    'My siter loves to play string instruments',
    'Musical instruments'
]
embed_query = embeddings_model.embed_documents(test_docs)

distances , index = index_flatl2.search(np.array(embed_query),3)
k=3

distances, indexes = index_flatl2.search(np.array(embed_query), k)

print("Distances : ", distances)
print("Indexes : ", indexes)

print("------")
for i, corpus_index in enumerate(indexes[0]):
    print(corpus[corpus_index],"  (",  distances[0][i],")")