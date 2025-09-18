import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv

class Command(BaseCommand):
    # Calcular la similitud entre un prompt y las películas
    prompt = "película sobre la Segunda Guerra Mundial"
    prompt_emb = get_embedding(prompt)
    sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
    sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)
    self.stdout.write(f"📄 Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
    self.stdout.write(f"📄 Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")
    help = "Generate and store embeddings for all movies in the database"

    def handle(self, *args, **kwargs):
        # ✅ Load OpenAI API key
        load_dotenv('../openAI.env')
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        # ✅ Fetch all movies from the database
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies in the database")

        def get_embedding(text):
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Ejemplo: calcular similitud entre dos películas y un prompt
        try:
            movie1 = Movie.objects.get(title="La lista de Schindler")
            movie2 = Movie.objects.get(title="El club de la pelea")
            emb1 = get_embedding(movie1.description)
            emb2 = get_embedding(movie2.description)
            similarity = cosine_similarity(emb1, emb2)
            self.stdout.write(f"🎬 {movie1.title} vs {movie2.title}: {similarity:.4f}")

            # Calcular la similitud entre un prompt y las películas
            prompt = "película sobre la Segunda Guerra Mundial"
            prompt_emb = get_embedding(prompt)
            sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
            sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)
            self.stdout.write(f"📄 Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
            self.stdout.write(f"📄 Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")
        except Exception as e:
            self.stderr.write(f"❌ Error calculating similarity: {e}")
