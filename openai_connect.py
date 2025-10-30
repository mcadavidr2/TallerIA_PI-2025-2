from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# Carga las variables de entorno desde el archivo correcto
env_path = Path(_file_).parent / 'api_keys.env'
load_dotenv(env_path)

# Inicializa el cliente de OpenAI con la API Key
gpt_client = OpenAI(api_key=os.environ.get('openai_apikey'))

# Ejemplo de prueba (puedes comentar esto si no lo necesitas)
# response = gpt_client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "Hola, ¿quién eres?"}]
# )
# print(response.choices[0].message.content)

def get_completion(prompt, model="gpt-3.5-turbo"):
    """
    Envía un prompt a la API de OpenAI y devuelve la respuesta generada.
    """
    messages = [{"role": "user", "content": prompt}]
    
    response = gpt_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0  # 0 = más preciso, menos creativo
    )

    return response.choices[0].message.content.strip()