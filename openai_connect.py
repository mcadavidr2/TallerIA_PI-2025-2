from openai import OpenAI
import os
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv('.env')

# Inicializa el cliente de OpenAI con la API Key
gpt_client = OpenAI(api_key=os.environ.get('openai_apikey'))

# Ejemplo: obtener información de la API
# response = gpt_client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "Hola, ¿quién eres?"}]
# )
# print(response.choices[0].message.content)


def get_completion(prompt, model="gpt-3.5-turbo"):
	"""
	Recibe un prompt, consulta la API de OpenAI y retorna solo el texto generado.
	"""
	messages = [{"role": "user", "content": prompt}]
	response = gpt_client.chat.completions.create(
		model=model,
		messages=messages,
		temperature=0
	)
	return response.choices[0].message.content.strip()
