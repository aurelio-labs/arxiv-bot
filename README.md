# Arxiv Bot
Usage:
```python
from arxiv_bot.manager import bots
arx = bots.Arxiver(index_name='langchain2', pinecone_api_key='<YOUR_PINECONE_API_KEY', pinecone_environment='<YOUR_PINECONE_ENVIRONMENT>', openai_api_key='<YOUR_OPENAI_API_KEY>')
## add article by arxiv id
arx('Please add the following article from ArXiv to the database: 2305.00606')
## query database
arx('What is latent diffusion?')
```