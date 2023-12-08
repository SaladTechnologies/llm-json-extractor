import email
from email import policy
import nltk
from nltk.tokenize import word_tokenize
from typing import Union, IO, Any, List, Tuple
from bs4 import BeautifulSoup
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from uuid import uuid4
import aiohttp
import asyncio
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
import json

nltk.download("punkt")
env = Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape())

chat_template = env.get_template("chat.jinja")

embedding_server = os.getenv("TEI_SERVER", "http://localhost:3001")
generation_server = os.getenv("TGI_SERVER", "http://localhost:3000")
qdrant_server = os.getenv("QDRANT_SERVER", "http://localhost:6333")

qdrant_client = QdrantClient(url=qdrant_server)


def extract_html_from_mhtml(file_path: Union[str, IO[Any]]) -> str:
    if isinstance(file_path, str):
        with open(file_path, "rb") as file:
            # Parse the .mhtml file as a MIME message
            message = email.message_from_binary_file(file, policy=policy.default)
    else:
        message = email.message_from_binary_file(file_path, policy=policy.default)

    # Iterate through the message parts
    for part in message.walk():
        # Check if the part is an HTML document
        if part.get_content_type() == "text/html":
            # Return the HTML content
            return part.get_content()


def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    # Tokenize the text
    tokens = word_tokenize(text)

    # Chunk the tokens
    chunks = []
    current_chunk = []
    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")

    meta = soup.find_all("meta")
    scripts = soup.find_all("script")

    json_content = [
        tag.get_text() for tag in scripts if tag.get("type") and "json" in tag["type"]
    ]

    text = soup.get_text(separator="\n", strip=True)

    return json_content + [str(tag) for tag in meta] + chunk_text(text, max_tokens=256)


async def get_embedding(session: aiohttp.ClientSession, text: str) -> List[List[float]]:
    payload = {"inputs": text, "normalize": True, "truncate": False}
    async with session.post(f"{embedding_server}/embed", json=payload) as response:
        return await response.json()


async def get_all_embeddings(chunks: List[str]) -> List[List[List[float]]]:
    async with aiohttp.ClientSession() as session:
        tasks = [get_embedding(session, chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)


async def get_embedding_from_mhtml(
    file_path: Union[str, IO[Any]]
) -> Tuple[List[List[float]], List[str]]:
    html = extract_html_from_mhtml(file_path)
    chunks = chunk_html(html)
    embeddings = await get_all_embeddings(chunks)
    return embeddings, chunks


def index_for_search(
    collection_name: str,
    chunks: List[str],
    embeddings: List[List[List[float]]],
    page_id: str,
):
    points = [
        PointStruct(
            **{
                "id": str(uuid4()),
                "vector": embedding[0],
                "payload": {"content": str(chunks[i]), "page_id": page_id},
            }
        )
        for i, embedding in enumerate(embeddings)
    ]

    qdrant_client.upsert(collection_name=collection_name, points=points)


def search_page(query: str, collection_name: str, page_id: str, limit: int = 5):
    query_embedding = requests.post(
        f"{embedding_server}/embed",
        json={"inputs": query, "normalize": True, "truncate": False},
    ).json()[0]

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
        ),
        limit=limit,
    )

    return search_result


def get_extraction_prompt(prompt: str, context: str):
    chat = [
        {
            "role": "system",
            "content": "extract the most relevant product details from web snippets for use by downstream applications. Always answer with ONLY a JSON object that has ONLY a single key. Never include any other information in your answer. Ensure that the content of the answer makes sense in the context of the question.",
        },
        {"role": "user", "content": prompt + "\n\n" + context},
    ]

    prompt = chat_template.render(
        messages=chat, add_generation_prompt=True, eos_token="<\s>"
    )

    return prompt


def get_normalize_prompt(prompt: str):
    chat = [
        {
            "role": "system",
            "content": "Rewrite the content as prose. Include all of the details present. Use a neutral tone, professional tone, and speak only about the product.",
        },
        {"role": "user", "content": prompt},
    ]

    prompt = chat_template.render(
        messages=chat, add_generation_prompt=True, eos_token="<\s>"
    )

    return prompt


def generate(prompt: str, generate_params: dict = {}):
    payload = {"inputs": prompt, "parameters": generate_params}
    response = requests.post(f"{generation_server}/generate", json=payload).json()
    if not "generated_text" in response:
        print(response)
    return response["generated_text"]


def ask_question(
    question: str,
    page_id: str,
    limit: int = 5,
    max_tokens: int = 256,
    search=None,
    collection_name="harrods",
):
    if search is None:
        search = question
    search_results = search_page(search, collection_name, page_id, limit=limit)
    if len(search_results) == 0:
        return ""
    searched_content = [
        search_result.payload["content"] for search_result in search_results
    ]
    answer = "\n".join(searched_content)
    prompt = get_extraction_prompt(question, answer)
    answer = generate(
        prompt,
        generate_params={
            "best_of": 1,
            "stop": ["}"],
            "temperature": 0.1,
            "max_new_tokens": max_tokens,
        },
    )
    answer = answer.strip()
    try:
        answer = json.loads(answer)
        if len(answer.keys()) == 1:
            return list(answer.values())[0]
        elif len(answer.keys()) > 1:
            prompt = get_normalize_prompt(json.dumps(answer))
            answer = generate(
                prompt,
                generate_params={
                    "best_of": 1,
                    "temperature": 0.2,
                    "max_new_tokens": max_tokens,
                },
            )
            return answer.strip()
    except Exception as e:
        prompt = get_normalize_prompt(answer)
        answer = generate(
            prompt,
            generate_params={
                "best_of": 1,
                "temperature": 0.2,
                "max_new_tokens": max_tokens,
            },
        )
        return answer.strip()
    return answer


def extract_from_page(page_id: str):
    name = ask_question(
        "What is the name of the product?",
        page_id,
        # search="product name or model",
        max_tokens=64,
    )
    price = ask_question(f"What is the price of {name}", page_id, max_tokens=16)
    currency = ask_question(
        f"What currency is the price of {name} ({price})", page_id, max_tokens=16
    )
    description = ask_question(
        f"Description of {name}, or features", page_id, limit=7, max_tokens=512
    )
    is_available_online = ask_question(
        f"Is {name} available to purchase online? Answer only yes or no.",
        page_id,
        max_tokens=16,
    )
    manufacturer = ask_question(
        f"Who is the manufacturer (or manufacturers) of {name}? Answer as a string, even if there's multiple",
        page_id,
        # search="manufacturer",
        max_tokens=64,
    )

    return {
        "name": name,
        "manufacturer": manufacturer,
        "price": price,
        "currency": currency,
        "description": description,
        "is_available_online": is_available_online,
    }
