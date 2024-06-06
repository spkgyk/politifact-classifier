from tqdm import tqdm
import pandas as pd
import asyncio
import aiohttp
import os

openai_key = os.environ.get("OPENAI_KEY")


async def fetch_embeddings(session, texts, model="text-embedding-3-small"):
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
    data = {"input": texts, "model": model}
    async with session.post(api_url, headers=headers, json=data) as response:
        response_data = await response.json()
        return [item["embedding"] for item in response_data["data"]]


async def get_all_embeddings(texts, model="text-embedding-3-small", batch_size=20):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tasks.append(fetch_embeddings(session, batch, model))

        embeddings = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            embeddings_batch = await task
            embeddings.extend(embeddings_batch)

        return embeddings


def apply_async(df, func, model, batch_size):
    loop = asyncio.get_event_loop()
    embeddings = loop.run_until_complete(func(df.tolist(), model, batch_size))
    return embeddings


if __name__ == "__main__":
    raw_data = pd.read_csv("../data/data.csv")
    batch_size = 2
    raw_data["ada_embedding"] = apply_async(raw_data["statement"], get_all_embeddings, "text-embedding-3-small", batch_size)
    raw_data.to_csv("../data/data_ada.csv")
