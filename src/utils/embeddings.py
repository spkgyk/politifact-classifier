from tqdm import tqdm
import pandas as pd
import asyncio
import aiohttp
import os

openai_key = os.environ.get("OPENAI_KEY")


async def fetch_embeddings(session, texts, batch_id, model="text-embedding-3-small"):
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
    data = {"input": texts, "model": model}

    async with session.post(api_url, headers=headers, json=data) as response:
        response_data = await response.json()
        embeddings = [item["embedding"] for item in response_data["data"]]
        return {batch_id: embeddings}


async def get_all_embeddings(texts, model="text-embedding-3-small", batch_size=20):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_embeddings(session, texts[batch_id : batch_id + batch_size], batch_id, model)
            for batch_id in range(0, len(texts), batch_size)
        ]

        embeddings = {}
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            batch_embeddings = await task
            embeddings.update(batch_embeddings)

        sorted_embeddings = [embeddings[i] for i in sorted(embeddings.keys())]
        flattened_embeddings = [embedding for batch in sorted_embeddings for embedding in batch]

        return flattened_embeddings


def apply_async(df, func, model, batch_size):
    loop = asyncio.new_event_loop()
    embeddings = loop.run_until_complete(func(df.tolist(), model, batch_size))
    return embeddings


if __name__ == "__main__":
    raw_data = pd.read_csv("../data/data.csv")
    batch_size = 50

    raw_data["statement_embedding"] = apply_async(raw_data["statement"], get_all_embeddings, "text-embedding-3-small", batch_size)

    raw_data.to_csv("../data/data_ada.csv", index=False)
