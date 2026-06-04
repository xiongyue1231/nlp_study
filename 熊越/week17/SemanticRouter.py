import os
import numpy as np
import redis
import faiss
import uuid
from typing import Optional, List, Union, Any, Dict, Callable

class Route:
    def __init__(
        self,
        name: str,
        references: List[str],
        metadata: Optional[Dict] = None,
        distance_threshold: float = 0.3
    ):
        self.name = name
        self.references = references
        self.metadata = metadata or {}
        self.distance_threshold = distance_threshold

class SemanticRouter:
    def __init__(
        self,
        name: str,
        embedding_method: Callable[[Union[str, List[str]]], Any],
        routes: Optional[List[Route]] = None,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None
    ):
        self.name = name
        self.embedding_method = embedding_method
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        self.index_key = f"{self.name}:router_index"
        self.route_info_key = f"{self.name}:route_info"
        
        if os.path.exists(f"{self.name}.router.index"):
            self.index = faiss.read_index(f"{self.name}.router.index")
            self.route_ids = self.redis.lrange(self.index_key, 0, -1)
        else:
            self.index = None
            self.route_ids = []
        
        if routes:
            for route in routes:
                self.add_route(route.references, route.name, route.metadata, route.distance_threshold)

    def add_route(
        self,
        questions: List[str],
        target: str,
        metadata: Optional[Dict] = None,
        distance_threshold: float = 0.3
    ):
        if not questions:
            return
            
        embeddings = self.embedding_method(questions)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            
        for i, q in enumerate(questions):
            route_id = str(uuid.uuid4())
            self.index.add(embeddings[i:i+1])
            
            route_info = {
                "id": route_id,
                "target": target,
                "reference": q,
                "metadata": metadata or {},
                "distance_threshold": distance_threshold
            }
            
            self.redis.hset(self.route_info_key, route_id, str(route_info))
            self.redis.rpush(self.index_key, route_id)
            
        faiss.write_index(self.index, f"{self.name}.router.index")
        self.route_ids = self.redis.lrange(self.index_key, 0, -1)

    def route(self, question: str) -> Optional[Dict]:
        if self.index is None or not self.route_ids:
            return None
            
        embedding = self.embedding_method(question)
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
            
        dis, ind = self.index.search(embedding, k=5)
        
        best_route = None
        min_distance = float('inf')
        
        for i, (distance, idx) in enumerate(zip(dis[0], ind[0])):
            if idx < len(self.route_ids):
                route_id = self.route_ids[idx]
                route_info_str = self.redis.hget(self.route_info_key, route_id)
                
                if route_info_str:
                    route_info = eval(route_info_str)
                    if distance < min_distance and distance < route_info["distance_threshold"]:
                        min_distance = distance
                        best_route = {
                            "name": route_info["target"],
                            "distance": float(distance),
                            "metadata": route_info["metadata"]
                        }
        
        return best_route

    def __call__(self, question: str) -> Optional[Dict]:
        return self.route(question)

    def clear(self):
        self.redis.delete(self.route_info_key, self.index_key)
        if os.path.exists(f"{self.name}.router.index"):
            os.unlink(f"{self.name}.router.index")
        self.index = None
        self.route_ids = []

if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        rng = np.random.default_rng(seed=42)
        return np.array([rng.normal(size=768) for t in text])

    routes = [
        Route(
            name="greeting",
            references=["Hi, good morning", "Hi, good afternoon"],
            metadata={"type": "greeting"},
            distance_threshold=1000,
        ),
        Route(
            name="refund",
            references=["如何退货", "退款流程"],
            metadata={"type": "refund"},
            distance_threshold=1000,
        ),
    ]

    router = SemanticRouter(
        name="topic-router",
        embedding_method=get_embedding,
        routes=routes,
        redis_url="localhost"
    )

    print("Route for 'Hi, good morning':", router("Hi, good morning"))
    print("Route for '如何退货':", router("如何退货"))
    router.clear()
