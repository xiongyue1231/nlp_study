import numpy as np
from llm_cache.EmbeddingsCache import EmbeddingsCache
from llm_cache.SemanticCache import SemanticCache
from llm_cache.SemanticMessageHistory import SemanticMessageHistory
from llm_cache.SemanticRouter import SemanticRouter, Route

def get_random_embedding(text):
    """生成随机embedding用于演示"""
    if isinstance(text, str):
        text = [text]
    rng = np.random.default_rng(seed=42)
    return np.array([rng.normal(size=768) for t in text])

def demo_embeddings_cache():
    """演示 EmbeddingsCache"""
    print("=" * 50)
    print("1. EmbeddingsCache 演示")
    print("=" * 50)
    
    embed_cache = EmbeddingsCache(
        name="demo_embedding_cache",
        ttl=360,
        redis_url="localhost"
    )
    
    text = "这是一段测试文本"
    embedding = get_random_embedding(text)
    
    print(f"\n存储 embedding: {text}")
    embed_cache.store(text=text, embedding=embedding)
    
    print(f"检索 embedding:")
    result = embed_cache.call(text=text)
    print(f"检索结果 shape: {result[0].shape}")
    
    print(f"删除 embedding")
    embed_cache.delete(text=text)
    print("EmbeddingsCache 演示完成!\n")

def demo_semantic_cache():
    """演示 SemanticCache"""
    print("=" * 50)
    print("2. SemanticCache 演示")
    print("=" * 50)
    
    semantic_cache = SemanticCache(
        name="demo_semantic_cache",
        embedding_method=get_random_embedding,
        ttl=360,
        redis_url="localhost",
        distance_threshold=1000
    )
    
    semantic_cache.clear_cache()
    
    print("\n存储问题和回答:")
    print("问题1: 北京的首都是什么?")
    print("回答1: 北京")
    print("问题2: 法国的首都是哪里?")
    print("回答2: 巴黎")
    
    semantic_cache.store(
        prompt=["北京的首都是什么?", "法国的首都是哪里?"],
        response=["北京", "巴黎"]
    )
    
    print("\n检索相似问题: '法国的首都是哪里?'")
    result = semantic_cache.check(prompt="法国的首都是哪里?")
    if result:
        for item in result:
            print(f"找到回答: {item['response']}, 距离: {item['distance']:.4f}")
    
    print("\nSemanticCache 演示完成!\n")

def demo_semantic_message_history():
    """演示 SemanticMessageHistory"""
    print("=" * 50)
    print("3. SemanticMessageHistory 演示")
    print("=" * 50)
    
    history = SemanticMessageHistory(
        name="demo-session",
        redis_url="localhost"
    )
    history.clear_history()
    
    print("\n添加对话历史:")
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "What's the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
    ]
    history.add_messages(messages)
    
    print("\n获取所有历史:")
    all_history = history.get_history()
    for msg in all_history:
        print(f"{msg['role']}: {msg['content']}")
    
    print("\n获取最近1条记录:")
    recent = history.get_recent(top_k=1)
    print(f"{recent[0]['role']}: {recent[0]['content']}")
    
    print("\n获取所有user记录:")
    user_messages = history.get_recent(role="user")
    for msg in user_messages:
        print(f"{msg['role']}: {msg['content']}")
    
    print("\n查找与'weather'相关的记录:")
    relevant = history.get_relevant("weather")
    for msg in relevant:
        print(f"{msg['role']}: {msg['content']}")
    
    print("\nSemanticMessageHistory 演示完成!\n")

def demo_semantic_router():
    """演示 SemanticRouter"""
    print("=" * 50)
    print("4. SemanticRouter 演示")
    print("=" * 50)
    
    routes = [
        Route(
            name="greeting",
            references=["Hi, good morning", "Hi, good afternoon", "hello"],
            metadata={"type": "greeting"},
            distance_threshold=1000,
        ),
        Route(
            name="refund",
            references=["如何退货", "退款流程", "我想退款"],
            metadata={"type": "refund"},
            distance_threshold=1000,
        ),
        Route(
            name="shipping",
            references=["什么时候发货", "物流信息"],
            metadata={"type": "shipping"},
            distance_threshold=1000,
        ),
    ]
    
    router = SemanticRouter(
        name="demo-topic-router",
        embedding_method=get_random_embedding,
        routes=routes,
        redis_url="localhost"
    )
    
    test_queries = [
        "Hi, good morning",
        "我想退货",
        "我的快递什么时候到",
    ]
    
    print("\n测试语义路由:")
    for query in test_queries:
        result = router(query)
        if result:
            print(f"问题: '{query}' -> 路由到: {result['name']}, 距离: {result['distance']:.4f}")
        else:
            print(f"问题: '{query}' -> 未找到匹配路由")
    
    router.clear()
    print("\nSemanticRouter 演示完成!\n")

def main():
    print("\n" + "=" * 50)
    print("LLM 智能缓存系统演示")
    print("=" * 50 + "\n")
    
    demo_embeddings_cache()
    demo_semantic_cache()
    demo_semantic_message_history()
    demo_semantic_router()
    
    print("=" * 50)
    print("所有演示完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
