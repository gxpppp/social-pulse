---
name: "project-developer"
description: "Full-stack project development skill with parallel sub-agents and web search for optimal solutions. Invoke when starting new project development or implementing complex features."
---

# Project Developer Skill

A comprehensive development workflow skill for building full-stack applications efficiently.

## When to Invoke

- Starting a new project from scratch
- Implementing complex multi-module features
- Need to research best practices and technologies
- Breaking down large tasks into parallel work streams

## Core Principles

### 1. Parallel Sub-Agent Development

**Always use parallel sub-agents for independent tasks:**

```
Task A ──────────────────────►
Task B ──────────────────────►  → Integration
Task C ──────────────────────►
Task D ──────────────────────►
```

**Benefits:**
- Reduced development time
- Better resource utilization
- Independent module testing
- Cleaner code separation

**Example - Multi-Platform Crawler Development:**
```python
# Instead of sequential development:
# Task 1: Twitter crawler → Task 2: Weibo crawler → Task 3: Reddit crawler

# Use parallel sub-agents:
agents = [
    TaskAgent("twitter-crawler", implement_twitter_crawler),
    TaskAgent("weibo-crawler", implement_weibo_crawler),
    TaskAgent("reddit-crawler", implement_reddit_crawler),
]
results = await asyncio.gather(*[agent.run() for agent in agents])
```

### 2. Technology Research Workflow

**When encountering technical decisions:**

1. **Identify the problem domain**
   - Data storage? → Research database options
   - Performance? → Research optimization techniques
   - Integration? → Research API patterns

2. **Web search for best practices**
   ```
   Search queries:
   - "best practices for [technology] [year]"
   - "[framework] vs [alternative] comparison"
   - "[technology] performance optimization"
   ```

3. **Evaluate options based on:**
   - Project requirements (scale, budget, timeline)
   - Team expertise
   - Community support
   - Long-term maintainability

### 3. Architecture Decision Framework

| Decision Point | Research Focus | Key Considerations |
|---------------|----------------|-------------------|
| **Database** | Compare SQL vs NoSQL vs Graph | Data relationships, query patterns, scale |
| **API Design** | REST vs GraphQL vs gRPC | Client needs, real-time requirements |
| **Deployment** | Docker vs Serverless vs VM | Cost, scalability, complexity |
| **ML Framework** | PyTorch vs TensorFlow vs sklearn | Model complexity, GPU needs |

## Development Workflow

### Phase 1: Planning & Research

```
┌─────────────────────────────────────────┐
│  1. Define requirements                  │
│  2. Research technologies (web search)   │
│  3. Design architecture                   │
│  4. Create task breakdown                │
│  5. Identify parallelizable tasks        │
└─────────────────────────────────────────┘
```

### Phase 2: Parallel Implementation

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Sub-Agent 1 │  │  Sub-Agent 2 │  │  Sub-Agent 3 │
│  (Module A)  │  │  (Module B)  │  │  (Module C)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
              ┌──────────────────┐
              │  Integration     │
              │  & Testing       │
              └──────────────────┘
```

### Phase 3: Integration & Testing

- Merge parallel development streams
- Run integration tests
- Performance benchmarking
- Documentation update

## Technology Stack Selection Guide

### Data Collection Layer
| Need | Recommended | Alternative |
|------|-------------|-------------|
| High concurrency | Playwright + asyncio | Scrapy |
| Anti-detection | Residential proxies | Browser fingerprinting |
| Rate limiting | Token bucket | Sliding window |

### Data Storage Layer
| Need | Recommended | Alternative |
|------|-------------|-------------|
| Relational data | SQLite/PostgreSQL | MySQL |
| Graph data | NetworkX/Neo4j | ArangoDB |
| Time series | DuckDB/ClickHouse | TimescaleDB |
| Full-text search | Whoosh/Elasticsearch | Meilisearch |

### Analysis Layer
| Need | Recommended | Alternative |
|------|-------------|-------------|
| Feature engineering | scikit-learn | Pandas |
| Anomaly detection | Isolation Forest | LOF |
| Graph neural networks | PyTorch Geometric | DGL |
| NLP | Transformers | spaCy |

### Visualization Layer
| Need | Recommended | Alternative |
|------|-------------|-------------|
| Dashboard | Streamlit | Dash |
| Charts | Pyecharts | Plotly |
| Network graphs | D3.js | Sigma.js |

## Best Practices

### Code Organization
```
project/
├── core/           # Core business logic
├── adapters/       # External integrations
├── models/         # Data models
├── services/       # Business services
├── utils/          # Shared utilities
└── tests/          # Test suites
```

### Error Handling
```python
# Always include comprehensive error handling
try:
    result = await operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Graceful degradation
    result = fallback_operation()
```

### Documentation
- Document architecture decisions
- Include API documentation
- Provide usage examples
- Maintain changelog

## Quick Reference

### Parallel Task Pattern
```python
# Identify independent tasks
independent_tasks = [task for task in tasks if not task.dependencies]

# Run in parallel
results = await asyncio.gather(*[
    sub_agent.execute(task) for task in independent_tasks
])
```

### Technology Research Pattern
```python
# When facing technical decision
search_queries = [
    f"{technology} best practices 2024",
    f"{technology} vs {alternative}",
    f"{technology} performance benchmark"
]
for query in search_queries:
    results = web_search(query)
    evaluate_options(results)
```

---

**Remember:** Always research before implementing, and parallelize when possible!
