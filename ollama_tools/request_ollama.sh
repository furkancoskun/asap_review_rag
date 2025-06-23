curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ],
  "stream": false
}'