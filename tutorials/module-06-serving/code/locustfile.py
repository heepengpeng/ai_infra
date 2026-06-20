"""locust 版压测:带 Web UI,实时看 RPS 与分位数,适合团队共享。

被测服务需为 OpenAI 兼容。流式请求下,locust 的 response time 统计到「整条流结束」,
若想单独统计 TTFT,可在 on_message 里打点(见 TODO)。
运行:locust -f locustfile.py --host http://localhost:8000
浏览器开 http://localhost:8089 设并发即可。
"""

import json
import logging

from locust import HttpUser, between, task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("locust_llm")

MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LLMUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def chat(self) -> None:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "用三句话介绍张量并行。"}],
            "max_tokens": 200,
            "stream": True,
            "temperature": 0.7,
        }
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True,
            name="chat_stream",
        ) as resp:
            tokens = 0
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode() if isinstance(line, bytes) else line
                if text.startswith("data: ") and not text.endswith("[DONE]"):
                    json.loads(text[len("data: ") :])
                    tokens += 1
            # TODO(进阶): 记录首个 chunk 到达时间作为 TTFT,用 events.request.fire 自定义上报,
            #   在 locust 里单独画 TTFT 分位数曲线。
            if tokens > 0:
                resp.success()
            else:
                resp.failure("no tokens received")
