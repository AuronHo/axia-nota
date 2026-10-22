# Free hosting on a Hugging Face Space (CPU basic: 2 vCPU, 16 GB RAM).
# llama.cpp's server speaks the same OpenAI-style API the game uses and downloads your GGUF from the Hub at start.
# Replace AuronHo/bu-melati-1.5b-GGUF with your model repo (see README, "Host it for free").
FROM ghcr.io/ggml-org/llama.cpp:server

ENV LLAMA_CACHE=/tmp/llama-cache
EXPOSE 7860
CMD ["--host", "0.0.0.0", "--port", "7860", \
     "-hf", "AuronHo/bu-melati-1.5b-GGUF:Q4_K_M", \
     "--alias", "bu-melati", "-c", "4096", "--threads", "2", \
     "--parallel", "2", "-n", "400"]
