FROM macroverse-cn-beijing.cr.volces.com/macroverse/ai-agent-base:v1

WORKDIR /root/deepddg

RUN tosutil cp -r -flat tos://dp-macroverse/lyh/deepddg/ .

COPY . .

RUN uv sync

ENTRYPOINT ["uv", "run", "test.py"]
